import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms

from taming_src.taming_models import VQModel
from src.image_component import MaskedTransformer, Resnet_Encoder, Refine_Module,PatchDecoder,GlobalDecoder
from src.loss import VGG19, PerceptualLoss
from utils.pytorch_optimization import AdamW, get_linear_schedule_with_warmup
from utils.utils import torch_show_all_params, torch_init_model
from utils.utils import Config
from utils.evaluation import evaluation_image
from utils.loss import CrossEntropyLoss


class C2F_Seg(nn.Module):
    def __init__(self, config, g_path=None, mode=None, logger=None, save_eval_dict={}):
        
        super(C2F_Seg, self).__init__()
        self.config = config
        self.iteration = 0
        self.sample_iter = 0
        self.name = config.model_type
        self.latent_dim = getattr(config, 'latent_dim', 64)
        self.z_loss_weight = getattr(config, 'z_loss_weight', 5.0)
        # 如果传入了 g_path，给出警告（向后兼容）
        if g_path is not None and logger is not None:
            logger.warning("⚠️ MAE version does not use VQ-VAE. Parameter 'g_path' is ignored.")

        self.root_path = config.path
        self.transformer_path = os.path.join(config.path, self.name)

        self.mode = mode
        self.save_eval_dict = save_eval_dict

        self.eps = 1e-6
        self.train_sample_iters = config.train_sample_iters
        
        

        self.img_encoder = Resnet_Encoder().to(config.device)
        self.refine_module = Refine_Module().to(config.device)
        self.transformer = MaskedTransformer(config).to(config.device)
        
        

        self.coarse_bce_loss = nn.BCEWithLogitsLoss()
        self.refine_criterion = nn.BCEWithLogitsLoss()
        self.refine_criterion_pointwise = nn.BCEWithLogitsLoss(reduction='none')
        self.criterion = nn.MSELoss(reduction='none') 

        # ========== VGG 感知损失（如果需要）==========
        if config.train_with_dec:
            if not config.gumbel_softmax:
                self.temperature = nn.Parameter(torch.tensor([config.tp], dtype=torch.float32),
                                                requires_grad=True).to(config.device)
            if config.use_vgg:
                vgg = VGG19(pretrained=True, vgg_norm=config.vgg_norm).to(config.device)
                vgg.eval()
                reduction = 'mean' if config.balanced_loss is False else 'none'
                self.perceptual_loss = PerceptualLoss(vgg, weights=config.vgg_weights,
                                                      reduction=reduction).to(config.device)
        else:
            self.perceptual_loss = None
    

        if logger is not None:
            logger.info('Transformer Parameters:{}'.format(torch_show_all_params(self.transformer)))
            logger.info('Image Encoder Parameters:{}'.format(torch_show_all_params(self.img_encoder)))
            logger.info('Refine Module Parameters:{}'.format(torch_show_all_params(self.refine_module)))
        else:
            print('Transformer Parameters:{}'.format(torch_show_all_params(self.transformer)))
            print('Image Encoder Parameters:{}'.format(torch_show_all_params(self.img_encoder)))
            print('Refine Module Parameters:{}'.format(torch_show_all_params(self.refine_module)))

        # loss
        no_decay = ['bias', 'ln1.bias', 'ln1.weight', 'ln2.bias', 'ln2.weight']
        param_optimizer = self.transformer.named_parameters()
        param_optimizer_encoder = self.img_encoder.named_parameters()
        param_optimizer_refine= self.refine_module.named_parameters()
        
        
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any([nd in n for nd in no_decay])],
            'weight_decay': config.weight_decay},
            {'params': [p for n, p in param_optimizer if any([nd in n for nd in no_decay])],
            'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer_encoder], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in param_optimizer_refine], 'weight_decay': config.weight_decay},
            
        ]

        self.opt = AdamW(params=optimizer_parameters,
                         lr=float(config.lr), betas=(config.beta1, config.beta2))
        self.sche = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_iters,
                                                    num_training_steps=config.max_iters)

        self.rank = dist.get_rank()
        # --- 2. 新增：从 config 加载损失权重 ---
        # (您需要在 .yml 文件中添加这些超参数, 否则默认为 1.0)
        self.point_loss_weight = getattr(config, 'point_loss_weight', 1.0)
        # --- (结束) ---
        self.gamma = self.gamma_func(mode=config.gamma_mode)
        
        self.choice_temperature = 4.5
        self.Image_W = config.Image_W
        self.Image_H = config.Image_H
        self.patch_W = config.patch_W
        self.patch_H = config.patch_H

   
    def extract_patches(self, masks, patch_size=16):
        patches_unfolded = F.unfold(
            masks.float(), 
            kernel_size=patch_size, 
            stride=patch_size
        )  # [B, 1*16*16, 16*16] = [B, 256, 256]
        
        patches = patches_unfolded.transpose(1, 2)  # [B, 256, 256]
        return patches
    
    def get_attn_map(self, feature, guidance):
        guidance = F.interpolate(guidance, scale_factor=(1/16))
        b,c,h,w = guidance.shape
        q = torch.flatten(guidance, start_dim=2)
        v = torch.flatten(feature, start_dim=2)

        k = v * q
        k = k.sum(dim=-1, keepdim=True) / (q.sum(dim=-1, keepdim=True) + 1e-6)
        attn = (k.transpose(-2, -1) @  v) / 1
        attn = F.softmax(attn, dim=-1)
        attn = attn.reshape(b, c, h, w)
        return attn

    def get_losses(self, meta):
        self.iteration += 1
        
        img_feat = self.img_encoder(meta['img_crop'].permute((0,3,1,2)).to(torch.float32))
        B = meta['vm_crop'].shape[0]
        patch_size = 16
        num_patches_side = 16 # (256 // 16)
        
        # 1. 提取 Patches (目标和输入)
        vm_patches_input = self.extract_patches(meta['vm_crop'], patch_size) 
        fm_patches_input = self.extract_patches(meta['fm_crop'], patch_size) 
        
        #  目标(Target)是完整的像素块
        target = fm_patches_input # 形状: [B, 256, 256]
        
        num_patches = target.shape[1]  # 256
        if num_patches != 256:
             raise ValueError(f"Dataloader returned patch sequence of length {num_patches}, expected 256.")

        r = np.maximum(self.gamma(np.random.uniform()), self.config.min_mask_rate)
        r = math.floor(r * num_patches)

        sample = torch.rand(B, num_patches, device=target.device).topk(r, dim=1).indices
        random_mask = torch.zeros(B, num_patches, dtype=torch.bool, device=target.device)
        random_mask.scatter_(dim=1, index=sample, value=True)

        # 3. 创建 Masked Input (z_indices_input)
        mask_expanded = random_mask.unsqueeze(-1)  # [B, 256, 1]
        
        z_indices_input = torch.where(
            mask_expanded,
            torch.zeros_like(fm_patches_input),
            fm_patches_input
        )
        #5. Transformer 预测 Logits 
        pred_pixel_patches = self.transformer(
            img_feat[-1],
            vm_patches_input,
            z_indices_input,
            mask=None
        ) # Shape: [B, 256, 256] (现在是 Logits)
        
        #  6. 重建粗糙的 Logits 掩码
        patches_for_fold = pred_pixel_patches.transpose(1, 2)
        
        coarse_pred_fm_logits = F.fold(
            patches_for_fold,
            output_size=(256, 256),
            kernel_size=patch_size, 
            stride=patch_size      
        ) # Shape: [B, 1, 256, 256] (这现在是 *真正* 的 Logits)

        #7. 计算 Coarse 阶段的 BCE 损失
        #    我们直接在 Logits 上计算损失，目标是 0/1 掩码
        coarse_loss = self.coarse_bce_loss(coarse_pred_fm_logits, meta['fm_crop'])
    

        # 8. 为 Refine 阶段准备输入 [FIXED]
        #  移除 .detach() 和 with torch.no_grad()
        # Transformer 输出的是 Logits, 所以 sigmoid() 是正确的
        coarse_pred_fm_sig = torch.sigmoid(coarse_pred_fm_logits)
        
        #平滑 F.fold 产生的粗糙图 (概率图)
        coarse_pred_fm_sig_smooth = F.avg_pool2d(coarse_pred_fm_sig, kernel_size=5, stride=1, padding=2)
        
        # 8b.  计算并平滑不确定性图
        #    梯度将从 uncertainty_map 流回 coarse_pred_fm_sig -> transformer
        uncertainty_map = 1.0 - torch.abs(coarse_pred_fm_sig - 0.5) * 2.0
        uncertainty_map_smooth = F.avg_pool2d(uncertainty_map, kernel_size=5, stride=1, padding=2)

        # 9. Refine 阶段 [FIXED]
        #传入带梯度的平滑输入
        #   梯度将从 refine_module 流回 combined_refine_input -> transformer
        combined_refine_input = torch.cat([coarse_pred_fm_sig_smooth, uncertainty_map_smooth], dim=1)
        pred_vm_crop, pred_fm_crop = self.refine_module(img_feat, combined_refine_input)
        
        # 10. 计算 Refine Loss (使用 BCEWithLogitsLoss)
        pred_vm_crop = F.interpolate(pred_vm_crop, size=(256, 256), mode="bilinear", align_corners=False)
        loss_vm = self.refine_criterion(pred_vm_crop, meta['vm_crop_gt']) # 直接在 logits 上计算

        pred_fm_crop = F.interpolate(pred_fm_crop, size=(256, 256), mode="bilinear", align_corners=False)
        loss_fm_bce = self.refine_criterion(pred_fm_crop, meta['fm_crop']) # 直接在 logits 上计算
        
        point_loss_map = self.refine_criterion_pointwise(pred_fm_crop, meta['fm_crop'])
        uncertain_weights = uncertainty_map_smooth.detach() # .detach() 在这里是正确的
        loss_fm_point = (point_loss_map * uncertain_weights).sum() / (uncertain_weights.sum() + 1e-6)
        
        loss_fm = loss_fm_bce + self.point_loss_weight * loss_fm_point
        
        # 11. 【修改】更新日志
        logs = [
            ("z_loss_coarse_bce", coarse_loss.item()), # 新的 Coarse 损失
            ("loss_vm", loss_vm.item()),
            ("loss_fm_bce", loss_fm_bce.item()),
            ("loss_fm_point", (self.point_loss_weight * loss_fm_point).item()),
        ]
        
        total_refine_loss = loss_vm + loss_fm
        
        # 12. 【修改】返回加权后的总损失
        #    z_loss_weight 现在用于 Coarse BCE 损失
        total_z_loss = self.z_loss_weight * coarse_loss
        
        return total_z_loss, total_refine_loss, logs
    
    
    def align_raw_size(self, full_mask, obj_position, vm_pad, meta):
        vm_np_crop = meta["vm_no_crop"].squeeze()
        H, W = vm_np_crop.shape[-2], vm_np_crop.shape[-1]
        bz, seq_len = full_mask.shape[:2]
        new_full_mask = torch.zeros((bz, seq_len, H, W)).to(torch.float32).cuda()
        if len(vm_pad.shape)==3:
            vm_pad = vm_pad[0]
            obj_position = obj_position[0]
        for b in range(bz):
            paddings = vm_pad[b]
            position = obj_position[b]
            new_fm = full_mask[
                b, :,
                :-int(paddings[0]) if int(paddings[0]) !=0 else None,
                :-int(paddings[1]) if int(paddings[1]) !=0 else None
            ]
            vx_min = int(position[0])
            vx_max = min(H, int(position[1])+1)
            vy_min = int(position[2])
            vy_max = min(W, int(position[3])+1)
            resize = transforms.Resize([vx_max-vx_min, vy_max-vy_min])
            try:
                new_fm = resize(new_fm)
                new_full_mask[b, :, vx_min:vx_max, vy_min:vy_max] = new_fm[0]
            except:
                new_fm = new_fm
        return new_full_mask

    def loss_and_evaluation(self, pred_fm, meta, iter, mode, pred_vm=None, pred_fm_coarse=None):
        loss_eval = {}
        pred_fm = pred_fm.squeeze()
        counts = meta["counts"].reshape(-1).to(pred_fm.device)
        fm_no_crop = meta["fm_no_crop"].squeeze()
        vm_no_crop = meta["vm_no_crop"].squeeze()
        pred_vm = pred_vm.squeeze()

        # --- [新增：计算粗糙掩码的IoU] ---
        if pred_fm_coarse is not None:
            pred_fm_coarse_squeezed = pred_fm_coarse.squeeze()
            pred_fm_coarse_int = (pred_fm_coarse_squeezed > 0.5).to(torch.int64)
            # 调用 evaluation_image 来计算粗糙掩码的指标
            iou_coarse, inv_iou_coarse, occ_count_coarse = evaluation_image(
                pred_fm_coarse_int, fm_no_crop, counts, meta, self.save_eval_dict
            )
            # 将新指标存入 loss_eval 字典
            loss_eval["iou_coarse"] = iou_coarse
            loss_eval["invisible_iou_coarse"] = inv_iou_coarse
        # post-process
        pred_fm = (pred_fm > 0.5).to(torch.int64)
        pred_vm = (pred_vm > 0.5).to(torch.int64)
        
        iou, invisible_iou_, iou_count = evaluation_image((pred_fm > 0.5).to(torch.int64), fm_no_crop, counts, meta, self.save_eval_dict)
        loss_eval["iou"] = iou
        loss_eval["invisible_iou_"] = invisible_iou_
        loss_eval["occ_count"] = iou_count
        loss_eval["iou_count"] = torch.Tensor([1]).cuda()
        pred_fm_post = pred_fm + vm_no_crop
        
        pred_fm_post = (pred_fm_post>0.5).to(torch.int64)
        iou_post, invisible_iou_post, iou_count_post = evaluation_image(pred_fm_post, fm_no_crop, counts, meta, self.save_eval_dict)
        loss_eval["iou_post"] = iou_post
        loss_eval["invisible_iou_post"] = invisible_iou_post
        return loss_eval

    def backward(self, loss=None):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.sche.step()

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def batch_predict_maskgit(self, meta, iter, mode, T=3, start_iter=0):
        self.sample_iter += 1
        # ========== 1. 初始化 ==========
        img_feat = self.img_encoder(meta['img_crop'].permute((0,3,1,2)).to(torch.float32))
        B = meta['vm_crop'].shape[0]
        patch_size = 16
        num_patches_side = 16 # (256 // 16)
        # 输入：可见掩码（像素）

        vm_patches = self.extract_patches(meta['vm_crop'], patch_size)  # [B, 256, 256]

        # ========== 2. 创建全掩码输入 (新) ==========

        fm_patches_input = torch.zeros_like(vm_patches)  # [B, 256, 256]
       # 1. Transformer 预测完整的像素 patches

        pred_pixel_patches = self.transformer(
            img_feat[-1],
            vm_patches,         # 可见掩码 (作为引导)
            fm_patches_input,   # 全 0 的画布
            mask=None
        ) # Shape: [B, 256, 256]

        # 3. 使用 F.fold 重建粗糙掩码
        coarse_pred_fm_logits = F.fold(
            pred_pixel_patches.transpose(1, 2), # [B, 256, 256] -> [B, 256, 256]
            output_size=(256, 256), 
            kernel_size=16, 
            stride=16
        ) # [B, 1, 256, 256]
        # 4. 应用 Sigmoid 得到粗糙掩码 (概率图)
        # 这就是我们用于 Refine 模块的输入
        pred_fm_crop_old = torch.sigmoid(coarse_pred_fm_logits)
        # ========== 4. PLUG 不确定性引导 (不变) ==========
        # (计算高分辨率的不确定性图)
        uncertainty_map = 1.0 - torch.abs(pred_fm_crop_old - 0.5) * 2.0
        # 拼接为 2 通道输入
        combined_refine_input = torch.cat([pred_fm_crop_old.detach(), uncertainty_map.detach()], dim=1)
        # ========== 5. Refine 阶段 (不变) ==========
        pred_vm_crop, pred_fm_crop = self.refine_module(img_feat, combined_refine_input)
        # # ========== 6. 损失计算与评估 (不变) ==========
        # pred_vm_crop = F.interpolate(pred_vm_crop, size=(256, 256),  mode='bilinear', align_corners=False)
        # pred_vm_crop = torch.sigmoid(pred_vm_crop)
        # loss_vm = self.refine_criterion(pred_vm_crop, meta['vm_crop_gt'])
        # pred_fm_crop = F.interpolate(pred_fm_crop, size=(256, 256), mode='bilinear', align_corners=False)
        # pred_fm_crop = torch.sigmoid(pred_fm_crop)
        # loss_fm = self.refine_criterion(pred_fm_crop, meta['fm_crop'])
        # --- 6a. 在 Logits 上计算损失 ---
        pred_vm_logits = F.interpolate(pred_vm_crop, size=(256, 256),  mode='bilinear', align_corners=False)
        loss_vm = self.refine_criterion(pred_vm_logits, meta['vm_crop_gt'])
        pred_fm_logits = F.interpolate(pred_fm_crop, size=(256, 256), mode='bilinear', align_corners=False)
        loss_fm = self.refine_criterion(pred_fm_logits, meta['fm_crop'])
        # --- 6b. 现在应用 Sigmoid 得到概率，用于可视化和评估 ---
        pred_vm_crop = torch.sigmoid(pred_vm_logits)
        pred_fm_crop = torch.sigmoid(pred_fm_logits)
        pred_vm = self.align_raw_size(pred_vm_crop, meta['obj_position'], meta["vm_pad"], meta)
        pred_fm = self.align_raw_size(pred_fm_crop, meta['obj_position'], meta["vm_pad"], meta)
        pred_fm_coarse_aligned = self.align_raw_size(pred_fm_crop_old, meta['obj_position'], meta["vm_pad"], meta)
        self.visualize(pred_vm, pred_fm, meta, mode, iter)
        loss_eval = self.loss_and_evaluation(pred_fm, meta, iter, mode, pred_vm=pred_vm,pred_fm_coarse=pred_fm_coarse_aligned)
        loss_eval["loss_fm"] = loss_fm
        loss_eval["loss_vm"] = loss_vm
        return loss_eval
    def visualize(self, pred_vm, pred_fm, meta, mode, iteration):
        pred_fm = pred_fm.squeeze()
        pred_vm = pred_vm.squeeze()
        gt_vm = meta["vm_no_crop"].squeeze()
        gt_fm = meta["fm_no_crop"].squeeze()
        to_plot = torch.cat((pred_vm, pred_fm, gt_vm, gt_fm)).cpu().numpy()
        save_dir = os.path.join(self.root_path, '{}_samples'.format(mode))
        image_id, anno_id= meta["img_id"], meta["anno_id"]
        plt.imsave("{}/{}_{}_{}.png".format(save_dir, iteration, int(image_id.item()), int(anno_id.item())), to_plot)
    
    # def visualize_crop(self, pred_vm, pred_fm, meta, mode, count, pred_fm_crop_old):
    #     pred_fm = pred_fm.squeeze()
    #     pred_vm = pred_vm.squeeze()
    #     pred_fm_crop_old = pred_fm_crop_old.squeeze()
    #     gt_vm = meta["vm_crop"].squeeze()
    #     gt_fm = meta["fm_crop"].squeeze()
    #     to_plot = torch.cat((pred_vm, gt_vm, pred_fm_crop_old, pred_fm, gt_fm)).cpu().numpy()
    #     save_dir = os.path.join(self.root_path, '{}_samples'.format(mode))
    #     image_id, anno_id= meta["img_id"], meta["anno_id"]
    #     plt.imsave("{}/{}_{}_{}_{}.png".format(save_dir, count, int(image_id.item()), int(anno_id.item()), "crop"), to_plot)

    def create_inputs_tokens_normal(self, num, device):
        self.num_latent_size = self.config['resolution'] // self.config['patch_size']
        blank_tokens = torch.ones((num, self.num_latent_size ** 2), device=device)
        masked_tokens = self.mask_token_idx * blank_tokens

        return masked_tokens.to(torch.int64)

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        elif mode == "log":
            return lambda r, total_unknown: - np.log2(r) / np.log2(total_unknown)
        else:
            raise NotImplementedError

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(probs.device)
        sorted_confidence, _ = torch.sort(confidence, dim=-1) # from small to large
        # Obtains cut off threshold given the mask lengths.
        # cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        cut_off = sorted_confidence.gather(dim=-1, index=mask_len.to(torch.long))
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking
    
    def load(self, is_test=False, prefix=None):
        if prefix is not None:
            transformer_path = self.transformer_path + prefix + '.pth'
        else:
            transformer_path = self.transformer_path + '_last.pth'
        if self.config.restore or is_test:
            if os.path.exists(transformer_path):
                print('Rank {} is loading {} Transformer...'.format(self.rank, transformer_path))
                data = torch.load(transformer_path, map_location="cpu")
                
                torch_init_model(self.transformer, transformer_path, 'model')
                torch_init_model(self.img_encoder, transformer_path, 'img_encoder')
                torch_init_model(self.refine_module, transformer_path, 'refine')
                
                if self.config.restore:
                    self.opt.load_state_dict(data['opt'])
                    # skip sche
                    from tqdm import tqdm
                    for _ in tqdm(range(data['iteration']), desc='recover sche...'):
                        self.sche.step()
                self.iteration = data['iteration']
                self.sample_iter = data['sample_iter']
            else:
                print(transformer_path, 'not Found')
                raise FileNotFoundError

    def restore_from_stage1(self, prefix=None):
        if prefix is not None:
            g_path = self.g_path + prefix + '.pth'
        else:
            g_path = self.g_path + '_last.pth'
        if os.path.exists(g_path):
            print('Rank {} is loading {} G Mask ...'.format(self.rank, g_path))
            torch_init_model(self.g_model, g_path, 'g_model')
        else:
            print(g_path, 'not Found')
            raise FileNotFoundError
    
    def save(self, prefix=None):
        if prefix is not None:
            save_path = self.transformer_path + "_{}.pth".format(prefix)
        else:
            save_path = self.transformer_path + ".pth"

        print('\nsaving {} {}...\n'.format(self.name, prefix))
        torch.save({
            'iteration': self.iteration,
            'sample_iter': self.sample_iter,
            'model': self.transformer.state_dict(),
            'img_encoder': self.img_encoder.state_dict(),
            'refine': self.refine_module.state_dict(),
            'opt': self.opt.state_dict(),
        }, save_path)

        
