import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision.models import resnet18, resnet50

logger = logging.getLogger(__name__)

class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()
        self.model = resnet50(pretrained=True)
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        features = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        features.append(x)
        x = self.model.layer2(x)
        features.append(x)
        x = self.model.layer3(x)
        features.append(x)
        x = self.model.layer4(x)
        features.append(x)
        return features

class Resnet_Encoder(nn.Module):
    def __init__(self):
        super(Resnet_Encoder, self).__init__()
        self.encoder = base_resnet()

    def forward(self, img):
        features = self.encoder(img)
        return features

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # mask:[B,1,L,L]
        att = att.masked_fill(mask == 0, float('-inf'))

        if x.dtype == torch.float16:
            att = att.to(torch.float32)
            fp16 = True
        else:
            fp16 = False
        att = F.softmax(att, dim=-1)
        if fp16:
            att = att.to(torch.float16)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            # nn.GELU(),  # nice, GELU is not valid in torch<1.6
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class Transformer_Prediction(nn.Module):
    def __init__(self, config):
        super(Transformer_Prediction, self).__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.gelu = GELU()
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = self.ln1(x)
        x = self.dense(x)
        x = self.gelu(x)
        x = self.ln2(x)
        return x
class PatchDecoder(nn.Module):
    """
    一个轻量级的卷积解码器，用于将低维 latent 向量还原为 16x16 的像素块。
    """
    def __init__(self, latent_dim=64, patch_size=16, start_dim=4, channels=64):
        """
        Args:
            latent_dim (int): 输入的 latent 向量维度。
            patch_size (int): 目标 patch 的边长 (例如 16)。
            start_dim (int): 卷积上采样的起始空间维度 (例如 4x4)。
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.start_dim = start_dim
        self.start_channels = channels

        # 1. 线性层，将 latent 向量映射到 CNN 的起始输入
        self.fc = nn.Linear(latent_dim, (start_dim ** 2) * self.start_channels)

        # 2. 卷积上采样 (ConvTranspose2d)
        self.decoder = nn.Sequential(
            # 当前: [B*L, 64, 4, 4]
            nn.ConvTranspose2d(self.start_channels, 32, kernel_size=4, stride=2, padding=1), # -> [B*L, 32, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # -> [B*L, 16, 16, 16]
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1) # -> [B*L, 1, 16, 16] (输出1通道)
        )
    def forward(self, x_latent):
        # x_latent 形状: [B*L, latent_dim] (B*L 是总的 patch 数量)
        # 1. 映射到 CNN 输入
        x = self.fc(x_latent) # -> [B*L, 4*4*64]
        # 2. Reshape 为空间图像格式
        x = x.view(x.shape[0], self.start_channels, self.start_dim, self.start_dim) # -> [B*L, 64, 4, 4]
        # 3. 卷积解码
        x_pixels = self.decoder(x) # -> [B*L, 1, 16, 16]       
        # 4. 展平为像素序列
        return x_pixels.view(x_pixels.shape[0], -1) # -> [B*L, 256] (即 16*16)

class GlobalDecoder(nn.Module):
    """
    将插值后的潜变量场 (e.g., [B, 64, 64, 64]) 解码为
    完整的粗糙掩码 (e.g., [B, 1, 256, 256])。
    """
    def __init__(self, latent_dim=64, start_res=64):
        super().__init__()
        # start_res 是您插值后的大小 (例如 64x64)
        
        self.decoder = nn.Sequential(
            # 输入: [B, 64, 64, 64]
            nn.Conv2d(latent_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 上采样到 128x128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 上采样到 256x256
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 输出 1 通道
            nn.Conv2d(8, 1, kernel_size=3, padding=1)
            # (注意：这里不加 Sigmoid，因为 z_loss 的 self.criterion = nn.MSELoss() 
            # 并且 refine_module 的输入也需要 logits)
        )

    def forward(self, smooth_field):
        # smooth_field: [B, 64, 64, 64]
        return self.decoder(smooth_field) # [B, 1, 256, 256]


class MaskedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_dim = config.n_embd
        
        # config中定义latent_dim
        self.latent_dim = getattr(config, 'latent_dim', 64)
        
        self.conv_in = torch.nn.Conv2d(2048, embedding_dim//2, 3, padding=1)
        
   
        
        # 新代码（MAE）：Linear 层可以处理连续的像素值
        self.vm_patch_emb = nn.Linear(256, embedding_dim//4)  # Visible mask patches: [B,256,256]->[B,256,D//4]
        self.fm_patch_emb = nn.Linear(256, embedding_dim//4)  # Amodal mask patches: [B,256,256]->[B,256,D//4]
        
        # posotion embedding
        self.pos_emb = nn.Embedding(config.sequence_length, embedding_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.dec = Transformer_Prediction(config)
        
     
        # 新代码：预测连续的像素值 [0, 1]
        #self.pixel_pred_head = nn.Linear(embedding_dim, 1)  # 输出 [B, 256, 1]
        self.pixel_pred_head = nn.Linear(embedding_dim, 256)
        
        
        self.sequence_length = config.sequence_length
        self.apply(self._init_weights)
        self.config = config

    def forward(self, img_feat, vm_patches, fm_patches, mask=None):
        
        i_embeddings = self.conv_in(img_feat) # [B, D//2, 16, 16]
        i_embeddings = i_embeddings.flatten(2).transpose(-2, -1)  # [B, 256, D//2]
        
        
        # ✅ 新代码：Linear 层处理连续值
        vm_embeddings = self.vm_patch_emb(vm_patches)  # [B, 256, D//4]
        fm_embeddings = self.fm_patch_emb(fm_patches)  # [B, 256, D//4]
        
        token_embeddings = torch.cat([i_embeddings, vm_embeddings, fm_embeddings], dim=2) # [B, 256, D]
        
        # add positional embeddings
        n_tokens = token_embeddings.shape[1] # 256
        position_ids = torch.arange(n_tokens, dtype=torch.long, device=vm_patches.device)
        position_ids = position_ids.unsqueeze(0).repeat(vm_patches.shape[0], 1) # [B, 256]
        position_embeddings = self.pos_emb(position_ids)                   # [B, 256, D]

        x = self.drop(token_embeddings + position_embeddings)

        batch_size = token_embeddings.shape[0]
        mask = torch.ones(batch_size, 1, n_tokens, n_tokens).cuda()

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.dec(x)
        
        # 新代码：输出 [B, 256, latent_dim] 像素值
        pred_pixels = self.pixel_pred_head(x) # [B, 256, 256]
        
        return pred_pixels # [修改] 返回预测的像素

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class Refine_Module(nn.Module):
    def __init__(self):
        super(Refine_Module, self).__init__()
        dim = 259  #(来自 conv_in) + 1 (来自 attn_map) + 2 (来自 coarse_mask + uncertainty_map)
        self.conv_adapter = torch.nn.Conv2d(2048, 2048, 1)
        self.conv_in = torch.nn.Conv2d(2048, 256, 3, padding=1)
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(dim)

        self.lay2 = torch.nn.Conv2d(dim, 128, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)

        self.lay3 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.adapter1 = torch.nn.Conv2d(1024, 128, 1)

        # visible mask branch
        self.lay4_vm = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.bn4_vm = torch.nn.BatchNorm2d(32)
        self.lay5_vm = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.bn5_vm = torch.nn.BatchNorm2d(16)
        self.adapter2_vm = torch.nn.Conv2d(512, 64, 1)
        self.adapter3_vm = torch.nn.Conv2d(256, 32, 1)
        self.out_lay_vm = torch.nn.Conv2d(16, 1, 3, padding=1)

        # amodal mask branch
        self.lay4_am = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.bn4_am = torch.nn.BatchNorm2d(32)
        self.lay5_am = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.bn5_am = torch.nn.BatchNorm2d(16)
        self.adapter2_am = torch.nn.Conv2d(512, 64, 1)
        self.adapter3_am = torch.nn.Conv2d(256, 32, 1)
        self.out_lay_am = torch.nn.Conv2d(16, 1, 3, padding=1)
    
    def get_attn_map(self, feature, guidance):
        b,c,h,w = guidance.shape
        q = torch.flatten(guidance, start_dim=2)
        v = torch.flatten(feature, start_dim=2)

        k = v * q
        k = k.sum(dim=-1, keepdim=True) / (q.sum(dim=-1, keepdim=True) + 1e-6)
        attn = (k.transpose(-2, -1) @  v) / 1
        attn = F.softmax(attn, dim=-1)
        attn = attn.reshape(b, c, h, w)
        return attn
    
    def forward(self, features, combined_refine_input):
        # features:    [B, 2048, 16,   16]
        # attn_map:    [B, 1,    16,   16]
        # coarse_mask: [B, 2,    256, 256]
        feat = self.conv_adapter(features[-1])

        # --- 1. 修改输入处理 ---
        # (a) 插值 2 通道输入
        coarse_maps_interp = F.interpolate(combined_refine_input, scale_factor=(1/16)) # -> [B, 2, 16, 16]
        
        # (b) attn_map 应该只由“掩码”引导
        #     我们从插值后的 2 通道张量中切片出第 0 个通道
        coarse_mask_interp = coarse_maps_interp[:, 0:1, :, :] # -> [B, 1, 16, 16]
        attn_map = self.get_attn_map(feat, coarse_mask_interp)
        # (c) 拼接 x (256), attn_map (1), 和 2 通道的 coarse_maps_interp (2)
        #     总通道数 = 256 + 1 + 2 = 259，匹配 __init__ 中的 dim


        x = self.conv_in(feat) # -> [B, 256, 16, 16]
        # --- (修正结束) ---
        x = torch.cat((x, attn_map, coarse_maps_interp), dim=1)

        
        x = F.relu(self.bn1(self.lay1(x)))
        
        x = F.relu(self.bn2(self.lay2(x)))
        
        cur_feat = self.adapter1(features[-2])
        x = cur_feat + x
        x = F.interpolate(x, size=(32, 32), mode="nearest")
        x = F.relu(self.bn3(self.lay3(x)))

        # TODO: visible mask branch
        cur_feat_vm = self.adapter2_vm(features[-3])
        x_vm = cur_feat_vm + x
        x_vm = F.interpolate(x_vm, size=(64, 64), mode="nearest")
        x_vm = F.relu(self.bn4_vm(self.lay4_vm(x_vm)))

        cur_feat_vm = self.adapter3_vm(features[-4])
        x_vm = cur_feat_vm + x_vm
        x_vm = F.interpolate(x_vm, size=(128, 128), mode="nearest")
        x_vm = F.relu(self.bn5_vm(self.lay5_vm(x_vm)))
        
        x_vm = self.out_lay_vm(x_vm)

        # TODO: full mask branch
        cur_feat_am = self.adapter2_am(features[-3])
        x_am = cur_feat_am + x
        x_am = F.interpolate(x_am, size=(64, 64), mode="nearest")
        x_am = F.relu(self.bn4_am(self.lay4_am(x_am)))

        cur_feat_am = self.adapter3_am(features[-4])
        x_am = cur_feat_am + x_am
        x_am = F.interpolate(x_am, size=(128, 128), mode="nearest")
        x_am = F.relu(self.bn5_am(self.lay5_am(x_am)))
        
        x_am = self.out_lay_am(x_am)

        return x_vm, x_am
    
