import torch
import torch.nn as nn
from torch.jit import Final
import timm
import math
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, RmsNorm, PatchDropout, use_fused_attn, SwiGLUPacked
from itertools import repeat
import collections.abc

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class svdMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
    def svd_forward(self, x, w, b):
        u, s, v = torch.svd(w)
        w_svd = torch.mm(torch.mm(u.detach(), torch.diag(s)), v.detach().t())
        x = torch.matmul(x, w_svd.T) 
        if b is not None:
            x += b.unsqueeze(0)
        return x
    def forward(self, x):
        #print("this is svd_mlp ", x.size())
        #x = self.fc1(x)
        x = self.svd_forward(x, self.fc1.weight, self.fc1.bias) if self.training else self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        #x = self.fc2(x)
        x = self.svd_forward(x, self.fc2.weight, self.fc2.bias) if self.training else self.fc2(x)
        x = self.drop2(x)
        return x
class svdAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def svd_forward(self, x, w, b):
        u, s, v = torch.svd(w)
        w_svd = torch.mm(torch.mm(u.detach(), torch.diag(s)), v.detach().t())
        x = torch.matmul(x, w_svd.T) 
        if b is not None:
            x += b.unsqueeze(0)
        return x
    
    def forward(self, x):
        #print("this is svd attention", x.size(), self.qkv.in_features)
        B, N, C = x.shape
        #qkv = self.qkv(x)
        qkv = self.svd_forward(x, self.qkv.weight, self.qkv.bias) if self.training else self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
 
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class SVD_ViT(VisionTransformer):
    def __init__(self,):
        super(SVD_ViT, self).__init__(pre_norm=True)  # if you want to load CLIP ViT, please set pre_norm=True; otherwise set pr_norm=False
        for t_layer_i, blk in enumerate(self.blocks):
            
            num_heads, dim = blk.attn.num_heads, blk.attn.qkv.in_features
            bias = True if blk.attn.qkv.bias is not None else False
            self.blocks[t_layer_i].attn = svdAttention(dim=dim, num_heads=num_heads, qkv_bias=bias)
            
            in_features, hidden_features, out_features = blk.mlp.fc1.in_features, blk.mlp.fc1.out_features, blk.mlp.fc2.out_features
            bias = True if blk.mlp.fc1.bias is not None else False
            self.blocks[t_layer_i].mlp = svdMlp(in_features, hidden_features, out_features, bias=bias)
            
        self.head = nn.Identity()
class ViT(VisionTransformer): # if you want to load CLIP ViT, please set pre_norm=True; otherwise set pre_norm=False
    def __init__(self,):
        super(ViT, self).__init__(pre_norm=True) 
        self.head = nn.Identity()
            
def get_model(version="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k"):
    model_pretrain = timm.create_model(version, pretrained=True)
    model = SVD_ViT()
    model.load_state_dict(model_pretrain.state_dict(), False)
    return model

if __name__ == "__main__":  # Debug
    img = torch.randn(1, 3, 224, 224)
    vit = get_model().train()
    pred = vit(img)
    print(pred.shape)
