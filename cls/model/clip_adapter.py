import torch
import torch.nn as nn
from torch import Tensor
from timm.models.vision_transformer import VisionTransformer as timm_ViT
import timm
class CLIP_adapter(nn.Module):
    # paper: https://link.springer.com/article/10.1007/s11263-023-01891-x
    def __init__(self,
                vit_model: timm_ViT, alpha = 0.2):
        super(CLIP_adapter, self).__init__()
        vit_model.head = nn.Identity()
        self.alpha = alpha
        for param in vit_model.parameters():
            param.requires_grad = False

        self.dim = vit_model.blocks[0].attn.qkv.in_features
        self.adapter = nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim, bias=False),
        )

        self.backbone = vit_model

    def forward(self, x: Tensor) -> Tensor:
        f = self.backbone(x)
        f_a = self.adapter(f)
        return self.alpha * f_a + (1 - self.alpha) * f

def get_model(version="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", alpha=0.2):
    model = timm.create_model(version, pretrained=True)
    clip_adapter_model = CLIP_adapter(model, alpha=alpha)
    return clip_adapter_model
if __name__ == "__main__":
    adapter_model = get_model()
    x = torch.randn(1,3,224,224)
    y = adapter_model(x)
    print(y.size())
