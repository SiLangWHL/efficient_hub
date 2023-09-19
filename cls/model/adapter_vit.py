import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.vision_transformer import VisionTransformer as timm_ViT
import timm
class Adapter_ViT(nn.Module):
    """Applies mlp adapter to a vision transformer.

    Args:
        vit_model: a vision transformer model, see base_vit.py
        num_layers: number of hidden layers
        num_classes: how many classes the model output, default to the vit model

    Examples::
        >>> model = timm.create_model("vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True)
        >>> adapter_model = Adapter_ViT(model, r=4)
        >>> preds = adapter_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self,
                vit_model: timm_ViT,
                num_classes: int = 0):
        super(Adapter_ViT, self).__init__()

        for param in vit_model.parameters():
            param.requires_grad = False

        self.dim = vit_model.blocks[0].attn.qkv.in_features
        self.adapter = nn.Sequential()
        for t_layer_i in range(len(vit_model.blocks)//2):
            self.adapter.add_module("layer_" + str(t_layer_i), nn.Linear(self.dim, self.dim))
            self.adapter.add_module("relu_" + str(t_layer_i), nn.ReLU())

        self.backbone = vit_model
        self.backbone.head = self.adapter

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

def get_model(version="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k"):
    model = timm.create_model(version, pretrained=True)
    adapter_model = Adapter_ViT(model)
    return adapter_model

if __name__ == "__main__":
    adapter_model = get_model()
    x = torch.randn(1,3,224,224)
    y = adapter_model(x)
    print(y.size())
