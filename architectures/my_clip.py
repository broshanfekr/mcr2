
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyClip(nn.Module):
    def __init__(self, model_name="ViT-L/14", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clip_model, preprocess = clip.load(model_name, device=self.device)
        self.backbone = clip_model.visual

    def forward(self, x):
        feature = self.backbone(x)
        out = F.normalize(feature, 2)
        return out
