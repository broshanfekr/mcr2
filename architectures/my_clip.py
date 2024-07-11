
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


# class MyClip(nn.Module):
#     def __init__(self, model_name="ViT-L/14", *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         clip_model, preprocess = clip.load(model_name, device=self.device)
#         self.backbone = clip_model.visual
#         self.preprocess = preprocess
        

#     def forward(self, x):
#         feature = self.backbone(x)
#         out = F.normalize(feature, 2)
#         return out


class MyClip(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        #store the backbone
        # self.backbone = backbone
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clip_model, preprocess = clip.load("ViT-L/14", device=self.device)
        self.backbone = clip_model.visual
        
        
        self.pre_feature = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                         nn.BatchNorm1d(hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU(),
                                         )
        self.subspace = nn.Sequential(
            nn.Linear(hidden_dim, z_dim)
        )
        
    def forward(self, x):
        with torch.no_grad():
            feature = self.backbone(x)
        
        pre_feature = self.pre_feature(feature)
        Z = self.subspace(pre_feature)
        Z = F.normalize(Z, 2)
        return Z