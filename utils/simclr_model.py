import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ProjectionHead(nn.Module):
    """Projection head for SimCLR - Paper Exact Configuration
    
    From the paper: "We use a MLP with one hidden layer to obtain z_i = g(h_i) = W^(2)σ(W^(1)h_i)
    where σ is a ReLU non-linearity."
    
    For ResNet-50: 2048 -> 2048 -> 128
    For ResNet-18: 512 -> 512 -> 128
    """
    def __init__(self, input_dim=2048, hidden_dim=None, output_dim=128):
        super().__init__()
        # Paper uses same hidden dim as input dim
        if hidden_dim is None:
            hidden_dim = input_dim
            
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection_head(x)

class SimCLRModel(nn.Module):
    """Complete SimCLR model - Paper Exact Configuration
    
    Consists of:
    1. Base encoder (ResNet backbone)
    2. Projection head (MLP)
    
    The base encoder extracts representations, and the projection head
    maps them to a space where contrastive learning is performed.
    """
    def __init__(self, base_model='resnet50', out_dim=128):
        super().__init__()
        
        # Load ResNet and remove the final classification layer
        if base_model == 'resnet18':
            self.backbone = models.resnet18(pretrained=False)
            self.backbone.fc = nn.Identity()  # Remove final layer
            feature_dim = 512
        elif base_model == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            self.backbone.fc = nn.Identity()
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {base_model}")
        
        # Add projection head with paper-exact dimensions
        self.projection_head = ProjectionHead(feature_dim, feature_dim, out_dim)
    
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply projection head
        projections = self.projection_head(features)
        
        return features, F.normalize(projections, p=2, dim=-1)