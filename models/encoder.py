import torch
import torch.nn as nn
import torchvision.models as models

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class EncoderCNN(nn.Module):
    def __init__(self, embed_size=config.EMBED_SIZE):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Output: (batch, 2048, 7, 7) → 49 spatial regions per image
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze ResNet weights
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Project each of the 49 regions from 2048 → embed_size
        self.fc      = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, images):
        """
        images  : (batch, 3, 224, 224)
        returns : (batch, 49, embed_size)  ← 49 spatial regions
        """
        with torch.no_grad():
            features = self.resnet(images)          # (batch, 2048, 7, 7)

        batch = features.size(0)
        features = features.permute(0, 2, 3, 1)    # (batch, 7, 7, 2048)
        features = features.view(batch, -1, features.size(-1))  # (batch, 49, 2048)
        features = self.dropout(self.relu(self.fc(features)))   # (batch, 49, embed_size)
        return features