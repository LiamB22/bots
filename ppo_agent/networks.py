import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class COMBINED(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, cnn_features_dim=256, mlp_features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # ===== Enhanced CNN for board features =====
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space['board'].shape[0], 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 2)),  # Fixed output size
            nn.Flatten(),
        )
        
        # Fixed CNN output size calculation
        cnn_output_size = 256 * 3 * 2  # 1536
        
        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_output_size, cnn_features_dim),
            nn.LayerNorm(cnn_features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # ===== Enhanced MLP for numeric features =====
        self.mlp = nn.Sequential(
            nn.Linear(observation_space['numeric'].shape[0], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, mlp_features_dim),
            nn.LayerNorm(mlp_features_dim),
            nn.ReLU(),
        )
        
        # ===== Combine features with attention mechanism =====
        self.combine_fc = nn.Sequential(
            nn.Linear(cnn_features_dim + mlp_features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations):
        board_features = self.cnn(observations['board'])
        board_features = self.cnn_fc(board_features)
        
        numeric_features = self.mlp(observations['numeric'])
        
        combined = torch.cat([board_features, numeric_features], dim=1)
        return self.combine_fc(combined)