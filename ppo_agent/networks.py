import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNN(BaseFeaturesExtractor):
    
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # input shape: (C, H, W) = (channels, height, width) = (2*N + 12, 21, 11)
        self.cnn = nn.Sequential(
            # First convolutional block
            nn.Conv2d(observation_space['board'].shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Flatten
            nn.Flatten(),
        )
        
        # Calculate the flattened size dynamically
        with torch.no_grad():
            sample_input = torch.randn(1, *observation_space['board'].shape)
            flattened_size = self.cnn(sample_input).shape[1]
        
        # Final linear layer to get desired feature dimension
        self.linear = nn.Linear(flattened_size, features_dim)
        
    def forward(self, observations):
        # Extract board observations from the dictionary
        board_obs = observations['board']
        
        # Pass through CNN
        cnn_features = self.cnn(board_obs)
        
        # Final linear layer
        return self.linear(cnn_features)

class COMBINED(BaseFeaturesExtractor):
    """
    Handles both board (CNN) and numeric (MLP) features
    """
    
    def __init__(self, observation_space, features_dim=512, cnn_features_dim=256, mlp_features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # ===== CNN for board features =====
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space['board'].shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
        )
        
        # Calculate CNN output size dynamically
        with torch.no_grad():
            sample_board = torch.randn(1, *observation_space['board'].shape)
            cnn_output_size = self.cnn(sample_board).shape[1]
        
        self.cnn_fc = nn.Linear(cnn_output_size, cnn_features_dim)
        
        # ===== MLP for numeric features =====
        self.mlp = nn.Sequential(
            nn.Linear(observation_space['numeric'].shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, mlp_features_dim),
            nn.ReLU(),
        )
        
        # ===== Combine features =====
        self.combine_fc = nn.Linear(cnn_features_dim + mlp_features_dim, features_dim)
        
    def forward(self, observations):
        # Extract board features
        board_features = self.cnn(observations['board'])
        board_features = self.cnn_fc(board_features)
        
        # Extract numeric features  
        numeric_features = self.mlp(observations['numeric'])
        
        # Concatenate and combine
        combined = torch.cat([board_features, numeric_features], dim=1)
        return self.combine_fc(combined)