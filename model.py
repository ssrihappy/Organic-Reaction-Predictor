"""
PyTorch 기반 반응 성공률 예측 모델
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReactionPredictor(nn.Module):
    """반응 성공률을 예측하는 신경망 모델"""
    
    def __init__(self, input_size: int, hidden_sizes: list = [512, 256, 128]):
        super(ReactionPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers with BatchNorm and Dropout
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layer (확률 출력: 0~1)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ReactionGAN(nn.Module):
    """
    GAN 스타일 모델 (향후 확장용)
    Generator: 반응 조건 생성
    Discriminator: 반응 성공 여부 판별
    """
    
    def __init__(self, input_size: int, latent_dim: int = 100):
        super(ReactionGAN, self).__init__()
        self.latent_dim = latent_dim
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def generate(self, z):
        return self.generator(z)
    
    def discriminate(self, x):
        return self.discriminator(x)
