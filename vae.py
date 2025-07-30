import torch 
from torch import nn

class VAE(nn.Module):
    def __init__(self,latent_dim=32):
        super().__init__() #ensure that nn.module is initialized

        #the entire nn architecture is setup on the constructor

        #pipeline of layers of the encoder, each layer is a convolutional layer followed by a ReLU activation for non-linearity
        self.encoder = nn.Sequential(
            #dimensions below represent the reduction in spatial dimensions as we go deeper into the network
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten() # flatten the output to 4096 feed into the linear layers
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)  # output mu
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim) # output log variance

