import torch 
from torch import nn


class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__() #ensure that nn.module is initialized

        #the entire nn architecture is setup on the constructor

        #pipeline of layers of the encoder, each layer is a convolutional layer followed by a ReLU activation for non-linearity
        self.encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 4x4 -> 2x2
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Flatten() # flatten the output to 512*2*2 to feed into the linear layers
        )

        self.fc_mu = nn.Linear(512 * 2 * 2, latent_dim)     # output mu
        self.fc_logvar = nn.Linear(512 * 2 * 2, latent_dim) # output log variance

        self.decoder_input = nn.Linear(latent_dim, 512 * 2 * 2) #linear layer to transform latent space back to the shape of the last conv layer output

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 2, 2)), # reshape the flat vector into a grid (512, 2, 2) giving us 512 stack of 2x2 feature maps

            # the bottom layers upsample the image back to the original size with 1 feature map (in channels, out channels)

            # 2x2 -> 4x4
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),

            nn.Sigmoid() # output layer with sigmoid activation to normalize the pixel values between 0 and 1
        )

    #encodes the image into a latent space representation and returns the mean and log variance
    def encode(self, x):
        x = self.encoder(x)             
        mu = self.fc_mu(x)               
        log_var = self.fc_logvar(x)    
        return mu, log_var

    #reparameterization trick to sample from the latent space to produce a latent vector z
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)   
        eps = torch.randn_like(std)     
        return mu + eps * std           

    #decodes the latent vector z back to the original image space
    def decode(self, z):
        x = self.decoder_input(z)       
        x = self.decoder(x)             
        return x

    #forward pass through the VAE model
    def forward(self, x):
        mu, log_var = self.encode(x)    
        z = self.reparameterize(mu, log_var) 
        recon = self.decode(z)          
        return recon, mu, log_var