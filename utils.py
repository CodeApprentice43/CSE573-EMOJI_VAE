from torchvision.utils import save_image
import torch
import os

def save_image_grid(tensor, filename, nrow=4):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_image(tensor, filename, nrow=nrow, normalize=True)

def interpolate_latents(z1, z2, model, steps=10):
    z_interp = torch.stack([z1 * (1 - t) + z2 * t for t in torch.linspace(0, 1, steps)])
    return model.decode(z_interp)
