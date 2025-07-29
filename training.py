import argparse
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torch
import numpy as np

#from vae import VAE
from utils import save_image_grid, interpolate_latents

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vae')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--data_dir', type=str, default='./data/emojis')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def load_data(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_vae(args, dataloader):
    from torch import nn, optim
    model = VAE(latent_dim=args.latent_dim).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss(reduction='sum')

    os.makedirs(f"{args.output_dir}/reconstructions", exist_ok=True)
    os.makedirs(f"{args.output_dir}/generations", exist_ok=True)

    for epoch in range(args.epochs):
        total_loss = 0
        for imgs, _ in dataloader:
            imgs = imgs.to(args.device)
            recon, mu, logvar = model(imgs)
            recon_loss = loss_fn(recon, imgs)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} - Loss: {total_loss:.2f}")

        with torch.no_grad():
            sample = next(iter(dataloader))[0][:16].to(args.device)
            recon, _, _ = model(sample)
            save_image_grid(recon, f"{args.output_dir}/reconstructions/recon_epoch{epoch+1}.png")

            z = torch.randn(16, args.latent_dim).to(args.device)
            generated = model.decode(z)
            save_image_grid(generated, f"{args.output_dir}/generations/sample_epoch{epoch+1}.png")

    torch.save(model.state_dict(), f"{args.output_dir}/vae_model.pth")

def main():
    args = get_args()
    dataloader = load_data(args.data_dir, args.batch_size)
    train_vae(args, dataloader)

if __name__ == "__main__":
    main()
