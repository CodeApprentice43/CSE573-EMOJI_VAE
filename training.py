import argparse
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torch
import numpy as np
from utils import save_image_grid, interpolate_latents
from vae import VAE

def get_args():
    parser = argparse.AbutrgumentParser()
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
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def loss_function(recon_x, x, mu, log_var):
    bce = torch.nn.functional.binary_cross_entropy(recon_x, x.view(-1, 1, 64, 64), reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kld

def train_vae(args, dataloader):
    from torch import optim
    model = VAE(latent_dim=args.latent_dim).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs(f"{args.output_dir}/reconstructions", exist_ok=True)
    os.makedirs(f"{args.output_dir}/generations", exist_ok=True)

    fixed_batch, _ = next(iter(dataloader))
    fixed_batch = fixed_batch.to(args.device)
    utils.save_image(fixed_batch, os.path.join(args.output_dir, 'original_images.png'))

    loss_history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for imgs, _ in dataloader:
            imgs = imgs.to(args.device)
            recon, mu, logvar = model(imgs)
            loss = loss_function(recon, imgs, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        loss_history.append(avg_loss)

        model.eval()
        with torch.no_grad():
            recon, _, _ = model(fixed_batch)
            utils.save_image(recon, f"{args.output_dir}/reconstructions/recon_epoch{epoch+1}.png")

            z = torch.randn(16, args.latent_dim).to(args.device)
            generated = model.decode(z)
            utils.save_image(generated, f"{args.output_dir}/generations/sample_epoch{epoch+1}.png")

    torch.save(loss_history, os.path.join(args.output_dir, 'loss_history.pt'))
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'vae_model.pt'))

def main():
    args = get_args()
    dataloader = load_data(args.data_dir, args.batch_size)
    train_vae(args, dataloader)

if __name__ == "__main__":
    main()
