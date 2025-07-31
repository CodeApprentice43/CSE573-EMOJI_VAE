import argparse
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch import optim
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from utils import save_image_grid, interpolate_latents
from vae import VAE

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vae')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default='./data/training_data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def load_data(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def loss_function(recon_x, x, mu, log_var):
    bce = torch.nn.functional.binary_cross_entropy(recon_x, x.view(-1, 1, 64, 64), reduction='sum') #use sum reduction for higher loss gradient
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kld #total loss sum

#for computing additional metrics like MAE, SSIM, and PSNR as shown in the sample doc
def compute_metrics(recon, real):
    recon_np = recon.detach().cpu().numpy()
    real_np = real.detach().cpu().numpy()
    mae = np.mean(np.abs(recon_np - real_np))
    ssim = np.mean([ssim_metric(real_np[i, 0], recon_np[i, 0], data_range=1.0) for i in range(real_np.shape[0])])
    psnr = np.mean([psnr_metric(real_np[i, 0], recon_np[i, 0], data_range=1.0) for i in range(real_np.shape[0])])
    return mae, ssim, psnr

#plot the loss and metrics trends
def plot_loss_and_metrics(loss_history, mae_list, ssim_list, psnr_list, output_dir):
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 4, 1)
    plt.plot(loss_history)
    plt.title("VAE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()

    plt.subplot(1, 4, 2)
    plt.plot(mae_list, '-o', color='orange')
    plt.title("MAE Trend")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")

    plt.subplot(1, 4, 3)
    plt.plot(ssim_list, '-s', color='green')
    plt.title("SSIM Trend")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")

    plt.subplot(1, 4, 4)
    plt.plot(psnr_list, '-^', color='darkorange')
    plt.title("PSNR Trend")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_and_metrics.png"))
    plt.close()

def train_vae(args, dataloader):
    model = VAE(latent_dim=args.latent_dim).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs(f"{args.output_dir}/reconstructions", exist_ok=True)
    os.makedirs(f"{args.output_dir}/generations", exist_ok=True)

    for folder in ["reconstructions", "generations"]:
        files = glob.glob(os.path.join(args.output_dir, folder, "*.png"))
        for f in files:
            os.remove(f)

    fixed_batch, _ = next(iter(dataloader))
    fixed_batch = fixed_batch.to(args.device)
    utils.save_image(fixed_batch, os.path.join(args.output_dir, 'original_images.png'))

    loss_history = []
    mae_list, ssim_list, psnr_list = [], [], []

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
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        model.eval()
        with torch.no_grad():
            recon, _, _ = model(fixed_batch)
            utils.save_image(recon, f"{args.output_dir}/reconstructions/recon_epoch{epoch+1}.png")

            z = torch.randn(16, args.latent_dim).to(args.device)
            generated = model.decode(z)
            utils.save_image(generated, f"{args.output_dir}/generations/sample_epoch{epoch+1}.png")

            mae, ssim, psnr = compute_metrics(recon, fixed_batch)
            mae_list.append(mae)
            ssim_list.append(ssim)
            psnr_list.append(psnr)

    torch.save(loss_history, os.path.join(args.output_dir, 'loss_history.pt'))
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'vae_model.pt'))
    plot_loss_and_metrics(loss_history, mae_list, ssim_list, psnr_list, args.output_dir)

def main():
    args = get_args()
    dataloader = load_data(args.data_dir, args.batch_size)
    train_vae(args, dataloader)

if __name__ == "__main__":
    main()
