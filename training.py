import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

from emoji_vae import VAE

LATENT_DIM = 32
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 100

DATA_DIR = 'training_data'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VAE(latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def loss_function(recon_x, x, mu, log_var):
    # Reconstruction Loss + KL Divergence
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1, 64, 64), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

print(f"Starting training on {device}...")
fixed_batch, _ = next(iter(dataloader))
fixed_batch = fixed_batch.to(device)
save_image(fixed_batch, os.path.join(RESULTS_DIR, 'original_images.png'))

loss_history = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for i, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    avg_loss = train_loss / len(dataloader.dataset)
    print(f'====> Epoch: {epoch+1} Average loss: {avg_loss:.4f}')
    loss_history.append(avg_loss)

    model.eval()
    with torch.no_grad():
        recon_fixed, _, _ = model(fixed_batch)
        save_image(recon_fixed, os.path.join(RESULTS_DIR, f'reconstructed_epoch_{epoch+1}.png'))
        
        random_noise = torch.randn(BATCH_SIZE, LATENT_DIM).to(device)
        generated_images = model.decode(random_noise)
        save_image(generated_images, os.path.join(RESULTS_DIR, f'generated_epoch_{epoch+1}.png'))

print("Training finished!")
torch.save(loss_history, 'loss_history.pt')
torch.save(model.state_dict(), 'vae_model.pt')
print("Loss history and model weights saved.")