import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATCH_SIZE = 128
LATENT_DIM = 100
LEARNING_RATE = 2e-4
NUM_EPOCHS = 20
OUTPUT_DIR = "gan_samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ---------------------------------------------------------
# Generator
# ---------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


# ---------------------------------------------------------
# Discriminator
# ---------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------
# Initialize models
# ---------------------------------------------------------

generator = Generator(LATENT_DIM).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

g_optimizer = torch.optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)

d_optimizer = torch.optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)


# ---------------------------------------------------------
# Fixed noise for monitoring progress
# ---------------------------------------------------------

fixed_noise = torch.randn(
    64,
    LATENT_DIM,
    device=device
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

g_losses = []
d_losses = []

for epoch in range(NUM_EPOCHS):

    epoch_g_loss = 0.0
    epoch_d_loss = 0.0

    for real_images, _ in train_loader:

        real_images = real_images.view(
            real_images.size(0),
            -1
        ).to(device)

        batch_size = real_images.size(0)

        real_labels = torch.ones(
            batch_size,
            1,
            device=device
        )

        fake_labels = torch.zeros(
            batch_size,
            1,
            device=device
        )


        # -------------------------------------------------
        # 1. Train discriminator
        # -------------------------------------------------

        z = torch.randn(
            batch_size,
            LATENT_DIM,
            device=device
        )

        fake_images = generator(z)

        real_predictions = discriminator(real_images)

        real_loss = criterion(
            real_predictions,
            real_labels
        )

        fake_predictions = discriminator(
            fake_images.detach()
        )

        fake_loss = criterion(
            fake_predictions,
            fake_labels
        )

        d_loss = real_loss + fake_loss

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()


        # -------------------------------------------------
        # 2. Train generator
        # -------------------------------------------------

        z = torch.randn(
            batch_size,
            LATENT_DIM,
            device=device
        )

        fake_images = generator(z)

        predictions = discriminator(fake_images)

        g_loss = criterion(
            predictions,
            real_labels
        )

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()


        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()


    average_d_loss = (
        epoch_d_loss / len(train_loader)
    )

    average_g_loss = (
        epoch_g_loss / len(train_loader)
    )

    d_losses.append(average_d_loss)
    g_losses.append(average_g_loss)

    print(
        f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
        f"D loss: {average_d_loss:.4f} | "
        f"G loss: {average_g_loss:.4f}"
    )


    # -----------------------------------------------------
    # Save sample images after each epoch
    # -----------------------------------------------------

    generator.eval()

    with torch.no_grad():
        samples = generator(fixed_noise)
        samples = samples.view(
            -1,
            1,
            28,
            28
        )

        samples = (
            samples + 1
        ) / 2

        save_image(
            samples,
            os.path.join(
                OUTPUT_DIR,
                f"epoch_{epoch + 1:02d}.png"
            ),
            nrow=8
        )

    generator.train()


# ---------------------------------------------------------
# Plot training losses
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, NUM_EPOCHS + 1),
    d_losses,
    label="Discriminator loss"
)

plt.plot(
    range(1, NUM_EPOCHS + 1),
    g_losses,
    label="Generator loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN training losses")
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Generate final samples
# ---------------------------------------------------------

generator.eval()

with torch.no_grad():

    z = torch.randn(
        64,
        LATENT_DIM,
        device=device
    )

    generated_images = generator(z)

    generated_images = generated_images.view(
        -1,
        1,
        28,
        28
    )

    generated_images = (
        generated_images + 1
    ) / 2


# ---------------------------------------------------------
# Display generated digits
# ---------------------------------------------------------

grid = make_grid(
    generated_images.cpu(),
    nrow=8
)

plt.figure(figsize=(7, 7))

plt.imshow(
    grid.permute(1, 2, 0)
)

plt.axis("off")
plt.title("Generated MNIST digits")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Save trained models
# ---------------------------------------------------------

torch.save(
    generator.state_dict(),
    "generator.pth"
)

torch.save(
    discriminator.state_dict(),
    "discriminator.pth"
)

print("Training complete.")
print("Models saved as generator.pth and discriminator.pth.")
print(f"Generated samples saved in: {OUTPUT_DIR}")