import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image

image_size = 128
channels_img = 3
z_dim = 100
features_g = 64
features_d = 64
batch_size = 128
lr = 2e-4
num_epochs = 50
device = torch.device("mps")

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)])
])

dataset = datasets.ImageFolder(root="/Users/felipepesantez/Documents/development/datasets/landscapes", transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


#DCGAN
#Generator
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super().__init__()
        self.net = nn.Sequential(
          self._block(z_dim, features_g * 16, 4, 1, 0),
          self._block(features_g * 16, features_g * 8, 4, 2, 1),  
          self._block(features_g * 8, features_g * 4, 4, 2, 1),  
          self._block(features_g * 4, features_g * 2, 4, 2, 1),  
          self._block(features_g * 2, features_g, 4, 2, 1),  
          nn.ConvTranspose2d(features_g, channels_img, 2, 2, 1),
          nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)

#Discriminator

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            self._block(features_d * 8, features_d * 16, 4, 2, 1),
            nn.Conv2d(features_d * 16, 1, 2, 1, 0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)
    
#models

gen = Generator(z_dim, channels_img, features_g).to(device)
disc = Discriminator(channels_img, features_d).to(device)

#loss and optim
criterion = nn.BCELoss()
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, z_dim, 1, 1).to(device)


#training loop

print("Training starting.....")
# os.makedirs("generated", exist_ok=True)
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(real.size(0), z_dim, 1,1).to(device)
        fake = gen(noise)

        #Disc
        disc_real = disc(real).view(-1)
        disc_fake = disc(fake.detach()).view(-1)
        loss_disc = criterion(disc_real, torch.ones_like(disc_real)) + criterion(disc_fake, torch.zeros_like(disc_fake))

        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        #Gen
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    print(f"[{epoch+1}/{num_epochs}] Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")

    with torch.no_grad():
        fake_samples = gen(fixed_noise)
        fake_samples = (fake_samples + 1) / 2
        save_image(fake_samples, f"results/epoch_{epoch+1}.png", nrow=4)

print("Training Done!!!")