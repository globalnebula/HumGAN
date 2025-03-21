import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

#Load the data
samples_path = "Humans"
output_images_path = "Generated_Images"
os.makedirs(output_images_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class HumanDataset(Dataset):
    def __init__(self, images_path, transform=None):
        self.images = []
        self.transform = transform

        corrupted_files = []
        for root, _, files in os.walk(images_path):
            for file in files:
                if file.endswith((".jpg", ".png", ".jpeg")):  # Fixed .endsWith to .endswith
                    img_path = os.path.join(root, file)

                    try:
                        img = cv2.imread(img_path)
                        if img is None or img.size == 0:
                            corrupted_files.append(img_path)
                            continue
                        self.images.append(img_path)
                    except Exception as e:
                        print(f"Error Loading {img_path} : {str(e)}")
                        corrupted_files.append(img_path)
        
        if corrupted_files:
            print(f"Warning: Found {len(corrupted_files)} corrupted/unreadable files")
        
        print(f"Successfully Loaded {len(self.images)} valid images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image {img_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform:
                image = self.transform(image)
            
            return image
        
        except Exception as e:
            print(f"Error processing image {img_path} : {str(e)}")
            return torch.zeros(3, 64, 64)
        
dataset = HumanDataset(samples_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False), 
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
        
class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, ndf, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if len(dataset) == 0:
    print("Error : No images found in your dataset, Please check your data folder.")
    import sys
    sys.exit(1)

latent_dim = 100
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

generator.apply(weights_init)
discriminator.apply(weights_init)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

num_epochs = 50
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

print("Starting Training")
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        if batch.size(0) == 0:
            print(f"Skipping empty batch at epoch {epoch+1}, batch {i}")
            continue
        
        batch = batch.to(device)
        batch_size = batch.size(0)

        optimizer_D.zero_grad()
        real_labels = torch.full((batch_size, 1), 1.0, device=device)
        output = discriminator(batch)
        loss_D_real = criterion(output, real_labels)
        loss_D_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        fake_labels = torch.full((batch_size, 1), 0.0, device=device)
        output = discriminator(fake_images.detach())
        loss_D_fake = criterion(output, fake_labels)
        loss_D_fake.backward()
        D_G_z1 = output.mean().item()

        loss_D = loss_D_real + loss_D_fake
        optimizer_D.step()

        optimizer_G.zero_grad()
        output = discriminator(fake_images)
        loss_G = criterion(output, real_labels)
        loss_G.backward()
        D_G_z2 = output.mean().item()
        optimizer_G.step()

        if i % 50 == 0:
            print(f"[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}] "
                  f"Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}, "
                  f"D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")

    with torch.no_grad():
        fake_images = generator(fixed_noise)
        img_filename = os.path.join(output_images_path, f"generated_{epoch+1}.png")
        save_image(fake_images[:25], img_filename, nrow=5, normalize=True)
        print(f"Saved image: {img_filename}")

torch.save(generator.state_dict(), "humgan_generator.pth")
torch.save(discriminator.state_dict(), "humgan_discriminator.pth")
print("Training Complete!")
