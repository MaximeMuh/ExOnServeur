import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
from math import log
path = "C:/Users/maxim/Desktop/ExOnServeur"
data_path = "C:/Users/maxim/Desktop/ExOnServeur/CAT3"
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
full_dataset = datasets.ImageFolder(root=data_path, transform=transform)
test_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - test_size
train_set, test_set = random_split(full_dataset, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size = 9, padding = 4), # dimension [ 64 , 8, 128, 128]
                nn.ReLU(inplace=True),
                nn.Conv2d(16,32, kernel_size = 9, padding = 4), # dimension [64,32,128,128]
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=5, stride=2, padding = 2), # dimension [64,128,64,64]
                nn.Conv2d(32, 32,kernel_size = 5, stride = 2, padding = 2), # dimension [64,128,32,32]
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16,kernel_size = 5, padding = 2), # dimension [64,64,32,32]
                nn.ReLU(inplace=True),
                nn.Conv2d(16,3,kernel_size = 5, padding = 2), # dimension [64,3,32,32]
                nn.Sigmoid()
            )
        self.mean = nn.Conv2d(3, 3,kernel_size = 5, padding = 2) # prend en entre un [64,3,32,32] renvoie [64,128,8,8]
        self.logvar = nn.Conv2d(3, 3, kernel_size=5, padding = 2) # prend en entre un [64,3,32,32] renvoie [64,128,8,8]

        #decoder on donne ici un [64,3, 32, 32]
        self.decoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size = 5, padding = 2), # dimension [64,64,32,32]
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size = 5, padding = 2), # dimension [64,64,32,32]
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 32, kernel_size = 5, stride = 2, padding = 2), # dimension [64,128,64,64]
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, kernel_size = 8, stride = 2, padding = 3), # dimension [64,64,128,128]
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 3, kernel_size = 9, padding = 3), # dimension [64,3,128,128]
                nn.Sigmoid()
            )

    def encode(self, x):
        y = self.encoder(x) # x de dimension [64,3,128,128] et y de dim [64,3,32,32]
        return self.mean(y), self.logvar(y) # de dim [64, 3, 32, 32]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # de dim [64, 1, 1, 32*32*3]

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

def test_loss(test,mod):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(test):
        x = x.to(device)
        x_hat = mod(x)
        mean, log_var = mod.encode(x)
        loss = loss_function(x, x_hat, mean, log_var)
        overall_loss += loss.item()
    return overall_loss/(batch_idx * batch_size)

def loss_function(x,recon_x , mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, optimizer, epochs, device):
    model.train()
    loss_train_per_epoch=[]
    loss_test_per_epoch=[]
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            mean, log_var = model.encode(x)
            loss = loss_function(x, x_hat, mean, log_var)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        average_loss=overall_loss / (batch_idx * batch_size)
        loss_train_per_epoch.append(log(average_loss))
        loss_test_per_epoch.append(log(test_loss(test_loader,model)))
        if epoch%200 == 0 and epoch!=0:
            X=[k for k in range(epoch+1)]
            plt.plot(X,loss_train_per_epoch)
            plt.plot(X,loss_test_per_epoch)
            torch.save(model, path + "/model" + f"{epoch}" + ".pt")
            plt.savefig(path + "/courbe" + f"{epoch}" + ".png")
    return overall_loss

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train(model, optimizer, epochs=10000,device=device)


