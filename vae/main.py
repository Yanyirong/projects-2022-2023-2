import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
# import utils

import torch
import torch.nn as nn
import os

def to_img(x):
    x = x.clamp(0, 1)
    imgs = x.reshape(x.shape[0], 1, 28, 28)
    return imgs

def save_model(model: nn.Module, path):
    torch.save(model.state_dict(), path)
    print("model saved")

def load_model(model: nn.Module, path):
    model.load_state_dict(torch.load(path))
    print("model loaded")

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
class VAE(nn.Module):
    """Implementation of VAE(Variational Auto-Encoder)"""
    def __init__(self):
        super(VAE, self).__init__()
        # the main model,with batchnorm
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3_mu = nn.Linear(128, 2)
        self.fc3_log_std = nn.Linear(128, 2)
        self.fc4 = nn.Linear(2, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, 784)

    # encoder and decoder
    def encode(self, x):
        h1 = F.relu(self.bn1(self.fc1(x)))
        h2 = F.relu(self.bn2(self.fc2(h1)))
        mu = self.fc3_mu(h2)
        log_std = self.fc3_log_std(h2)
        return mu, log_std

    def decode(self, z):
        h4 = F.relu(self.bn4(self.fc4(z)))
        h5 = F.relu(self.bn5(self.fc5(h4)))
        recon = torch.sigmoid(self.fc6(h5))
        return recon

    # reparametrize
    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std.view(-1, 2))  
        eps = torch.randn_like(log_std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_std = self.encode(x)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z)
        return recon, mu, log_std

    def loss_function(self, recon, x, mu, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")  
        kl_loss = -0.5 * (1 + 2*log_std - mu.pow(2) - torch.exp(2*log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss

if __name__ == '__main__':
    # some hyperparameters
    epochs = 100
    batch_size = 64
    recon = None
    img = None
    make_dir("./img/vae")
    make_dir("./model_weights/vae")
    train_data = torchvision.datasets.MNIST(root='./data',train=True,transform=torchvision.transforms.ToTensor(),download=True)
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    vae = VAE()
    # some optimizers

    # optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    # optimizer = troch.optim.SGD(vae.parameters(), lr=1e-3, momentum=0.9)
    # optimizer = torch.optim.RMSprop(vae.parameters(), lr=1e-3, alpha=0.9)
    # optimizer = torch.optim.Adagrad(vae.parameters(), lr=1e-3)
    # optimizer = torch.optim.Adadelta(vae.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-3, weight_decay=1e-5)

    # training
    for epoch in range(epochs):
        train_loss = 0
        index = 0
        for batch_id, data in enumerate(data_loader):
            index += 1
            img, _ = data
            inputs = img.reshape(img.shape[0], -1)
            recon, mu, log_std = vae(inputs)
            loss = vae.loss_function(recon, inputs, mu, log_std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_id % 100 == 0:
                print("Epoch[{}/{}], Batch[{}/{}], batch_loss:{:.6f}".format(
                    epoch+1, epochs, batch_id+1, len(data_loader), loss.item()))
        print("epoch:{},\t epoch_average_batch_loss:{:.7f}".format(epoch+1, train_loss/index), "\n")

        # save imgs
        if epoch % 10 == 0:
            imgs = to_img(recon.detach())
            path = "./img/vae/epoch{}.png".format(epoch+1)
            torchvision.utils.save_image(imgs, path, nrow=10)
            print("save:", path, "\n")
            
            # 1 dim
            linear_recons = vae.decode(torch.linspace(-5, 5, 800).view(-1, 2))
            
            # # 2 dim
            # x = torch.linspace(-5, 5, 20)
            # y = torch.linspace(-5, 5, 20)
            # grid_x, grid_y = torch.meshgrid(x, y)
            # # 将网格点转化为一个shape为(400, 2)的向量
            # grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
            # linear_recons = vae.decode(grid)

            linear_imgs = to_img(linear_recons.detach())
            linear_path = "./img/vae/linear_epoch{}.png".format(epoch+1)
            torchvision.utils.save_image(linear_imgs, linear_path, nrow=20)
            print("save:", linear_path, "\n")

    # save val model
    save_model(vae, "./model_weights/vae/vae_weights.pth")


