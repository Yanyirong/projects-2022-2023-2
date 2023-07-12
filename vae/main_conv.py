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
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder (编码器) 部分
        self.encoder_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.encoder_relu1 = nn.ReLU()
        self.encoder_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.encoder_relu2 = nn.ReLU()
        self.fc_mu = nn.Linear(32 * 7 * 7, 2)
        self.fc_log_std = nn.Linear(32 * 7 * 7, 2)

        # Decoder (解码器) 部分
        self.decoder_fc = nn.Linear(2, 32 * 7 * 7)
        self.decoder_relu1 = nn.ReLU()
        self.decoder_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.decoder_relu2 = nn.ReLU()
        self.decoder_conv2 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        self.decoder_sigmoid = nn.Sigmoid()

    def encode(self, x):
        z = self.encoder_conv1(x)
        z = self.encoder_relu1(z)
        z = self.encoder_conv2(z)
        z = self.encoder_relu2(z)
        z = torch.flatten(z, start_dim=1)
        mu = self.fc_mu(z)
        log_std = self.fc_log_std(z)
        return mu, log_std

    def decode(self, z):
        z = self.decoder_fc(z)
        z = self.decoder_relu1(z)
        z = z.view(-1, 32, 7, 7)
        recon = self.decoder_conv1(z)
        recon = self.decoder_relu2(recon)
        recon = self.decoder_conv2(recon)
        recon = self.decoder_sigmoid(recon)
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
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
    epochs = 100
    batch_size = 64
    recon = None
    img = None
    make_dir("./img/vae")
    make_dir("./model_weights/vae")
    train_data = torchvision.datasets.MNIST(root='./data',train=True,transform=torchvision.transforms.ToTensor(),download=True)
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    vae = VAE()
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(epochs):
        train_loss = 0
        i = 0
        for batch_id, data in enumerate(data_loader):
            img, _ = data
            inputs = img.view(-1, 1, 28, 28) 
            recon, mu, log_std = vae(inputs)
            loss = vae.loss_function(recon, inputs, mu, log_std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            i += 1
            if batch_id % 100 == 0:
                print("Epoch[{}/{}], Batch[{}/{}], batch_loss:{:.6f}".format(
                    epoch+1, epochs, batch_id+1, len(data_loader), loss.item()))
        print("epoch:{},\t epoch_average_batch_loss:{:.7f}".format(epoch+1, train_loss/i), "\n")

        # save imgs
        if epoch % 10 == 0:
            imgs = to_img(recon.detach())
            print(recon.shape)
            path = "./img/vae/epoch{}.png".format(epoch+1)
            torchvision.utils.save_image(imgs, path, nrow=10)
            print("save:", path, "\n")

            # 1 dim
            linear_recons = vae.decode(torch.linspace(-5, 5, 800).view(-1, 2))
            
            # 2 dim
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