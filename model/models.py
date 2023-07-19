import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.fc1_y = nn.Linear(10, 512)
        self.fc2_y = nn.Linear(512, 1024)

        self.fc1 = nn.Linear(self.z_dim, 1024)
        self.fc2 = nn.Linear(1024 + 1024, 4 * 4 * 4 * 64)

        self.F_Leaky = nn.LeakyReLU(0.2)

        self.deconv1 = nn.ConvTranspose2d(4 * 64, 2 * 64, 3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(2 * 64, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x, labels):
        batch_size = x.size(0)
        y_ = self.fc1_y(labels)
        y_ = self.F_Leaky(y_)

        y_ = self.fc2_y(y_)
        y_ = self.F_Leaky(y_)

        x = self.fc1(x)
        x = self.F_Leaky(x)

        x = torch.cat([x, y_], 1)
        x = self.fc2(x)
        x = self.F_Leaky(x)

        x = x.view(batch_size, 4 * 64, 4, 4)

        x = self.deconv1(x)
        x = self.F_Leaky(x)

        x = self.deconv2(x)
        x = self.F_Leaky(x)

        x = self.deconv3(x)
        x = F.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 2 * 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(2 * 128, 4 * 128, 3, stride=2, padding=1)

        self.fc1 = nn.Linear(4 * 4 * 4 * 128 + 2048, 1024)
        self.fc2 = nn.Linear(1024, 1)

        self.fc1_y = nn.Linear(10, 1024)
        self.fc2_y = nn.Linear(1024, 2048)

        self.F_Leaky = nn.LeakyReLU(0.2)

    def forward(self, x, labels):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.F_Leaky(x)

        x = self.conv2(x)
        x = self.F_Leaky(x)

        x = self.conv3(x)
        x = self.F_Leaky(x)

        x = x.view(batch_size, 4 * 4 * 4 * 128)

        y_ = self.fc1_y(labels)
        y_ = self.F_Leaky(y_)

        y_ = self.fc2_y(y_)
        y_ = self.F_Leaky(y_)

        x = torch.cat([x, y_], 1)

        x = self.fc1(x)
        x = self.F_Leaky(x)

        x = self.fc2(x)
        return x
