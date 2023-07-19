import numpy as np
from torch import nn
import torch.nn.functional as F

# Written based on "DEFENSE-GAN: PROTECTING CLASSIFIERS AGAINST ADVERSARIAL ATTACKS USING GENERATIVE MODELS"
class CNNClassifierA(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(32, 64, 3), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x.view(x.size(0), -1))
        return x


class CNNClassifierB(nn.Module):
    def __init__(self):
        super().__init__()
        # 13x13, 6x6, 4x4
        self.cnn = nn.Sequential(nn.Dropout(p=0.2), nn.Conv2d(1, 64, 3, stride=2), nn.ReLU(),
                                 nn.Conv2d(64, 128, 3, stride=2), nn.ReLU(),
                                 nn.Conv2d(128, 128, 3, stride=1), nn.ReLU(), nn.Dropout(p=0.5))
        self.fc = nn.Sequential(nn.Linear(2048, 10))

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x.view(x.size(0), -1))
        return x


class MLPClassifierE(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 28 * 28
        self.fc = nn.Sequential(nn.Linear(input_size, 512), nn.ReLU(), nn.Dropout(p=0.5),
                                nn.Linear(512, 128), nn.ReLU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 10))

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))