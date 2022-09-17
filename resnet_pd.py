from torch import nn
from torch.nn import functional as F
import torch
from models import res_fer

class Res_pd_nofreeze(nn.Module):

    def __init__(self, pretrained=None, num_classes=None, drop_rate=0):
        super(Res_pd_nofreeze, self).__init__()
        self.drop_rate = drop_rate
        if pretrained is not None:
            resnet_fer = res_fer(pretrained=pretrained)
        else:
            print("No pretrained weights!")

        self.features = nn.Sequential(*list(resnet_fer.children())[:-1])
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.features(x)
        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        out = self.fc2(x)

        return out

