import torch
from neural_genesis import nn


class Model(nn.Module):
    def __init__(self,  ):
        super(Model, self).__init__()
        self.ResNetBackbone_g2 = nn.ResNetBackbone(in_channel=1)
        self.Flatten_g4 = nn.Flatten(start_dim=1)
        self.AdaptiveAvgPool2d_g6 = nn.AdaptiveAvgPool2d(output_size=1)
        self.Linear_g10 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        g2 = self.ResNetBackbone_g2(x)
        g6 = self.AdaptiveAvgPool2d_g6(g2)
        g4 = self.Flatten_g4(g6)
        g10 = self.Linear_g10(g4)
        return g10


def my_model(**kwargs):
    return Model()
