import torch
from neural_genesis import nn
# neural_genesis is a extension of pytorch.
# if anything gose wrong, instead with:
# from torch import nn


# define a what ever model you want
# recommend extend from nn.Module
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


# define a function here to build your model.
# this function should be imported in file "core/models/__init__.py".
def my_model(**kwargs):
    return Model()
