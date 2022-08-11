import torch.nn as nn


class CovNet(nn.Module):
    def __init__(
            self,
            out_dims=(64, 64, 64, 64),
            activation='relu',
            pooling_num=4,
    ):
        super(CovNet, self).__init__()
        layers = []
        indim = 3
        self.out_dim = out_dims[-1]
        activation_layer = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)

        class SimpleBlock(nn.Module):
            def __init__(self, inplane, outplane, pool=False):
                super(SimpleBlock, self).__init__()
                layers = [
                    nn.Conv2d(inplane, outplane, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(outplane),
                    activation_layer,
                ]
                if pool:
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                self.block = nn.Sequential(
                    *layers
                )

            def forward(self, x):
                return self.block(x)

        for idx, dim in enumerate(out_dims):
            layers.append(SimpleBlock(indim, dim, idx < pooling_num))
            indim = dim

        self.layers = nn.Sequential(*layers)

    def forward(self, ims):
        x = self.layers(ims)
        return x
