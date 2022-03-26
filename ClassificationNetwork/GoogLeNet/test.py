
import torch.nn as nn
class BasicCov2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):

        super(BasicCov2d, self).__init__()

        print('3')

    def forward(self, x):

        print('123')
        return x

m = BasicCov2d(1,1)
m(1)
