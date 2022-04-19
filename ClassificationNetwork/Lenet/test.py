import torch
import collections 
from model import LenNet

weights = torch.load( './LenNet.pth', map_location = 'cpu' )
model = LenNet()
for key, val in model.state_dict().items():

    print(key)



