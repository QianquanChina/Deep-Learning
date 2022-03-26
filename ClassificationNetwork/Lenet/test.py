import numpy as np
import torch.utils.data
import torch
a = torch.tensor([[1], [2], [6]])
print(a.shape)
print( torch.max(a, 1)[1] )


