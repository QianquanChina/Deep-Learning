import torch
import numpy as np
from PIL import Image
from torchvision import transforms

tr = transforms.ToTensor()
#print( tr( './data_set/flower_data/train/daisy/305160642_53cde0f44f.jpg').shape )
img_PIL = Image.open('./data_set/flower_data/train/daisy/305160642_53cde0f44f.jpg')#读取数

print("img_PIL:",img_PIL.size)

#将图片转换成np.ndarray格式
img_PIL = np.array(img_PIL)
print("img_PIL:",img_PIL.shape)

#img = torch.from_numpy(img_PIL.transpose((2, 0, 1))).contiguous()
img = img_PIL.transpose( (2,0,1) )

imge = tr(img_PIL)
print("img_PIL:",imge.shape)
print('img',img.shape)

