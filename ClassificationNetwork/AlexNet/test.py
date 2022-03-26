from PIL import Image
import numpy as np


img_PIL = Image.open('./data_set/flower_data/train/daisy/305160642_53cde0f44f.jpg')#读取数

print("img_PIL:",img_PIL)

print("img_PIL:",type(img_PIL))

#将图片转换成np.ndarray格式
img_PIL = np.array(img_PIL)
print("img_PIL:",img_PIL.shape)

