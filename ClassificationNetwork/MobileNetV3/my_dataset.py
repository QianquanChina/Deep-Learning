import torch
from PIL import Image
from torch.utils.data import Dataset

# 自定义dataset 需要继承Dataset 重写__len__()和__getitem__()方法
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path : list, images_class : list, transform = None):

        self.images_path  = images_path
        self.images_class = images_class
        self.transform    = transform

    def __len__(self):

        return len( self.images_path )


    # 例如 data_set = MyDataSet( params_one, params_two, ... ) data_set[m], 此时就会调用__getitem__()成员函数
    def __getitem__(self, item):

        img = Image.open( self.images_path[item] )

        if img.mode != 'RGB':

            raise ValueError( " image: {} isn't RGB mode ".format( self.images_path[item] ) )

        label = self.images_class[item]

        if self.transform is not None:

            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):

        images, labels = tuple( zip(*batch) )

        images = torch.stack( images, dim = 0 )
        labels = torch.as_tensor(labels)

        return images, labels
