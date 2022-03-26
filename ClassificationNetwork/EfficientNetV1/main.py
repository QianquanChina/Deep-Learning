import os
import torch
import argparse
import torch.utils.data
from my_dataset import MyDataSet 
from utils import read_split_data 
from torchvision import transforms

root = './data_set/flower_data/flower_photos'

def main(args):

    device = torch.device( args.device if torch.cuda.is_available() else 'cpu' )
    print(args)
    if os.path.exists('./weights') is False:

        os.makedirs('./weights')

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

    data_transform = {

            'train':transforms.Compose(

                                          [
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize( [ 0.485, 0.456, 0.406 ] ,
                                                                    [ 0.229, 0.224, 0.225 ]
                                                                  )
                                          ]
    
                                      ),
    
            'val':  transforms.Compose(
    
                                          [
                                              transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize( [ 0.485, 0.456, 0.406 ],
                                                                    [ 0.229, 0.224, 0.225 ]
                                                                  )
                                          ]
    
                                      )
    
                     }
    
    train_data_set = MyDataSet(
                                  images_path  = train_images_path,
                                  images_class = train_images_label,
                                  transform    = data_transform['train']
                              )

    batch_size = 8
    nw = min( [os.cpu_count(), batch_size if batch_size > 1 else 0, 8 ] ) #type:ignore
    print( 'Using {} dataloader workers'.format(nw) )

    train_loader = torch.utils.data.DataLoader(
                                                  train_data_set,
                                                  batch_size  = batch_size,
                                                  shuffle     = True,
                                                  num_workers = 0,
                                                  collate_fn  = train_data_set.collate_fn
                                              )
    #检测是否存在预训练权重，存在则载入
    model = 
    for step, data in enumerate(train_loader):

        images, labels = data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( '--num_classes'  , type = int  , default = 5     )
    parser.add_argument( '--epochs'       , type = int  , default = 30    )
    parser.add_argument( '--batch-size'   , type = int  , default = 16    )
    parser.add_argument( '--lr'           , type = float, default = 0.001 )
    parser.add_argument( '--lrf'          , type = float, default = 0.1   )
    parser.add_argument( '--data-path'    , type = str  , default = './data_set/flower_data/flower_photos' )
    parser.add_argument( '--weights'      , type = str  , default = '')
    parser.add_argument( '--freeze-layers', type = bool , default = False )
    parser.add_argument( '--device'       , type = str  , default = 'cuda', help = 'device id ( i.e. 0 or 0, 1 or cpu )' )
    opt = parser.parse_args()
    main(opt)

