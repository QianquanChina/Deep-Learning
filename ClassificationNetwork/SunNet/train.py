import os
import math
import torch
import argparse
import torch.utils.data
from torch import optim
from model import resNet34
from my_dataset import MyDataSet 
from torchvision import transforms 
import torch.optim.lr_scheduler as lr_scheduler
from train_eval_utils import evaluate, train_one_epoch
from data_utils import plot_class_preds, read_split_data
from torch.utils.tensorboard.writer import SummaryWriter

def main(args):

    device = torch.device( args.device if torch.cuda.is_available() else 'cpu' )
    print(args)
    print( ' Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/' )
    tb_writer = SummaryWriter()

    if os.path.exists('./weights') is False:

        os.makedirs('./weights')

    # 数据处理
    train_images_path, train_images_label, val_images_path, val_images_label, num_classes = read_split_data(args.data_path)# {{{

    assert args.num_classes == num_classes, 'dataset num_classes : {}, input {}'.format( args.num_classes, num_classes )

    data_transform = {

            'train':transforms.Compose(

                                          [
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(
                                                                    [ 0.485, 0.456, 0.406 ] ,
                                                                    [ 0.229, 0.224, 0.225 ]
                                                                  )
                                          ]
    
                                      ),
    
            'val':  transforms.Compose(
    
                                          [
                                              transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(
                                                                    [ 0.485, 0.456, 0.406 ],
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

    val_data_set   = MyDataSet(
                                  images_path  = val_images_path,
                                  images_class = val_images_label,
                                  transform    = data_transform['val']
                              )

    batch_size = args.batch_size

    nw = min( [os.cpu_count(), batch_size if batch_size > 1 else 0, 8 ] ) #type:ignore

    print( 'Using {} dataloader workers'.format(nw) )

    # DataLoader 返回的是一个迭代器
    train_loader = torch.utils.data.DataLoader(
                                                  train_data_set,
                                                  batch_size  = batch_size,
                                                  shuffle     = True,
                                                  num_workers = 0, #type: ignore
                                                  collate_fn  = train_data_set.collate_fn
                                              )
    val_loader   = torch.utils.data.DataLoader(
                                                  val_data_set,
                                                  batch_size  = batch_size,
                                                  shuffle     = True,
                                                  num_workers = 0, #type: ignore
                                                  collate_fn  = train_data_set.collate_fn
                                              )# }}}

    # 检测是否存在预训练权重，存在则载入
    model = resNet34( num_classes = args.num_classes ).to( device )
    init_img = torch.zeros( ( 1, 3, 224, 224 ), device = device )# {{{
    tb_writer.add_graph( model, init_img )

    if args.weights != '':

        if os.path.exists(args.weights):
    
            weights_dict = torch.load( args.weights, map_location = device )
            load_weights_dict = {
                                    k : v for k, v in weights_dict.items()

                                            if model.state_dict()[k].numel() == v.numel()
                                }

            model.load_state_dict( load_weights_dict, strict = False ) #type: ignore

        else:

            raise FileExistsError( 'not found weights file : {}'.format( args.weights ) )# }}}

    # 是否冻结权重
    if args.freeze_layers:# {{{

        for name, para in model.named_parameters():

            # 除最后的全连接层外，其他权重全部冻结
            if 'fc' not in name:

                para.requires_grad_(False)

    pg = [ 
            p for p in model.parameters()
                
                if p.requires_grad 
         ]# }}}

    # 准备开始训练
    optimizer = optim.SGD( pg, lr = args.lr, momentum = 0.9, weight_decay = 0.005 )# {{{
    lf        = lambda x : ( ( 1 + math.cos( x * math.pi / args.epochs ) ) / 2 ) * ( 1 - args.lrf ) + args.lrf
    scheduler = lr_scheduler.LambdaLR( optimizer, lr_lambda = lf)

    for epoch in range(args.epochs):

        mean_loss = train_one_epoch(
                                        model       = model,
                                        optimizer   = optimizer,
                                        data_loader = train_loader,
                                        device      = device,
                                        epoch       = epoch
                                   )

        scheduler.step()

        acc = evaluate(
                          model        = model,
                          data_loader  = val_loader,
                          device       = device,
                          epochs       = epoch,
                          num_val_data = len(val_data_set),
                          print_epochs = args.epochs
                      )


        fig = plot_class_preds( 
                                  net        = model,
                                  images_dir = './plot_img',
                                  transform  = data_transform['val'],
                                  num_plot   = 5,
                                  device     = device # type:ignore
                              )

        if fig is not None:

            tb_writer.add_figure(

                                    'prediction vs actuals',
                                    figure      = fig,
                                    global_step = epoch
                                )

        tags = [ 'loss', 'accuracy', 'learning_rate' ]
        tb_writer.add_scalar( tags[0], mean_loss, epoch )
        tb_writer.add_scalar( tags[1], acc, epoch )
        tb_writer.add_scalar( tags[2], optimizer.param_groups[0]['lr'], epoch )

        # 如果在一句在前面会丢失最后一次的上传数据
        torch.save( model.state_dict(), './weights/model-{}.pth'.format(epoch) )# }}}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( '--num_classes'  , type = int  , default = 5     )
    parser.add_argument( '--epochs'       , type = int  , default = 30    )
    parser.add_argument( '--batch-size'   , type = int  , default = 16    )
    parser.add_argument( '--lr'           , type = float, default = 0.001 )
    parser.add_argument( '--lrf'          , type = float, default = 0.1   )
    parser.add_argument( '--freeze-layers', type = bool , default = False )
    parser.add_argument( '--weights'      , type = str  , default = '', help = 'initial weights path' )
    parser.add_argument( '--data-path'    , type = str  , default = './data_set/flower_data/flower_photos' )
    parser.add_argument( '--device'       , type = str  , default = 'cuda', help = 'device id ( i.e. 0 or 0, 1 or cpu )' )
    opt = parser.parse_args()
    main(opt)

