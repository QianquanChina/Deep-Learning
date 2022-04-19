import os
import torch
import datetime
import argparse
import torch.utils.data
from utils import get_transform
from my_dataset import VOCSegmentation
from model import swin_base_patch4_window7_224
from train_utils import train_one_epoch, evaluate, create_lr_scheduler



def main(args):

    device = torch.device( args.device if torch.cuda.is_available() else 'cpu' )

    if os.path.exists('./weights') is False:

        os.makedirs('./weights')


    batch_size  = args.batch_size
    num_classes = args.num_classes + 1

    assert args.num_classes + 1 == num_classes, 'dataset num_classes : {}, input {}'.format( args.num_classes, num_classes )

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format( datetime.datetime.now().strftime( "%Y%m%d-%H%M%S" ) )

    # 数据处理
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt{{{
    train_dataset = VOCSegmentation(
                                       args.data_path,
                                       year = "2012",
                                       transforms = get_transform( train = True ),
                                       txt_name = "train.txt"
                                   )

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset   = VOCSegmentation(
                                       args.data_path,
                                       year = "2012",
                                       transforms = get_transform( train = False ),
                                       txt_name = "val.txt"
                                   )

    nw = min( [os.cpu_count(), batch_size if batch_size > 1 else 0, 8 ] ) #type:ignore
    print( 'Using {} dataloader workers'.format(nw) )

    train_loader = torch.utils.data.DataLoader(
                                                  train_dataset,
                                                  batch_size  = batch_size,
                                                  num_workers = nw, #type: ignore
                                                  shuffle     = True,
                                                  pin_memory  = True,
                                                  collate_fn  = train_dataset.collate_fn
                                              )

    val_loader = torch.utils.data.DataLoader(
                                                val_dataset,
                                                batch_size  = 1,
                                                num_workers = nw, # type: ignore
                                                pin_memory  = True,
                                                collate_fn  = val_dataset.collate_fn
                                            )# }}}

    # 检测是否存在预训练权重，存在则载入
    model = swin_base_patch4_window7_224( num_classes = args.num_classes ).to( device )# {{{

    if args.weights != '':

        assert os.path.exists(args.weights), "weights file: '{}' not exists.".format(args.weights)

        weights_dict = torch.load( args.weights, map_location = device )['model']

        # 删除有关分类的权重
        for k in list( weights_dict.keys() ):

            if 'head' in k:

                del weights_dict[k]

        print( model.load_state_dict( weights_dict, strict = False ) ) # type: ignore}}}

    # 是否冻结权重
    if args.freeze_layers:# {{{

        for name, para in model.named_parameters():

            # 除最后的全连接层外，其他权重全部冻结
            if 'head' not in name:

                para.requires_grad_(False)
            else:

                print( 'training {}'.format(name) )

    pg = [ 
            p for p in model.parameters()
                
                if p.requires_grad 
         ]# }}}

    # 准备开始训练
    optimizer = torch.optim.SGD( pg, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay )# {{{
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    for epoch in range( args.start_epoch, args.epochs ):

        mean_loss, lr = train_one_epoch( 
                                           model, 
                                           optimizer, 
                                           train_loader, 
                                           device, 
                                           epoch,
                                           lr_scheduler = lr_scheduler, 
                                           print_freq = args.print_freq
                                       )

        confmat = evaluate( model, val_loader, device = device, num_classes = num_classes )
        val_info = str(confmat)
        print(val_info)

        # write into txt
        with open(results_file, "a") as f:

            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f'[epoch: {epoch}]\n' \
                         f'train_loss: {mean_loss:.4f}\n' \
                         f'lr: {lr:.6f}\n'

            f.write( train_info + val_info + '\n\n')

        save_file = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args
                    }

    torch.save(save_file, "./weights/model_{}.pth".format(epoch)) # type: ignore # }}}}}}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( '--num_classes'  , type = int  , default = 20     )
    parser.add_argument( '--epochs'       , type = int  , default = 10     )
    parser.add_argument( '--batch-size'   , type = int  , default = 1      )
    parser.add_argument( '--lr'           , type = float, default = 0.0001 )
    parser.add_argument( '--freeze-layers', type = bool , default = False  )
    parser.add_argument('--momentum', default = 0.9, type = float, metavar = 'M',help = 'momentum')
    parser.add_argument('--wd', '--weight-decay', default = 1e-4, type = float,metavar = 'W', help = 'weight decay (default: 1e-4)', dest = 'weight_decay')
    parser.add_argument( '--weights'      , type = str  , default = './swin_base_patch4_window7_224.pth', help = 'initial weights path' )
    parser.add_argument( '--data-path'    , type = str  , default = '../DataSet' )
    parser.add_argument( '--device'       , type = str  , default = 'cuda', help = 'device id ( i.e. 0 or 0, 1 or cpu )' )
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    opt = parser.parse_args()
    main(opt)



