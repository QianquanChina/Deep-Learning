import os
import math
import torch
import argparse
import tempfile
import torch.utils.data
from torch import optim
from model import resNet34
from my_dataset import MyDataSet 
from torchvision import transforms 
import torch.optim.lr_scheduler as lr_scheduler
from distributed_utils import init_distributed_mode, dist, cleanup
from train_eval_utils import evaluate, train_one_epoch
from data_utils import plot_class_preds, read_split_data
from torch.utils.tensorboard.writer import SummaryWriter

def main(args):

    if torch.cuda.is_available() is False:

        raise EnvironmentError('Not find GPU device for training')


    # 初始化各进程环境
    init_distributed_mode( args = args )    
    rank         = args.rank
    device       = torch.device( args.device )
    num_classes  = args.num_classes
    weights_path = args.weights
    args.lr     *= args.world_size

    # 在第一进程中打印信息，并实例化Tensorboard。
    tb_writer = SummaryWriter()
    if rank == 0:

        print(args)
        print( ' Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/' )

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

    # 给每个RANK对应的进程分配训练索引
    train_sampler = torch.utils.data.DistributedSampler(train_data_set)
    val_sampler   = torch.utils.data.DistributedSampler(val_data_set  )

    # 将样本索引每batch_size个元素组成一个list
    train_batch_samper = torch.utils.data.BatchSampler( train_sampler, batch_size, drop_last = True )

    nw = min( [os.cpu_count(), batch_size if batch_size > 1 else 0, 8 ] ) #type:ignore

    if rank == 0:

        print( 'Using {} dataloader workers'.format(nw) )

    # DataLoader 返回的是一个迭代器
    train_loader = torch.utils.data.DataLoader(
                                                  train_data_set,
                                                  batch_sampler = train_batch_samper, # type: ignore
                                                  pin_memory    = True,
                                                  num_workers   = nw, #type: ignore
                                                  collate_fn    = train_data_set.collate_fn
                                              )
    val_loader   = torch.utils.data.DataLoader(
                                                  val_data_set,
                                                  batch_size  = batch_size,
                                                  sampler     = val_sampler,
                                                  pin_memory  = True,
                                                  num_workers = nw, #type: ignore
                                                  collate_fn  = train_data_set.collate_fn
                                              )# }}}

    # 检测是否存在预训练权重，存在则载入
    model = resNet34( num_classes = args.num_classes ).to(device)# {{{

    # init_img = torch.zeros( ( 1, 3, 224, 224 ), device = device )
    # tb_writer.add_graph( model, init_img )

    if os.path.exists(weights_path):
    
        weights_dict = torch.load( args.weights, map_location = device )
        load_weights_dict = {
                                k : v for k, v in weights_dict.items()

                                        if model.state_dict()[k].numel() == v.numel()
                            }

        model.load_state_dict( load_weights_dict, strict = False ) #type: ignore

    else:
        
        checkpoint_path = os.path.join( tempfile.gettempdir(), 'initial_weights.pth' )

        if rank == 0:

            torch.save( model.state_dict(), checkpoint_path )

        dist.barrier()
        model.load_state_dict( torch.load( checkpoint_path, map_location = device ) )# }}}

    # 是否冻结权重
    if args.freeze_layers:# {{{

        for name, para in model.named_parameters():

            # 除最后的全连接层外，其他权重全部冻结
            if 'fc' not in name:

                para.requires_grad_(False)
    else:

        # 只有训练带有BN结构的网络时使用SyncBatchNorm才有意义
        if args.syncBN:

            # 使用SyncBatchNorm后训练更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    model = torch.nn.parallel.DistributedDataParallel( model, device_ids = [args.gpu] )

    pg = [ 
            p for p in model.parameters()
                
                if p.requires_grad 
         ]# }}}

    # 准备开始训练
    optimizer = optim.SGD( pg, lr = args.lr, momentum = 0.9, weight_decay = 0.005 )# {{{
    lf        = lambda x : ( ( 1 + math.cos( x * math.pi / args.epochs ) ) / 2 ) * ( 1 - args.lrf ) + args.lrf
    scheduler = lr_scheduler.LambdaLR( optimizer, lr_lambda = lf)

    for epoch in range(args.epochs):
        
        train_sampler.set_epoch(epoch)
        mean_loss = train_one_epoch(
                                        model       = model,
                                        optimizer   = optimizer,
                                        data_loader = train_loader,
                                        device      = device,
                                        epoch       = epoch
                                   )

        scheduler.step()

        sum_num = evaluate(
                            
                             model        = model,
                             data_loader  = val_loader,
                             device       = device,
                          )

        acc = sum_num / val_sampler.total_size

        if rank == 0:

            tags = [ 'loss', 'accuracy', 'learning_rate' ]
            tb_writer.add_scalar( tags[0], mean_loss, epoch )
            tb_writer.add_scalar( tags[1], acc, epoch )
            tb_writer.add_scalar( tags[2], optimizer.param_groups[0]['lr'], epoch )

        # 如果在一句在前面会丢失最后一次的上传数据
            torch.save( model.state_dict(), './weights/model-{}.pth'.format(epoch) )# }}}

    if rank == 0:

        if os.path.exists(checkpoint_path) is True: 

            os.remove(checkpoint_path)

    cleanup()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( '--num_classes'  , type = int  , default = 5     )
    parser.add_argument( '--epochs'       , type = int  , default = 1     )
    parser.add_argument( '--batch-size'   , type = int  , default = 16    )
    parser.add_argument( '--lr'           , type = float, default = 0.001 )
    parser.add_argument( '--lrf'          , type = float, default = 0.1   )
    parser.add_argument( '--freeze-layers', type = bool , default = False )
    parser.add_argument( '--syncBN'       , type = bool , default = False )
    parser.add_argument( '--weights'      , type = str  , default = '', help = 'initial weights path' )
    parser.add_argument( '--data-path'    , type = str  , default = './data_set/flower_data/flower_photos' )

    parser.add_argument( '--device'       , type = str  , default = 'cuda', help = 'device id ( i.e. 0 or 0, 1 or cpu )' )
    parser.add_argument( '--world-size'   , type = int  , default = 4     , help = 'number of distributed processes' )
    parser.add_argument( '--dist-url'     , type = str  , default = 'env://', help = 'url used to set up distributed training' )
    opt = parser.parse_args()
    main(opt)

