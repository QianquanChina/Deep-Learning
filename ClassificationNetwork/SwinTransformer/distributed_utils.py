import os
import torch
import torch.distributed as dist

def init_distributed_mode(args):
# {{{
    """
    对单机多卡的情况，参数含义如下所示：

    参数：

        WORLD_SIZE : 这个机器有几块GPU。  
        RANK : 这是这个机器的第几块GPU。
        LOCAL_RANK : 这是这个机器的第几块GPU。

    python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py
    在训练的时候指定了use_env那么os.environ就存入rank world_size gpu的参数
    """

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        
        # 将字符型转换成整型的数据
        args.rank       = int( os.environ[   'RANK'   ] )
        args.world_size = int( os.environ['WORLD_SIZE'] )
        args.gpu        = int( os.environ['LOCAL_RANK'] )

    elif 'SLURM_PROCID' in os.environ:

        args.rank = int( os.environ['SLURM_PROCID'] )
        args.gpu  = args.rank % torch.cuda.device_count()

    else:

        print( 'Not using distributed mode' )
        args.distributed = False

        return

    args.distributed = True

    # 对当前的进程指指定的GPU，对单机多卡情况，对每一个GPU启用一个进程
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    # 打印一下该机器的的RANK
    print( '| distributed init (rank {}): {}'.format( args.rank, args.dist_url ), flush=True )

    # 创建进程组
    dist.init_process_group(
                                backend     =args.dist_backend, 
                                init_method =args.dist_url    ,
                                world_size  =args.world_size  ,
                                rank        =args.rank
                           )
    # 等待每一块GPU都运行到这个地方，在接着往下进行
    dist.barrier()# }}}

def cleanup():
# {{{
    dist.destroy_process_group()# }}}

def is_dist_avail_and_initialized():
# {{{
    if not dist.is_available():

        return False

    if not dist.is_initialized():

        return False

    return True# }}}

def get_world_size():
# {{{
    if not is_dist_avail_and_initialized():

        return 1

    return dist.get_world_size()# }}}

def get_rank():
    # {{{
    if not is_dist_avail_and_initialized():

        return 0

    return dist.get_rank()# }}}

def is_main_process():
# {{{
    return get_rank() == 0# }}}

def reduce_value(value, average=True):
# {{{
    world_size = get_world_size()

    if world_size < 2:

        return value

    with torch.no_grad():

       dist.all_reduce(value) 

       if average:

           value /= world_size

       return value# }}}


    
