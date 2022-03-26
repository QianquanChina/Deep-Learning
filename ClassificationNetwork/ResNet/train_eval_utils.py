import sys
import torch
import torch.nn
from tqdm import tqdm
from distributed_utils import is_main_process, reduce_value

def train_one_epoch(model, optimizer, data_loader, device, epoch):

    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练数据
    if is_main_process():

        data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):

        images, labels = data
        pred = model( images.to(device) )
        loss = loss_function( pred, labels.to(device) )
        loss.backward()
        loss = reduce_value( loss, average = True )
        mean_loss = ( mean_loss * step + loss.detach() ) / ( step + 1 ) 

        # 在进程0中打印平均loss
        if is_main_process():

            data_loader.desc = '[ epoch {} ] mean loss {}'.format( epoch, round( mean_loss.item(), 3 ) )

        if not torch.isfinite(loss):

            print( 'WARNING: non-finite loss, ending training', loss )
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算机
    if device != torch.device('cpu'):

        torch.cuda.synchronize(device)

    return mean_loss.item()

    
@torch.no_grad()
def evaluate(model, data_loader, device):

    model.eval()

    # 用于保存预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():

        data_loader = tqdm(data_loader)
    
    for step, data in enumerate(data_loader):

        images, labels = data
        pred = model( images.to(device) )
        pred = torch.max( pred, dim = 1 )[1]
        sum_num += torch.eq( pred, labels.to(device)).sum()

    # 等待所有进程计算机
    if device != torch.device('cpu'):

        torch.cuda.synchronize(device)

    sum_num = reduce_value( sum_num, average = False )

    return sum_num.item()
