import sys
import json
import torch
import torch.nn
import numpy as np
import pylab as plt
from tqdm import tqdm
from prettytable import PrettyTable
from distributed_utils import is_main_process, reduce_value

# 训练一个epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch):# {{{

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

    return mean_loss.item()# }}}

# 使用numpy和matlibplot绘制混淆矩阵
class ConfusionMatrix(object):# {{{

    def __init__(self, num_class : int, labels : list):

        self.matrix    = np.zeros( ( num_class, num_class ) )
        self.num_class = num_class
        self.labels    = labels

    def update(self, preds, labels):

        for p, t in zip( preds, labels ):

            self.matrix[ t, p ] += 1

    def summary(self):

        sum_tp = 0

        for i in range( self.num_class ):

            sum_tp += self.matrix[ i, i ]

        _ = sum_tp / np.sum( self.matrix ) # acc

        table = PrettyTable( [ '', 'Precision', 'Recall', 'Specificity' ] )

        for i in range( self.num_class ):

            tp = self.matrix[ i, i ]
            fp = np.sum( self.matrix[ i, : ] ) - tp
            fn = np.sum( self.matrix[ :, i ] ) - tp
            tn = np.sum( self.matrix ) - tp - fp - fn

            precesion   = round( tp / ( tp + fp ), 3 )
            recall      = round( tp / ( tp + fn ), 3 )
            specificity = round( tn / ( tn + fp ), 3 )
            table.add_row( [ self.labels[i], precesion, recall, specificity] )
        
        print(table)

    def plot_confusion_matrix(self):

        matrix = self.matrix

        plt.imshow( matrix, cmap = plt.cm.Blues ) #type: ignore
        plt.xticks( range( self.num_class ), self.labels, rotation = 45 )
        plt.yticks( range( self.num_class ), self.labels )
        plt.colorbar()
        plt.xlabel( 'True Labels')
        plt.ylabel( 'Predicted Labels' )
        plt.title( 'Confusion matrix' )

        thresh = matrix.max() / 2
        
        for x in range( self.num_class ):

            for y in range( self.num_class ):

                info = int( matrix[ y, x ] )
                plt.text( 
                            x,
                            y,
                            info,
                            verticalalignment   = 'center',
                            horizontalalignment = 'center',
                            color = 'white' if info > thresh else 'black'
                        )

        plt.tight_layout()
        plt.show() # }}}

# 预测结果    
@torch.no_grad()# {{{
def evaluate(model, data_loader, device, epochs, num_val_data, print_epochs = 1 ):

    try:

        json_file = open( './class_indices.json' )
        class_indict = json.load( json_file )

    except Exception as e:

        print(e)
        exit(-1)

    labels = [ label for _, label in class_indict.items() ]
    confusion = ConfusionMatrix( num_class = 5, labels = labels )

    model.eval()

    # 用于保存预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():

        data_loader = tqdm(data_loader)

    pred_list   = []
    pred_labels = []
    acc         = 0

    with torch.no_grad():

        for _, data in enumerate(data_loader):

            images, labels = data
            pred = model( images.to(device) )
            pred = torch.max( pred, dim = 1 )[1]
            pred_list.extend( pred.cpu().numpy() )
            pred_labels.extend( labels.cpu().numpy() )
            sum_num += torch.eq( pred, labels.to(device) ).sum()

        acc = sum_num.item() / num_val_data
        print( '[ epoch {} ] accuracy : {}'.format( epochs, round( acc, 3) ) )

        if  print_epochs - 1 == epochs:

            confusion.update( pred_list, pred_labels ) 
            confusion.summary()
            confusion.plot_confusion_matrix()

    # 等待所有进程计算机
    if device != torch.device('cpu'):

        torch.cuda.synchronize(device)

    sum_num = reduce_value( sum_num, average = False )

    # return sum_num.item()
    return acc # }}}

