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
    accu_loss     = torch.zeros(1).to(device)  # 累计损失
    accu_num      = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num  = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):

        images, labels = data
        sample_num    += images.shape[0]

        pred         = model( images.to(device) )
        pred_classes = torch.max( pred, dim = 1 )[1]
        accu_num    += torch.eq( pred_classes, labels.to(device) ).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
                                                                                  epoch,
                                                                                  accu_loss.item() / (step + 1),
                                                                                  accu_num.item() / sample_num
                                                                              )

        if not torch.isfinite(loss):

            print('WARNING: non-finite loss, ending training ', loss)

            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num #type:ignore # }}} 

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
def evaluate(model, data_loader, device, epoch ):

    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num  = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)   # 累计损失

    sample_num  = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):

        images, labels = data
        sample_num    += images.shape[0]

        pred         = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num    += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
                                                                                  epoch,
                                                                                  accu_loss.item() / (step + 1),
                                                                                  accu_num.item() / sample_num
                                                                              )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num # type:ignore # }}}
 

