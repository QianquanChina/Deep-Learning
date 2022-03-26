import os
import json
import torch
import pickle
import random
import pylab as plt
from PIL import Image

def read_split_data(root : str, val_rate : float = 0.2):

    random.seed(0)
    assert os.path.exists(root), 'dataset root: {} does no exist. '.format(root)

    # 遍历文件夹，一个文件对应一个类别
    class_names = [ cla for cla in os.listdir(root) if os.path.isdir( os.path.join( root, cla) ) ]
    class_names.sort()
    class_indices = dict( ( k, v ) for v, k in enumerate( class_names ) )
    json_str = json.dumps( dict( ( val, key ) for key, val, in class_indices.items() ), indent = 4 ) 
    with open( 'class_indices.json', 'w' ) as json_file:

        json_file.write(json_str)

    train_images_path  = []
    train_images_label = []
    val_images_path    = []
    val_images_label   = []
    every_class_num    = []
    supported = [ '.jpg', '.JPG', '.png', '.PNG' ]

    for cla in class_names:

        cla_path = os.path.join( root, cla )
        images   = [ os.path.join( root, cla, i ) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported ]
        images_class = class_indices[cla]
        every_class_num.append( len(images) )
        val_path = random.sample( images, k = int( len(images) * val_rate ) )

        for img_path in images:

            if img_path in val_path:

                val_images_path.append(img_path)
                val_images_label.append(images_class)

            else:

                train_images_path.append(img_path)
                train_images_label.append(images_class)

    print( '{} images were found in the dataset'.format( sum(every_class_num) ) )

    plot_image = False

    if plot_image:

        plt.bar( range( len(class_names) ), every_class_num, align = 'center' )
        plt.xticks( range( len(class_names) ), class_names )

        for i, v in enumerate(every_class_num):
            
            plt.text( x = i, y = v + 5, s = str(v), ha = 'center')

        plt.xlabel( 'images class' )
        plt.ylabel( 'number of images' )
        plt.title( 'flower class distribution' )
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label, len(class_names)


def plot_data_loader_image(data_loader):

    batch_size = data_loader.batch_size
    plot_num   = min( batch_size, 4 )

    json_path  = './class_indices.json'
    assert os.path.exists(json_path), json_path + 'does not exist'
    json_file  = open( json_path, 'r' )
    class_indices = json.load(json_file)

    for data in data_loader:

        images, labels = data

        for i in range(plot_num):

            img = images[i].numpy().transpose( 1, 2, 0 )
            img = ( img * [ 0.229, 0.224, 0.225 ] + [ 0.485, 0.456, 0.406 ] ) * 255
            label = labels[i].item()
            plt.subplot( 1, plot_num, i + 1 )
            plt.xlabel( class_indices[ str(label) ] )
            plt.xticks( [] )
            plt.yticks( [] )
            plt.imshow( img.astype('uint8') )

        plt.show()

def plot_class_preds(net, images_dir : str, transform, num_plot : int = 5, device = 'cpu'):

    if not os.path.exists(images_dir):

        print( 'not found {} path, ignore add figure.'.format(images_dir) )

        return None

    label_path = os.path.join( images_dir, 'label.txt')

    if not os.path.exists(label_path):

        print( 'not found {} path, ignore add figure.'.format( label_path ) )

        return None

    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), 'not found {}'.format(json_label_path)

    json_file      = open( json_label_path, 'r' )
    flower_class   = json.load(json_file)
    class_indices  = dict( ( v, k ) for k, v in flower_class.items() )

    label_info = []

    with open( label_path, 'r' ) as rd:

        for line in rd.readlines():

            line = line.strip()

            if len(line) > 0:

                split_info = [ i for i in line.split(' ') if len(i) > 0 ]
                assert len(split_info) == 2, 'label format error, expect file_name and class_name'
                image_name, class_name = split_info
                image_path = os.path.join( images_dir, image_name ) 

                if not os.path.exists(image_path):

                    print('not found {}, skip'.format(image_path) )
                    continue

                if class_name not in class_indices.keys():

                    print( 'unrecognize category {}, skip'.format(class_name) )
                    continue

                label_info.append( [ image_path, class_name ] )

    if len(label_info) == 0:

        return None

    if len(label_info) > num_plot:

        label_info = label_info[ :num_plot ]

    num_imgs = len(label_info)

    images   = []
    labels   = []

    for img_path, class_name in label_info:

        img = Image.open(img_path).convert('RGB')
        label_index = int( class_indices[class_name] )

        img = transform(img)
        images.append(img)
        labels.append(label_index)

    images = torch.stack( images, dim = 0 ).to(device)

    with torch.no_grad():

        output = net(images)
        probs, preds = torch.max( torch.softmax( output, dim = 1 ), dim = 1 )
        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()

    fig = plt.figure( figsize = ( num_imgs * 2.5, 3 ), dpi = 100 )
    for i in range(num_imgs):

        ax    = fig.add_subplot( 1, num_imgs, i + 1, xticks = [], yticks = [] )
        npimg = images[i].cpu().numpy().transpose( 1, 2, 0 )
        npimg = (npimg * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        plt.imshow( npimg.astype('uint8') )

        title = '{}, {:.2f} %\n( label: {} )'.format( 
                                                            flower_class[ str( preds[i] ) ],
                                                            probs[i] * 100,
                                                            flower_class[ str( labels[i] ) ]
                                                        )

        ax.set_title( title, color = ( 'green' if preds[i] == labels[i] else 'red' ) )
        
    return fig



