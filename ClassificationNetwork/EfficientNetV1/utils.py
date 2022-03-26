import os
import json
import pickle
import random

import pylab as plt

def read_split_data(root : str, val_rate : float = 0.2):

    random.seed(0)
    assert os.path.exists(root), 'dataset root: {} does no exist. '.format(root)

    # 遍历文件夹，一个文件对应一个类别
    flower_class = [ cla for cla in os.listdir(root) if os.path.isdir( os.path.join( root, cla) ) ]
    flower_class.sort()
    class_indices = dict( ( k, v ) for v, k in enumerate( flower_class ) )
    json_str = json.dumps( dict( ( val, key ) for key, val, in class_indices.items() ), indent = 4 ) 
    with open( 'class_indices.json', 'w' ) as json_file:

        json_file.write(json_str)

    train_images_path  = []
    train_images_label = []
    val_images_path    = []
    val_images_label   = []
    every_class_num    = []
    supported = [ '.jpg', '.JPG', '.png', '.PNG' ]

    for cla in flower_class:

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

        plt.bar( range( len(flower_class) ), every_class_num, align = 'center' )
        plt.xticks( range( len(flower_class) ), flower_class )

        for i, v in enumerate(every_class_num):
            
            plt.text( x = i, y = v + 5, s = str(v), ha = 'center')

        plt.xlabel( 'images class' )
        plt.ylabel( 'number of images' )
        plt.title( 'flower class distribution' )
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


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
