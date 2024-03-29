import os
import time
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import swin_base_patch4_window7_224 as swin_model

def time_synchronized():

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    return time.time()

def main():

    classes      = 20
    weights_path = './weights'
    img_path     = ''
    palette_path = ''

    assert os.path.exists(weights_path), f'weights {weights_path} not found.'
    assert os.path.exists(img_path), f'weights {img_path} not found.'
    assert os.path.exists(palette_path), f'weights {palette_path} not found.'

    with open( palette_path, 'rb' ) as f:

        pallette_dict = json.load(f)
        pallette = []

        for v in pallette_dict.values():

            pallette += v

    device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )
    print( 'using {} device.'.format(device) )

    model = swin_model( num_classes = classes )

    model.load_state_dict(weights_path)
    model.to(device)

    original_img   = Image.open(img_path)
    data_transform = transforms.Compose(
                                           [
                                               transforms.Resize(520),
                                               transforms.ToTensor(),
                                               transforms.Normalize(
                                                                       mean = ( 0.485, 0.456, 0.406 ),
                                                                       std  = ( 0.229, 0.224, 0.225 )
                                                                   )
                                           ]
                                       )

    img = data_transform(original_img)
    img = torch.unsqueeze( img, dim = 0 ) 

    model.eval()

    with torch.no_grad():

        img_height, img_width = img.shape[ -2: ]
        init_img = torch.zeros( ( 1, 3, img_height, img_width ), device = device )
        model(init_img)

        t_start = time_synchronized()
        output  = model( img.to(device) )
        t_end   = time_synchronized()
        print( 'inference + NMS time : {}'.format( t_end - t_start ) )

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to('cpu').numpy().astype( np.uint8 )

        mask = Image.fromarray(prediction)
        mask.putpalette(pallette)
        mask.save('test_result.png')

if __name__ == '__main__':

    main()
        
