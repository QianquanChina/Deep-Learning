import json
import torch 
import pylab
from model import AlexNet
from PIL import Image
from torchvision import transforms

data_transform = transforms.Compose(

                                       [
                                           transforms.Resize( ( 224, 224 ) ),
                                           transforms.ToTensor(),
                                           transforms.Normalize( ( 0.5, 0.5, 0.5 ),
                                                                 ( 0.5, 0.5, 0.5 )
                                                               )
                                       ]

                                    )

image = Image.open('./tulip.jpg')
image = data_transform(image)
image = torch.unsqueeze( image, dim=0 ) #type:ignore

try:

    json_file    = open( './class_indices.json', 'r' )
    class_indict = json.load(json_file)

except Exception as e:

    print(e)
    exit(-1)

model = AlexNet(num_classes=5)
model_weight_path = './AlexNet.pth'
model.load_state_dict( torch.load(model_weight_path) )
model.eval()
with torch.no_grad():

    output      = torch.squeeze( model(image) )
    predict     = torch.softmax( output, dim=0 )
    predict_cal = torch.argmax(predict).numpy()

print( class_indict[ str(predict_cal) ], predict[predict_cal].item() )
pylab.show()

