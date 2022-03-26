import json
import torch 
import pylab
from model import resNet34
from PIL import Image
from torchvision import transforms

data_transform = transforms.Compose(

                                       [
                                           transforms.Resize(256),
                                           transforms.ToTensor(),
                                           transforms.CenterCrop(224),
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

model = resNet34(num_classes=5)
model_weight_path = './resNet34.pth'
missing_key, unexpected_key = model.load_state_dict( torch.load(model_weight_path), strict=False )
model.eval()
with torch.no_grad():

    output      = torch.squeeze( model(image) )
    predict     = torch.softmax( output, dim=0 )
    predict_cal = torch.argmax(predict).numpy()

print( class_indict[ str(predict_cal) ], predict[predict_cal].item() )

