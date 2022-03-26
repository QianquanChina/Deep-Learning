import torch
from PIL import Image
from model import LenNet
import torchvision.transforms as transforms

transform = transforms.Compose(
                                [
                                    transforms.Resize( ( 32, 32 ) ),
                                    transforms.ToTensor(),
                                    transforms.Normalize( ( 0.5, 0.5, 0.5 ),
                                                          ( 0.5, 0.5, 0.5 )
                                                        )
                                ]
                              )

classes = ( 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' )

net = LenNet()
net.load_state_dict( torch.load('./LenNet.pth') )

image = Image.open('./1.jpg')
image = transform(image)
image = torch.unsqueeze( image, dim=0 ) # type: ignore 

with torch.no_grad():

    outputs = net(image)
    predict = torch.max( outputs, dim=1 )[1].data.numpy()

#print(predict)
print( classes[ int(predict) ] )
