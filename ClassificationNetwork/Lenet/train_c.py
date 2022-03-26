import torch
import torchvision
import torch.nn as nn
import torch.utils.data
from   model import LenNet
import torch.optim as optim 
import torchvision.transforms as transforms

transform = transforms.Compose(
                                [
                                    transforms.ToTensor(),
                                    transforms.Normalize( ( 0.5, 0.5, 0.5 ),
                                                          ( 0.5, 0.5, 0.5 )
                                                        )
                                ]
                              )

traninset  = torchvision.datasets.CIFAR10( root='./data', train=True, download=False, transform=transform )
trainloder = torch.utils.data.DataLoader( traninset, batch_size=36, shuffle=True, num_workers=1 )
testset    = torchvision.datasets.CIFAR10( root='./data', train=True, download=False, transform=transform )
testloder  = torch.utils.data.DataLoader( traninset, batch_size=10000, shuffle=False, num_workers=1 )

test_data_iter = iter(testloder)
test_image, test_label = test_data_iter.next()

classes = ( 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' )

net = LenNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam( net.parameters() , lr=0.001 )

for epoch in range(5):

    running_loss = 0.0 

    for step, data in enumerate( trainloder, start=0 ):

        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function( outputs, labels )
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % 500 == 499:

            with torch.no_grad():

                outputs   = net(test_image)
                predict_y = torch.max( outputs, dim=1 )[1]
                accuracy  = ( predict_y == test_label ).sum().item() / test_label.size(0)
                print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f' %( epoch + 1, step + 1, running_loss / 500, accuracy ) )
                running_loss = 0.0


print( 'Finished Training' )
save_path = './LenNet.pth'
torch.save( net.state_dict(), save_path )
