import os
import json
import time 
import torch
import torch.nn as nn
import torch.utils.data
from model import resNet34, resNet101
import torch.optim as optim
from torchvision import transforms, datasets

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

data_transform = {

        'train':transforms.Compose(

                                      [
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize( [ 0.485, 0.456, 0.406 ] ,
                                                                [ 0.229, 0.224, 0.225 ]
                                                              )
                                      ]

                                  ),

        'val':  transforms.Compose(

                                      [
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize( [ 0.485, 0.456, 0.406 ],
                                                                [ 0.229, 0.224, 0.225 ]
                                                              )
                                      ]

                                  )

                 }

data_root     = os.path.abspath( os.path.join( os.getcwd() ) )
image_path    = data_root + "/data_set/flower_data" 
train_dataset = datasets.ImageFolder( root=image_path + '/train', transform=data_transform['train'] )
flower_list   = train_dataset.class_to_idx
cla_dict      = dict( ( val, key ) for key, val in flower_list.items() )
json_str      = json.dumps( cla_dict, indent=4 )

with open( 'class_indices.json', 'w' ) as json_file:

    json_file.write(json_str)

batch_size    = 32
train_loader  = torch.utils.data.DataLoader( train_dataset, batch_size=batch_size, shuffle=True, num_workers=0 )
val_dataset   = datasets.ImageFolder( root=image_path + '/val', transform=data_transform['val'] )
val_num       = len(val_dataset)
val_loader    = torch.utils.data.DataLoader( val_dataset, batch_size=batch_size, shuffle=False, num_workers=0 )

net           = resNet34(num_classes=5)
net.to(device)

#model_weight_path = './resnet34-333f7ec4.pth'
#missing_keys, unexpected_keys = net.load_state_dict( torch.load(model_weight_path), strict=False )
#inchannel = net.fc.in_features
#net.fc    = nn.Linear( inchannel, 5 ).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer     = optim.Adam( net.parameters(), lr=0.0001 )

save_path     = './resNet34.pth'
best_acc      = 0.0



for epoch in range(30):

    step = 0
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    
    for step, data in enumerate( train_loader, start=0 ):

        image, labels = data
        optimizer.zero_grad()
        logits = net( image.to(device) )
        loss = loss_function( logits, labels.to(device) )
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        rate = ( step + 1 ) / len(train_loader)
        a    = '#' * int( rate * 50 )
        b    = '-' * int( ( 1 - rate )* 50 )
        print("\rtrain loss: {:^3.0f}%[{}{}]{:.3f}". format( int( rate * 100), a, b, loss ), end='' )
    print()
    print( time.perf_counter() - t1 )

    net.eval()
    acc = 0.0
    
    with torch.no_grad():

        for data_test in val_loader:

            test_image, test_labels = data_test
            outputs   = net( test_image.to(device) )
            predict_y = torch.max( outputs, dim=1 )[1]
            acc      += ( predict_y == test_labels.to(device) ).sum().item()
        accurate_test = acc / val_num

        if accurate_test > best_acc:

            torch.save( net.state_dict(), save_path )

        print( '[epoch %d] train_loss: %.3f test_accuracy: %.3f' %( epoch + 1, running_loss / step, acc / val_num ) )

print(' Finished Train' )




