import torch.nn as nn
import torch.nn.functional as F

class LenNet(nn.Module):

    def __init__(self):

        super( LenNet, self ).__init__()
        self.conv1 = nn.Conv2d( 3, 16, 5 )
        self.pool1 = nn.MaxPool2d( 2, 2 )
        self.conv2 = nn.Conv2d( 16, 32, 5 )
        self.pool2 = nn.MaxPool2d( 2, 2 )
        self.fc1   = nn.Linear( 32*53*53, 120 )
        self.fc2   = nn.Linear( 120, 84 )
        self.fc3   = nn.Linear( 84, 5 )

    def forward(self, x):

        print('123')
        x = F.relu( self.conv1(x) )
        x = self.pool1(x)
        x = F.relu( self.conv2(x) )
        x = self.pool2(x)
        x = x.view( -1, 32*53*53 )
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        x = self.fc3(x)

        return x

