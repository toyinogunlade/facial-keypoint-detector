## TODO: define the convolutional neural network architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        ## To calculate the size of the output activation map we use this formula ((W-F + 2P)/S)+1, where W -> width of input image,
        ## F is the size of the receptive field of the conv filter, P is the size of the kernel padding = 0 and the stride S is the
        ##  step size of convolution  operation.
        ## the output Tensor for one grayscale image will have the dimensions: (32, 220, 220) - ((W-F + 2P)/S)+1 = (224-5 + 0)/1 +1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5)
        ## after the pool layer, this becomes (68, 110, 110)
        self.pool = nn.MaxPool2d(2, 2)   
        self.dropout1 = nn.Dropout(p = 0.05)
        ## dimensions -> (64, 108, 108) -> (110 - 3 + 0)/1 +1 = 108
        self.conv2 = nn.Conv2d(32, 64, 3) 
        ## after the pool layer, this becomes -> (64, 54, 54)
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.conv2_bn = nn.BatchNorm2d(64)
        ## dimensions -> (128, 52, 52) -> (54 - 3 + 0)/1 +1 = 52
        self.dropout2 = nn.Dropout(p = 0.10)
        self.conv3 = nn.Conv2d(64, 128, 3)
        ## after the pool layer, this becomes -> (128, 26, 26)
        self.pool3 = nn.MaxPool2d(2, 2) 
        self.dropout3 = nn.Dropout(p = 0.15)
        ## dimensions -> (256, 24, 24) -> (26 - 3 + 0)/1 +1 = 24
        self.conv4 = nn.Conv2d(128, 256, 3)
        ## after the pool layer, this becomes -> (256, 12, 12)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout(p = 0.20)
        ## dimensions -> (512, 12, 12) -> (12 - 1 + 0)/1 +1 = 12
        self.conv5 = nn.Conv2d(256, 512, 1)
        ## after the pool layer, this becomes -> (512, 6, 6) - which is flattened 
        ## and passed as input to the fc layers. 
        self.pool5 = nn.MaxPool2d(2, 2)
        
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.drop1 = nn.Dropout(p = 0.45)
        self.fc2 = nn.Linear(1024, 1024) 
        self.drop2 = nn.Dropout(p = 0.60)
        self.fc3 = nn.Linear(1024, 136) 
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2_bn(self.conv2(x))))
        ##x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = self.pool4(F.relu(self.conv4_bn(self.conv4(x))))
        ##x = self.pool4(F.relu(self.conv4(x)))
        x = self.dropout4(x)
        x = self.pool5(F.relu(self.conv5(x)))
        # Flattening the layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x