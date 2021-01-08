"""
File that defines different network architectures. Currently includes shallow/unmasked and deep/masked
convolutional and fully connected networks
"""

import torch.nn as nn
import torch.nn.functional as F

#%%
# Conv NN class definition with basic architecture: 4 filters, 3x3 kernel, same padding (to preserve spatial dimension)
class BasicCNN(nn.Module):
    def __init__(self, num_filters=4, filter_size=3,  stride=1, pad=1):
        super(BasicCNN, self).__init__()
        # Convolutional layer
        #      2x9x4 board tensor as  input
        #      Pass to num_filters channels by filter_size square convolution
        self.conv = nn.Conv2d(in_channels=2,out_channels=num_filters,kernel_size=filter_size,padding=pad)
        # Recompute dimensions post convolution and output to fully connected layer
        width = (9-filter_size+2*pad+1)
        height = (4-filter_size+2*pad+1)
        self.fc = nn.Linear(num_filters*width*height, 36)

    def forward(self, x):
        # Relu applied to convolution layer
        x = F.relu(self.conv(x))
        # Squeeze to 1-dimension, pass to fully connected output layer
        x = x.view(-1, num_flat_features(x))
        x = self.fc(x)
        return x

#%%
# Adds additional convolutional layers
class DeepCNN(nn.Module):
    def __init__(self, num_filters=32, filter_size=3,  stride=1, pad=1):
        super(DeepCNN, self).__init__()
        # Many convolutional layers
        self.conv1 = nn.Conv2d(in_channels=2,out_channels=num_filters,kernel_size=filter_size,padding=pad)
        self.conv2 = nn.Conv2d(in_channels=num_filters,out_channels=num_filters,kernel_size=filter_size,padding=pad)
        self.conv3 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=filter_size, padding=pad)
        self.conv4 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=filter_size, padding=pad)
        # Recompute dimensions post convolution and output to fully connected layer
        width = (9-filter_size+2*pad+1)
        height = (4-filter_size+2*pad+1)
        self.fc = nn.Linear(num_filters*width*height, 36)

    def forward(self, x):
        input = x
        # Relu applied to convolution layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # Squeeze to 1-dimension, pass to fully connected output layer
        x = x.view(-1, num_flat_features(x))
        x = self.fc(x)
        # Enforce move legality by setting the output logits to be very negative where there are pieces
        x += input.sum(dim=1).view(-1, 36)*-1000
        return x

#%%
# Basic fully connected network: input, output, 1 hidden layer with 200 units
class BasicLinear(nn.Module):
    def __init__(self):
        super(BasicLinear, self).__init__()
        # Input linear layer
        self.fc1 = nn.Linear(2*4*9, 200)
        # Output linear layer
        self.fc2 = nn.Linear(200, 36)

    def forward(self, x):
        # Squeeze to 1-dimension, pass to fully connected input layer
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        # Pass to fully-connected output layer
        x = self.fc2(x)
        return x

#%%
# Deep fully connected network with 4 hidden layers with 200 units each (6 layers total)
class DeepLinear(nn.Module):
    def __init__(self):
        super(DeepLinear, self).__init__()
        # Input linear layer
        self.fc1 = nn.Linear(2*4*9, 200)
        # Many hidden layers
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        # Output linear layer
        self.fc5 = nn.Linear(200, 36)

    def forward(self, x):
        input = x
        # Squeeze to 1-dimension, pass to fully connected input layer
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        # Forward passes
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # Pass to fully-connected output layer
        x = self.fc5(x)
        # Enforce move legality by setting the output logits to be very negative where there are pieces
        x += input.sum(dim=1).view(-1, 36)*-1000
        return x

#%% HELPER FUNCTIONS

# Custom function to perform max pooling across channels rather than spatially
class ChannelPool(nn.MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n,c,w*h).permute(0,2,1)
        pooled =  F.max_pool1d(input, self.kernel_size, self.stride,
                        self.padding, self.dilation, self.ceil_mode,
                        self.return_indices)
        _, _, c = pooled.size()
        pooled = pooled.permute(0,2,1)
        return pooled.view(n,c,w,h)

# Calculates the number of features when flattening a tensor
def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features