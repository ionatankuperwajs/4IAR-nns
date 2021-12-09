"""
File that defines different network architectures. Currently includes shallow/unmasked and deep/masked
convolutional and fully connected networks
"""

import torch.nn as nn
import torch.nn.functional as F

#%%
# Fully connected network with default architecture: no hidden layers, 200 units each
class Linear(nn.Module):
    def __init__(self, num_layers=1, num_units=200):
        super(Linear, self).__init__()
        # Input linear layer
        self.in_layer = nn.Linear(2*4*9, num_units)

        # Many hidden layers
        self.hidden = nn.ModuleList()
        for h_layer in range(num_layers-1):
            self.hidden.append(nn.Linear(num_units, num_units))

        # Output linear layer
        self.out_layer = nn.Linear(num_units, 36)

    def forward(self, x):
        input = x
        # Squeeze to 1-dimension, pass to fully connected input layer
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.in_layer(x))

        # Forward passes through hidden layers if they exist
        if not self.hidden:
            pass
        else:
            for h_layer in self.hidden:
                x = F.relu(h_layer(x))

        # Pass to fully-connected output layer
        x = self.out_layer(x)
        # Enforce move legality by setting the output logits to be very negative where there are pieces
        x += input.sum(dim=1).view(-1, 36)*-1000
        return x

#%%
# Fully connected network with skip connections and default architecture: no hidden layers, 200 units each
class LinearSkip(nn.Module):
    def __init__(self, num_layers=1, num_units=200, bottleneck=50):
        super(LinearSkip, self).__init__()
        self.num_layers = num_layers

        # Input linear layer
        self.in_layer = nn.Linear(2*4*9, num_units)

        # Many hidden layers
        self.layerin = nn.ModuleList()
        self.layerout = nn.ModuleList()
        for h_layer in range(num_layers-1):
            self.layerin.append(nn.Linear(num_units, bottleneck))
            self.layerout.append(nn.Linear(bottleneck, num_units))

        # Output linear layer
        self.out_layer = nn.Linear(num_units, 36)

    def forward(self, x):
        input = x
        # Squeeze to 1-dimension, pass to fully connected input layer
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.in_layer(x))

        # Forward passes through hidden layers if they exist with skip connections
        if not self.layerin:
            pass
        else:
            for i_layer in range(self.num_layers-1):
                identity = x
                x = F.relu(self.layerin[i_layer](x))
                x = self.layerout[i_layer](x)
                x += identity

        # Pass to fully-connected output layer
        x = self.out_layer(x)
        # Enforce move legality by setting the output logits to be very negative where there are pieces
        x += input.sum(dim=1).view(-1, 36)*-1000
        return x

#%%
# Conv NN class definition with default architecture: 1 layer, 4 filters, 3x3 kernel, same padding (to preserve spatial dimension)
class CNN(nn.Module):
    def __init__(self, num_layers=1, num_filters=4, filter_size=3,  stride=1, pad=1):
        super(CNN, self).__init__()
        # First convolutional layer:
        #      2x9x4 board tensor as  input
        #      Pass to num_filters channels by filter_size square convolution
        self.conv = nn.Conv2d(in_channels=2,out_channels=num_filters,kernel_size=filter_size,padding=pad)

        # Hidden convolutional layers stored in a list
        self.hidden = nn.ModuleList()
        for h_layer in range(num_layers-1):
            self.hidden.append(nn.Conv2d(in_channels=num_filters,out_channels=num_filters,kernel_size=filter_size,padding=pad))

        # Recompute dimensions post convolution and output to fully connected layer
        width = (9-filter_size+2*pad+1)
        height = (4-filter_size+2*pad+1)
        self.fc = nn.Linear(num_filters*width*height, 36)

    def forward(self, x):
        # Relu applied to convolution layer
        x = F.relu(self.conv(x))

        # Apply Relu to hidden conv layers if they exist
        if not self.hidden:
            pass
        else:
            for h_layer in self.hidden:
                x = F.relu(h_layer(x))

        # Squeeze to 1-dimension, pass to fully connected output layer
        x = x.view(-1, num_flat_features(x))
        x = self.fc(x)
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