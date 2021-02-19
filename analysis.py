"""
Main file to train and test networks and then analyze them
"""

#%%
# TODO: automate early stopping
# TODO: look at TQDM for progress

#%%
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats
import pandas as pd
import torch
import torch.optim as optim
from custom_dataset import PeakDataset
from training import train
from network import BasicCNN, DeepCNN, BasicLinear, DeepLinear
from testing import test_performance, test_performance_user, test_with_guesses, test_by_move

#%% DATA LOADING
train_set = PeakDataset('../../Data/small_data/train_moves.pt', '../../Data/small_data/train/train_%d.pt')
val_set = PeakDataset('../../Data/small_data/val_moves.pt', '../../Data/small_data/val/val_%d.pt')
test_set = PeakDataset('../../Data/small_data/test_moves.pt', '../../Data/small_data/test/test_%d.pt')

#%% TRAINING A NEW MODEL

# Initialize the network and train it
net = BasicLinear()
train_loss, val_loss = train(net, batch_size=32, n_epochs=15, learning_rate=0.1, train_set=train_set, val_set=val_set,
                             L2 = 0, model_name='BasicLinear_200_data.pt')

#%% LEARNING CURVES

# Initialize this list when learning rate annealing occurs
# lr_updates = [11]

# Plot the learning curves for train and validation
def plot_learning(train_loss, val_loss, lr, lb, ub):
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(np.arange(1,16), train_loss, lw=2, color='darkblue', marker='o')
        ax.plot(np.arange(1,16), val_loss, lw=2, color='cornflowerblue', marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Negative log-likelihood')
        ax.set_ylim(lb, ub)
        # Add vertical lines for when lr was changed
        for epoch in lr:
            plt.axvline(x=epoch, color='firebrick', linestyle='--')
        ax.legend(['train', 'validation'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()
        # plt.savefig('learning.png', format='png', dpi=1000, bbox_inches='tight')

plot_learning(train_loss, val_loss, lr_updates, 1.9, 2.3)

#%% LOADING A TRAINED MODEL

# Just the state dicts for four models used in course paper
net_DL = DeepLinear()
net_DL.load_state_dict(torch.load('saved_checkpoints/DeepLinear5_200_data.pth'))
net_DL.eval()

net_BasicL = BasicLinear()
net_BasicL.load_state_dict(torch.load('saved_checkpoints/BasicLinear_200_data.pth'))
net_BasicL.eval()

net_DCNN = DeepCNN(num_filters=4, filter_size=3)
net_DCNN.load_state_dict(torch.load('saved_checkpoints/DeepCNN5_4_data.pth'))
net_DCNN.eval()

net_BasicCNN = BasicCNN(num_filters=4, filter_size=3)
net_BasicCNN.load_state_dict(torch.load('saved_checkpoints/BasicCNN_4_data.pth'))
net_BasicCNN.eval()

# The entire model
# TODO: decide what to do with optimizer to restart training
net_DL_full = DeepLinear()
optimizer = optim.SGD(net_DL_full.parameters(), lr=0.1, weight_decay=0)
checkpoint = torch.load('fullDeepLinear5_200_data.pth')
net_DL_full.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
running_loss = checkpoint['running_loss']
train_loss = checkpoint['train_loss']
val_loss = checkpoint['val_loss']
net_DL_full.eval()
# - or -
# net.train()

#%% TESTING AND ANALYSIS

# Run the network on the test set, plot the NLL compared to the cognitive models
perc_correct_DCNN, nll_DCNN = test_performance(net_DCNN, test_set)
perc_correct_DL, nll_DL = test_performance(net_DL, test_set)
perc_correct_BasicCNN, nll_BasicCNN = test_performance(net_BasicCNN, test_set)
perc_correct_BasicL, nll_BasicL = test_performance(net_BasicL, test_set)

# Run the network on the test set by individual user, for averaging and comparison with the cognitive model

with open('larger_data/test_paths.txt', 'r') as filehandle:
    test_paths = json.load(filehandle)

# Set a dictionary with each user as a key and each of their paths as the values in a list
user_paths = {}
for path in test_paths:
    curr_user = path[36:46]
    if curr_user not in user_paths.keys():
        user_paths[curr_user] = [path]
    else:
        user_paths[curr_user].append(path)

perc_correct_DCNN_all, nll_DCNN_all = test_performance_user(net_DCNN, user_paths)
perc_correct_DL_all, nll_DL_all = test_performance_user(net_DL, user_paths)
perc_correct_BasicCNN_all, nll_BasicCNN_all = test_performance_user(net_BasicCNN, user_paths)
perc_correct_BasicL_all, nll_BasicL_all = test_performance_user(net_BasicL, user_paths)

# Read in already formatted log-likelihoods and change to numpy array
model_lls = pd.read_csv('cognitive_model/loglik_tree_notree.txt', sep=" ", header=None)
model_lls = model_lls.values
tree_ll = np.average(model_lls[:, 0])
notree_ll = np.average(model_lls[:, 1])
tree_ll_sem = stats.sem(model_lls[:, 0])
notree_ll_sem = stats.sem(model_lls[:, 1])

# Take the average and sem for each network
nll_DL_user = np.mean(np.asarray(nll_DL_all))
nll_DL_sem = stats.sem(np.asarray(nll_DL_all))
nll_DCNN_user = np.mean(np.asarray(nll_DCNN_all))
nll_DCNN_sem = stats.sem(np.asarray(nll_DCNN_all))
nll_BasicL_user = np.mean(np.asarray(nll_BasicL_all))
nll_BasicL_sem = stats.sem(np.asarray(nll_BasicL_all))
nll_BasicCNN_user = np.mean(np.asarray(nll_BasicCNN_all))
nll_BasicCNN_sem = stats.sem(np.asarray(nll_BasicCNN_all))

# Plot the model comparison
lls = np.asarray([notree_ll, tree_ll, nll_BasicCNN_user, nll_BasicL_user, nll_DCNN_user, nll_DL_user])
models = ( 'Myopic', 'Full', 'CN', 'FC', 'CN', 'FC')
x_pos = np.arange(len(models))
fig, ax = plt.subplots(figsize=(7,5))
ax.bar(x_pos, lls-2, align='center', width = 0.8, color = ['firebrick', 'firebrick', 'mediumpurple', 'mediumpurple','darkblue', 'darkblue'],
       yerr=[tree_ll_sem, notree_ll_sem, nll_BasicCNN_sem, nll_BasicL_sem, nll_DCNN_sem, nll_DL_sem], capsize=5)
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
pos = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
ax.set(yticks=pos)
ax.set_yticklabels(['2.0', '2.1', '2.2', '2.3', '2.4', '2.5'])
ax.set_ylabel('Negative log-likelihood')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
custom_lines = [Line2D([0], [0], color='firebrick', lw=5),
                Line2D([0], [0], color='mediumpurple', lw=5),
                Line2D([0], [0], color='darkblue', lw=5)]
ax.legend(custom_lines, ['cognitive models', 'lesioned neural networks', 'full neural networks'])
plt.show()
# plt.savefig('comparison.png', format='png', dpi=1000, bbox_inches='tight')

# Plot accuracy as a function of number of guesses for the best network
guesses_accuracy = test_with_guesses(net_DL, test_set)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(np.arange(1,37), guesses_accuracy, lw=2, color='darkblue', marker='o')
ax.set_xlim(1,37)
ax.set_ylim(0,1.05)
ax.set_xlabel('Number of guesses')
ax.set_ylabel('Accuracy')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('guesses.png', format='png', dpi=1000, bbox_inches='tight')

# Plot accuracy as a function of move number for the best network
moves_accuracy = test_by_move(net_DL, test_set)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(np.arange(1,37,2), moves_accuracy, lw=2, color='darkblue')
ax.set_xlim(1,37)
ax.set_xlabel('Move number')
ax.set_ylabel('Accuracy')
ax.set_ylim(0,1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('move.png', format='png', dpi=1000, bbox_inches='tight')

#%% CONVOLUTIONAL FEATURE MAPS

# Function to visualize feature maps
def plot_kernels(tensor, num_cols=4):
    num_kernels = tensor.shape[0]
    num_rows = 1 + num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(num_cols):
        ax1 = fig.add_subplot(num_rows, num_cols, i+1)
        ax2 = fig.add_subplot(num_rows, num_cols, i+num_cols+1)
        ax1.imshow(tensor[i][0], cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax2.imshow(tensor[i][1], cmap='gray')
        ax2.axis('off')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])

    plt.show()
    # plt.savefig('filters4.png', format='png', dpi=1000, bbox_inches='tight')

# Pick the first and last convolutional layers and plot them
layer1 = [i for i in net_DCNN.children()][0]
layer4 = [i for i in net_DCNN.children()][3]
tensor1 = layer1.weight.data.numpy()
tensor4 = layer4.weight.data.numpy()
plot_kernels(tensor1)
plot_kernels(tensor4)
