"""
Main file to test and analyze networks
"""

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
from network import Linear, LinearSkip

#%% NETWORK COMPARISON

layers = [5, 10, 20, 40]
val_loss200 = [2.095, 2.063, 2.045, 2.030]
val_loss500 = [2.062, 2.042, 2.027, 2.014]
val_loss1000 = [2.046, 2.027, 2.014, 2.002]
val_loss2000 = [2.031, 2.014, 2.000, 1.992]

# 'mistyrose','darksalmon','red','firebrick','maroon'
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(layers, val_loss200, lw=2, color='maroon', marker='o')
ax.plot(layers, val_loss500, lw=2, color='firebrick', marker='o')
ax.plot(layers, val_loss1000, lw=2, color='red', marker='o')
ax.plot(layers, val_loss2000, lw=2, color='darksalmon', marker='o')
ax.set_xlabel('Number of hidden layers')
ax.set_ylabel('Negative log-likelihood')
ax.legend(['200 units', '500 units', '1000 units', '2000 units'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('val_comparison.png', format='png', dpi=1000, bbox_inches='tight')

#%% LEARNING CURVES

# Load and plot the learning curves
losses = torch.load('../networks/21/losses_9')
train_loss = losses['train_loss']
val_loss = losses['val_loss']

# Plot the learning curves for train and validation
def plot_learning(train_loss, val_loss, lb=1.9, ub=3.0):
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(np.arange(1,len(train_loss)+1), train_loss, lw=2, color='darkblue', marker='o')
        ax.plot(np.arange(1,len(train_loss)+1), val_loss, lw=2, color='cornflowerblue', marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Negative log-likelihood')
        ax.set_ylim(lb, ub)
        # Add vertical lines for when lr was changed
        # for epoch in lr:
        #     plt.axvline(x=epoch, color='firebrick', linestyle='--')
        ax.legend(['train', 'validation'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plt.show()
        plt.savefig('learning_16.png', format='png', dpi=1000, bbox_inches='tight')

plot_learning(train_loss, val_loss)

#%% ANALYZING TEST RESULTS

# Load in the text file with board positions
results_path = '../networks/1/results_file.txt'
results_file = open(results_path, 'r')
results_lines = results_file.read().splitlines()

# Initialize a numpy array with number of guesses
guesses = np.zeros(36)

# Initialize a numpy array for each move (number correct, total number)
moves = np.zeros(36)
totals = np.zeros(36)

# For each line in the text file
for line in results_lines:
        # Break down the line from the text file into its components
        line_list = [float(s) for s in line.split(',')]
        board = [int(f) for f in line_list[0:36]]
        output = line_list[36:72]
        prediction = int(line_list[72])
        target = int(line_list[73])

        # Compute number of guesses
        preds, idxs = torch.topk(torch.tensor(output), 36, sorted=True)
        # Now iterate through the sorted values until one matches the ground truth
        for guess in range(len(preds)):
                if torch.eq(idxs[guess], target):
                        guesses[guess] += 1
                        break

        # For the current move number, get the prediction and compare with ground truth
        move_num = np.sum(np.absolute(board))
        moves[move_num] += np.equal(prediction, target)
        totals[move_num] += 1

# Convert to cumulative percent accuracy
guesses_accuracy = np.zeros(36)
for i in range(len(guesses)):
        guesses_accuracy[i] = np.sum(guesses[0:i+1])/np.sum(guesses)

# Divide to compute the accuracy, remove the nans and  return
move_accuracy = moves/totals
move_accuracy = move_accuracy[~np.isnan(move_accuracy)]

#%% PLOTS

# Plot accuracy as a function of number of guesses
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

# Plot accuracy as a function of move number
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(np.arange(1,37,2), move_accuracy, lw=2, color='darkblue')
ax.set_xlim(1,37)
ax.set_xlabel('Move number')
ax.set_ylabel('Accuracy')
ax.set_ylim(0,1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('move.png', format='png', dpi=1000, bbox_inches='tight')

#%% OLDER CODE

# #%% LOADING A TRAINED MODEL
#
# # Just the state dicts for four models used in course paper
# net_DL = DeepLinear()
# net_DL.load_state_dict(torch.load('saved_checkpoints/DeepLinear5_200_data.pth'))
# net_DL.eval()
#
# net_BasicL = BasicLinear()
# net_BasicL.load_state_dict(torch.load('saved_checkpoints/BasicLinear_200_data.pth'))
# net_BasicL.eval()
#
# net_DCNN = DeepCNN(num_filters=4, filter_size=3)
# net_DCNN.load_state_dict(torch.load('saved_checkpoints/DeepCNN5_4_data.pth'))
# net_DCNN.eval()
#
# net_BasicCNN = BasicCNN(num_filters=4, filter_size=3)
# net_BasicCNN.load_state_dict(torch.load('saved_checkpoints/BasicCNN_4_data.pth'))
# net_BasicCNN.eval()
#
# # The entire model
# # TODO: decide what to do with optimizer to restart training
# net_DL_full = DeepLinear()
# optimizer = optim.SGD(net_DL_full.parameters(), lr=0.1, weight_decay=0)
# checkpoint = torch.load('fullDeepLinear5_200_data.pth')
# net_DL_full.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# running_loss = checkpoint['running_loss']
# train_loss = checkpoint['train_loss']
# val_loss = checkpoint['val_loss']
# net_DL_full.eval()
# # - or -
# # net.train()
#
# #%% TESTING AND ANALYSIS
#
# # Run the network on the test set, plot the NLL compared to the cognitive models
# perc_correct_DCNN, nll_DCNN = test_performance(net_DCNN, test_set)
# perc_correct_DL, nll_DL = test_performance(net_DL, test_set)
# perc_correct_BasicCNN, nll_BasicCNN = test_performance(net_BasicCNN, test_set)
# perc_correct_BasicL, nll_BasicL = test_performance(net_BasicL, test_set)
#
# # Run the network on the test set by individual user, for averaging and comparison with the cognitive model
#
# with open('larger_data/test_paths.txt', 'r') as filehandle:
#     test_paths = json.load(filehandle)
#
# # Set a dictionary with each user as a key and each of their paths as the values in a list
# user_paths = {}
# for path in test_paths:
#     curr_user = path[36:46]
#     if curr_user not in user_paths.keys():
#         user_paths[curr_user] = [path]
#     else:
#         user_paths[curr_user].append(path)
#
# perc_correct_DCNN_all, nll_DCNN_all = test_performance_user(net_DCNN, user_paths)
# perc_correct_DL_all, nll_DL_all = test_performance_user(net_DL, user_paths)
# perc_correct_BasicCNN_all, nll_BasicCNN_all = test_performance_user(net_BasicCNN, user_paths)
# perc_correct_BasicL_all, nll_BasicL_all = test_performance_user(net_BasicL, user_paths)
#
# # Read in already formatted log-likelihoods and change to numpy array
# model_lls = pd.read_csv('cognitive_model/loglik_tree_notree.txt', sep=" ", header=None)
# model_lls = model_lls.values
# tree_ll = np.average(model_lls[:, 0])
# notree_ll = np.average(model_lls[:, 1])
# tree_ll_sem = stats.sem(model_lls[:, 0])
# notree_ll_sem = stats.sem(model_lls[:, 1])
#
# # Take the average and sem for each network
# nll_DL_user = np.mean(np.asarray(nll_DL_all))
# nll_DL_sem = stats.sem(np.asarray(nll_DL_all))
# nll_DCNN_user = np.mean(np.asarray(nll_DCNN_all))
# nll_DCNN_sem = stats.sem(np.asarray(nll_DCNN_all))
# nll_BasicL_user = np.mean(np.asarray(nll_BasicL_all))
# nll_BasicL_sem = stats.sem(np.asarray(nll_BasicL_all))
# nll_BasicCNN_user = np.mean(np.asarray(nll_BasicCNN_all))
# nll_BasicCNN_sem = stats.sem(np.asarray(nll_BasicCNN_all))
#
# # Plot the model comparison
# lls = np.asarray([notree_ll, tree_ll, nll_BasicCNN_user, nll_BasicL_user, nll_DCNN_user, nll_DL_user])
# models = ( 'Myopic', 'Full', 'CN', 'FC', 'CN', 'FC')
# x_pos = np.arange(len(models))
# fig, ax = plt.subplots(figsize=(7,5))
# ax.bar(x_pos, lls-2, align='center', width = 0.8, color = ['firebrick', 'firebrick', 'mediumpurple', 'mediumpurple','darkblue', 'darkblue'],
#        yerr=[tree_ll_sem, notree_ll_sem, nll_BasicCNN_sem, nll_BasicL_sem, nll_DCNN_sem, nll_DL_sem], capsize=5)
# ax.set_xticks(x_pos)
# ax.set_xticklabels(models)
# pos = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
# ax.set(yticks=pos)
# ax.set_yticklabels(['2.0', '2.1', '2.2', '2.3', '2.4', '2.5'])
# ax.set_ylabel('Negative log-likelihood')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# custom_lines = [Line2D([0], [0], color='firebrick', lw=5),
#                 Line2D([0], [0], color='mediumpurple', lw=5),
#                 Line2D([0], [0], color='darkblue', lw=5)]
# ax.legend(custom_lines, ['cognitive models', 'lesioned neural networks', 'full neural networks'])
# plt.show()
# # plt.savefig('comparison.png', format='png', dpi=1000, bbox_inches='tight')