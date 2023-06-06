"""
Main file to analyze improvements in the cognitive model
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy import stats
from matplotlib import colors, patches
import torch
import torch.nn as nn
import datashader as ds
from datashader.mpl_ext import dsshow
import pandas as pd

plt.rcParams.update({'font.size': 16})

#%% Loading in the results from the baseline model

test_path = '../../Data/test_fits_baseline/'

# Load in the text file with the board positions and results
results_path = '../../Data/networks/24/results_file.txt'
results_file = open(results_path, 'r')
results_lines = results_file.read().splitlines()

# Transform the results into separate numpy arrays
num_moves = len(results_lines)

test_ll = np.zeros(num_moves)
moves = np.zeros((num_moves, 36))

counter = 0
for i in tqdm.tqdm(range(55)):
    curr_ll = np.loadtxt(test_path+'out'+str(i)+'_lltest.csv', delimiter=',')
    curr_moves = np.loadtxt(test_path+'out'+str(i)+'_moves.csv', delimiter=',')
    test_ll[counter:counter+np.shape(curr_ll)[0]] = curr_ll
    moves[counter:counter+np.shape(curr_moves)[0],:] = curr_moves
    counter += np.shape(curr_ll)[0]

#%% Loading in the results from the improved cognitive models

test_path_1 = '../../Data/test_fits_corner/'
test_ll_1 = np.zeros(num_moves)
moves_1 = np.zeros((num_moves, 36))

counter = 0
for i in tqdm.tqdm(range(55)):
    curr_ll = np.loadtxt(test_path_1+'out'+str(i)+'_lltest.csv', delimiter=',')
    curr_moves = np.loadtxt(test_path_1+'out'+str(i)+'_moves.csv', delimiter=',')
    test_ll_1[counter:counter+np.shape(curr_ll)[0]] = curr_ll
    moves_1[counter:counter+np.shape(curr_moves)[0],:] = curr_moves
    counter += np.shape(curr_ll)[0]

test_path_2 = '../../Data/test_fits_defensive/'
test_ll_2 = np.zeros(num_moves)
moves_2 = np.zeros((num_moves, 36))

counter = 0
for i in tqdm.tqdm(range(55)):
    curr_ll = np.loadtxt(test_path_2+'out'+str(i)+'_lltest.csv', delimiter=',')
    curr_moves = np.loadtxt(test_path_2+'out'+str(i)+'_moves.csv', delimiter=',')
    test_ll_2[counter:counter+np.shape(curr_ll)[0]] = curr_ll
    moves_2[counter:counter+np.shape(curr_moves)[0],:] = curr_moves
    counter += np.shape(curr_ll)[0]

test_path_3 = '../../Data/test_fits_3iar/'
test_ll_3 = np.zeros(num_moves)
moves_3 = np.zeros((num_moves, 36))

counter = 0
for i in tqdm.tqdm(range(55)):
    curr_ll = np.loadtxt(test_path_3+'out'+str(i)+'_lltest.csv', delimiter=',')
    curr_moves = np.loadtxt(test_path_3+'out'+str(i)+'_moves.csv', delimiter=',')
    test_ll_3[counter:counter+np.shape(curr_ll)[0]] = curr_ll
    moves_3[counter:counter+np.shape(curr_moves)[0],:] = curr_moves
    counter += np.shape(curr_ll)[0]

#%% Test/train likelihood plot

model_avgs = [np.average(test_ll), np.average(test_ll_1), np.average(test_ll_2), np.average(test_ll_3)]
model_errors = [stats.sem(test_ll), stats.sem(test_ll_1), stats.sem(test_ll_2), stats.sem(test_ll_3)]

train = [2.217, 2.189, 2.198, 2.212, 2.207, 2.198, 2.197, 2.194, 2.193, 2.190,
        2.191, 2.190, 2.202, 2.194, 2.199, 2.198, 2.195, 2.196, 2.195, 2.188]
train_1 = [2.195, 2.199, 2.199, 2.201, 2.205, 2.211, 2.211, 2.190, 2.208, 2.218,
           2.194, 2.241, 2.201, 2.213, 2.207, 2.193, 2.194, 2.194, 2.190, 2.245]
train_2 = [2.184, 2.176, 2.184, 2.160, 2.161, 2.154, 2.151, 2.167, 2.156, 2.163,
           2.166, 2.162, 2.177, 2.163, 2.158, 2.153, 2.172, 2.157, 2.178, 2.205]
train_3 = [2.156, 2.165, 2.189, 2.172, 2.167, 2.210, 2.158, 2.210, 2.202, 2.156,
           2.159, 2.183, 2.204, 2.168, 2.208, 2.154, 2.211, 2.179, 2.152, 2.159]

train_avgs = [np.min(train), np.min(train_1), np.min(train_2), np.min(train_3)]
# train_errors = [stats.sem(train), stats.sem(train_1), stats.sem(train_2), stats.sem(train_3)]

fig, ax = plt.subplots(figsize=(6,4))
ax.bar([0.075, 0.325, 0.575, 0.825], train_avgs, width=0.05, color='peachpuff', label='train')
ax.bar([0.125, 0.375, 0.625, 0.875], model_avgs, yerr=model_errors, width=0.05, color='darkorange', label='test')
# ax.scatter(np.concatenate([np.repeat(0.1,20), np.repeat(0.35,20), np.repeat(0.6,20), np.repeat(0.85,20)]),
#            np.concatenate([train, train_1, train_2, train_3]), color='darkorange', alpha=0.2, label='train')
ax.set_ylabel('Negative log-likelihood')
ax.set_ylim(2.1,2.3)
ax.set_xlim(0,0.95)
ax.set_xticks([0.1, 0.35, 0.6, 0.85])
ax.set_xticklabels(['Baseline', 'Opening\nbias','Defensive\nweighting', 'Phantom\nfeatures'])
ax.legend(frameon=False,loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plt.show()
plt.savefig('improvement_comparison.png', format='png', dpi=1000, bbox_inches='tight')

#%% Comparing likelihoods per move between models (density plot)

fig, ax = plt.subplots()
plot = plt.hist2d(test_ll, test_ll_2, bins=(200, 200), cmap=plt.cm.Oranges, norm = colors.LogNorm())
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', lw=2, ls='--')
ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.set_xticks([0,5,10])
ax.set_yticks([0,5,10])
ax.set_xlabel('Baseline model\nnegative log-likelihood')
ax.set_ylabel('Defensive weighting model\nnegative log-likelihood')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
cbar = plt.colorbar(plot[3])
cbar.minorticks_off()
cbar.ax.set_ylabel('Number of moves', rotation=-90, labelpad=20)
plt.show()
# plt.savefig('comparison_final.png', format='png', dpi=1000, bbox_inches='tight')


#%% Comparing likelihoods per move between models (histogram)

logs_diff = test_ll_2 - test_ll

fig, ax = plt.subplots()
ax.hist(logs_diff,edgecolor='white', color='darkorange',bins=np.arange(-2,2.25,.25))
ax.vlines(0,0,1800000,linewidth=2,linestyle='--',color='black')
ax.set_xlabel('Difference in log-likelihood per move')
ax.set_ylabel('Number of moves')
ax.set_xlim(-2,2)
ax.set_ylim(0,1800000)
ax.set_yticks([0,500000,1000000,1500000])
ax.set_yticklabels(['0','500k','1M', '1.5M'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.75, 0.9,'Evidence favoring \n baseline model',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
ax.text(0.25, 0.9,'Evidence favoring \n final model',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)

plt.show()
# plt.savefig('comparison_final.png', format='png', dpi=1000, bbox_inches='tight')


#%% Loading in the results from a comparison network

# Load in the text file with the board positions and results
results_path = '../../Data/networks/24/results_file.txt'
results_file = open(results_path, 'r')
results_lines = results_file.read().splitlines()

# Transform the results into separate numpy arrays
num_moves = len(results_lines)
boards = np.zeros((num_moves, 36))
outputs = np.zeros((num_moves, 36))
predictions = np.zeros((num_moves, 1))
targets = np.zeros((num_moves, 1))
counter = 0
for line in tqdm.tqdm(results_lines):
        # Break down the line from the text file into its components
        line_list = [float(s) for s in line.split(',')]
        boards[counter, :] = [int(f) for f in line_list[0:36]]
        outputs[counter, :] = line_list[36:72]
        predictions[counter, :] = int(line_list[72])
        targets[counter, :] = int(line_list[73])

        counter += 1

# %% Accuracy per model

# Initialize a numpy array for each move (number correct, total number)
accuracy = 0
accuracy_1 = 0
accuracy_2 = 0
accuracy_3 = 0

# Define the function to compute losses
loss = nn.CrossEntropyLoss()

# For each board position
for ind in tqdm.tqdm(range(num_moves)):
    # Define the components we need
    board = boards[ind, :].astype(np.int)
    output = outputs[ind, :]
    prediction = int(predictions[ind, :][0])
    target = int(targets[ind, :][0])

    # Get the cognitive model predictions
    # hist_model = np.flipud(moves[ind,:].reshape(4, 9, order='F')).flatten(order='F')
    hist_model = moves[ind, :]
    prediction_model = np.argmax(hist_model)
    hist_model1 = moves_1[ind, :]
    prediction_model1 = np.argmax(hist_model1)
    hist_model2 = moves_2[ind, :]
    prediction_model2 = np.argmax(hist_model2)
    hist_model3 = moves_3[ind, :]
    prediction_model3 = np.argmax(hist_model3)

    # Compare with ground truth
    accuracy += np.equal(prediction_model, target)
    accuracy_1 += np.equal(prediction_model1, target)
    accuracy_2 += np.equal(prediction_model2, target)
    accuracy_3 += np.equal(prediction_model3, target)

print(num_moves-accuracy)
accuracy = accuracy/num_moves
accuracy_1 = accuracy_1/num_moves
accuracy_2 = accuracy_2/num_moves
accuracy_3 = accuracy_3/num_moves

#%% Visualizing network output

# Function that takes a board and model output and visualizes it
def visualize_pred(board, output, target, model, outputOn=True, targetOn=True, modelOn=True, save=False, filename=''):
        # Flip to correct dimensions
        board = board.reshape(9,4).T.flatten()
        output = output.reshape(9,4).T.flatten()
        target = np.ravel_multi_index(np.unravel_index(target.astype(int), (9,4)), (9,4), order='F')
        model = np.ravel_multi_index(np.unravel_index(model.astype(int), (9, 4)), (9, 4), order='F')
        # modeloutput = modeloutput.reshape(9,4).T.flatten()

        # Preprocessing
        black_pieces = [index for index, element in enumerate(board) if element == 1]
        white_pieces = [index for index, element in enumerate(board) if element == -1]
        output_moves = np.exp(np.asarray(output))
        output_moves /= np.sum(output_moves)
        # modeloutput_moves = np.exp(np.asarray(modeloutput))
        # modeloutput_moves /= np.sum(modeloutput_moves)

        # Create the figure and colormap
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.vlines(np.arange(-0.5, 9.5, 1), -0.5, 3.5, color='black')
        ax.hlines(np.arange(-0.5, 4.5, 1), -0.5, 8.5, color='black')
        cm = colors.LinearSegmentedColormap.from_list('gray_red_map', [colors.to_rgb('darkgray'),
                                                                colors.to_rgb('red')], N=100)
        # cm_alt = colors.LinearSegmentedColormap.from_list('gray_orange_map', [colors.to_rgb('darkgray'),
        #                                                         colors.to_rgb('orange')], N=100)

       # Loop through the black and white pieces and place them
        for p in black_pieces:
                if p < 9:
                    p = 27+p
                elif 8 < p < 18:
                    p = p+9
                elif 17 < p < 27:
                    p = p-9
                else:
                    p = p-27
                circ = patches.Circle((p%9, p//9), 0.33, color="black", fill=True)
                ax.add_patch(circ)
        for p in white_pieces:
                if p < 9:
                    p = 27+p
                elif 8 < p < 18:
                    p = p+9
                elif 17 < p < 27:
                    p = p-9
                else:
                    p = p-27
                circ = patches.Circle((p%9,p//9),0.33,color="white",fill=True)
                ax.add_patch(circ)

        # Now place the target
        if targetOn is True:
                p = target
                if p < 9:
                        p = 27 + p
                elif 8 < p < 18:
                        p = p + 9
                elif 17 < p < 27:
                        p = p - 9
                else:
                        p = p - 27
                circ = patches.Circle((p % 9, p // 9), 0.33, color="black", linewidth=4, fill=False)
                ax.add_patch(circ)

        # Now place the target
        if modelOn is True:
                p = model
                if p < 9:
                        p = 27 + p
                elif 8 < p < 18:
                        p = p + 9
                elif 17 < p < 27:
                        p = p - 9
                else:
                        p = p - 27
                circ = patches.Circle((p % 9, p // 9), 0.33, color="black", linewidth=4, ls='--', fill=False)
                ax.add_patch(circ)

        # Plot with the likelihood mapping
        if outputOn is True:
                plt.imshow(np.flip(np.reshape(output_moves, [4, 9]),axis=0), cmap=cm, interpolation='nearest', origin='lower', vmin=0, vmax=0.5)
        else:
                plt.imshow(np.flip(np.reshape(output_moves, [4, 9]),axis=0), cmap=cm, interpolation='nearest', origin='lower', vmin=0, vmax=100)

        # if modeloutputOn is True:
        #         plt.imshow(np.flip(np.reshape(modeloutput_moves, [4, 9]),axis=0), cmap=cm_alt, interpolation='nearest', origin='lower', vmin=0, vmax=0.5)


        ax.axis('off')
        fig.tight_layout()
        if save is True:
                plt.savefig(filename, format='png', dpi=1000, bbox_inches='tight')
        else:
                plt.show()


def visualize_diff(board, output, target, model1, model2, save=False, filename=''):
    # Flip to correct dimensions
    board = board.reshape(9, 4).T.flatten()
    output = output.reshape(9, 4).T.flatten()
    target = np.ravel_multi_index(np.unravel_index(target.astype(int), (9, 4)), (9, 4), order='F')
    model1 = np.ravel_multi_index(np.unravel_index(model1.astype(int), (9, 4)), (9, 4), order='F')
    model2 = np.ravel_multi_index(np.unravel_index(model2.astype(int), (9, 4)), (9, 4), order='F')

    # Preprocessing
    black_pieces = [index for index, element in enumerate(board) if element == 1]
    white_pieces = [index for index, element in enumerate(board) if element == -1]
    output_moves = np.exp(np.asarray(output))
    output_moves /= np.sum(output_moves)

    # Create the figure and colormap
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.vlines(np.arange(-0.5, 9.5, 1), -0.5, 3.5, color='black')
    ax.hlines(np.arange(-0.5, 4.5, 1), -0.5, 8.5, color='black')
    cm = colors.LinearSegmentedColormap.from_list('gray_red_map', [colors.to_rgb('darkgray'),
                                                                   colors.to_rgb('red')], N=100)

    plt.imshow(np.flip(np.reshape(output_moves, [4, 9]), axis=0), cmap=cm, interpolation='nearest', origin='lower',
               vmin=0, vmax=100)

    # Loop through the black and white pieces and place them
    for p in black_pieces:
        if p < 9:
            p = 27 + p
        elif 8 < p < 18:
            p = p + 9
        elif 17 < p < 27:
            p = p - 9
        else:
            p = p - 27
        circ = patches.Circle((p % 9, p // 9), 0.33, color="black", fill=True)
        ax.add_patch(circ)
    for p in white_pieces:
        if p < 9:
            p = 27 + p
        elif 8 < p < 18:
            p = p + 9
        elif 17 < p < 27:
            p = p - 9
        else:
            p = p - 27
        circ = patches.Circle((p % 9, p // 9), 0.33, color="white", fill=True)
        ax.add_patch(circ)

    # Now place the target
    p = target
    if p < 9:
        p = 27 + p
    elif 8 < p < 18:
        p = p + 9
    elif 17 < p < 27:
        p = p - 9
    else:
        p = p - 27
    circ = patches.Circle((p % 9, p // 9), 0.33, color="black", linewidth=4, fill=False)
    ax.add_patch(circ)

    # Now place the target
    p = model1
    if p < 9:
        p = 27 + p
    elif 8 < p < 18:
        p = p + 9
    elif 17 < p < 27:
        p = p - 9
    else:
        p = p - 27
    circ = patches.Circle((p % 9, p // 9), 0.33, color="red", linewidth=4, ls='--', fill=False)
    ax.add_patch(circ)

    p = model2
    if p < 9:
        p = 27 + p
    elif 8 < p < 18:
        p = p + 9
    elif 17 < p < 27:
        p = p - 9
    else:
        p = p - 27
    circ = patches.Circle((p % 9, p // 9), 0.33, color="blue", linewidth=4, ls='--', fill=False)
    ax.add_patch(circ)

    ax.axis('off')
    fig.tight_layout()
    if save is True:
        plt.savefig(filename, format='png', dpi=1000, bbox_inches='tight')
    else:
        plt.show()

#%% Finding the largest differences bettween the network and model or two different models

# Look at all positions in the dataset and compute the KL divergence
kl_final = []

kl_loss = nn.KLDivLoss(log_target=False, reduction='batchmean')

# For each board position
for ind in tqdm.tqdm(range(num_moves)):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Get the cognitive model predictions
        # hist_model = np.flipud(moves[ind,:].reshape(4, 9, order='F')).flatten(order='F')
        hist_model_2 = moves_2[ind,:]
        prediction_model_2 = np.argmax(hist_model_2)

        # Normalize the nn output and compute the kl divergence
        output_norm = torch.from_numpy(output) - torch.logsumexp(torch.from_numpy(output), 0)
        kl_divergence = kl_loss(output_norm, torch.from_numpy(hist_model_2/np.sum(hist_model_2)))
        # kl_divergence = kl_loss(torch.from_numpy(hist_model_1/np.sum(hist_model_1)), torch.from_numpy(hist_model_2/np.sum(hist_model_2)))
        kl_final.append(kl_divergence.item())

#%% Look at board positions between the nn/model or two models

# Sort boards by kl divergence
inds_sort = np.argsort(kl_final)
kl_sort = np.sort(kl_final)

# Look at k boards
k = 50

# Visualize the top or bottom boards with their outputs
for i in range(k):
        curr_ind = inds_sort[len(inds_sort)-1-i]
        board = boards[curr_ind, :]
        output = outputs[curr_ind, :]
        target = targets[curr_ind, :]

        # hist_model_1 = moves_1[curr_ind,:]
        # hist_model_2 = moves_2[curr_ind, :]
        # model_1 = np.argmax(hist_model_1)
        # model_2 = np.argmax(hist_model_2)

        # visualize_diff(board, output, target, model_1, model_2)

        hist_model_2 = moves_2[curr_ind, :]
        model_2 = np.argmax(hist_model_2)

        # visualize_pred(board, output, target, model_3)
        visualize_pred(board, output, target, model_2, True, True, True, True, str(i)+'.png')
