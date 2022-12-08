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

plt.rcParams.update({'font.size': 16})

#%% LOADING IN THE RESULTS FROM THE BASELINE COGNITIVE MODEL
test_path = '../../fits/'

# Load in the text file with the board positions and results
results_path = '../../networks/24/results_file.txt'
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

#%% LOADING IN THE RESULTS FROM IMPROVED COGNITIVE MODELS
test_path_1 = '../../fits_7/'
test_ll_1 = np.zeros(num_moves)
moves_1 = np.zeros((num_moves, 36))

counter = 0
for i in tqdm.tqdm(range(55)):
    curr_ll = np.loadtxt(test_path_1+'out'+str(i)+'_lltest.csv', delimiter=',')
    curr_moves = np.loadtxt(test_path_1+'out'+str(i)+'_moves.csv', delimiter=',')
    test_ll_1[counter:counter+np.shape(curr_ll)[0]] = curr_ll
    moves_1[counter:counter+np.shape(curr_moves)[0],:] = curr_moves
    counter += np.shape(curr_ll)[0]

test_path_2 = '../../fits_8/'
test_ll_2 = np.zeros(num_moves)
moves_2 = np.zeros((num_moves, 36))

counter = 0
for i in tqdm.tqdm(range(55)):
    curr_ll = np.loadtxt(test_path_2+'out'+str(i)+'_lltest.csv', delimiter=',')
    curr_moves = np.loadtxt(test_path_2+'out'+str(i)+'_moves.csv', delimiter=',')
    test_ll_2[counter:counter+np.shape(curr_ll)[0]] = curr_ll
    moves_2[counter:counter+np.shape(curr_moves)[0],:] = curr_moves
    counter += np.shape(curr_ll)[0]


#%% Plot the comparisons
x = ['baseline', 'corner', 'defensive']
model_avgs = [np.average(test_ll), np.average(test_ll_1), np.average(test_ll_2)]
model_errors = [stats.sem(test_ll), stats.sem(test_ll_1), stats.sem(test_ll_2)]
colors = ['darkorange', 'peachpuff', 'peachpuff']

fig, ax = plt.subplots(figsize=(5,4))
ax.bar(x, model_avgs, yerr=model_errors, color=colors)
ax.set_ylabel('Negative log-likelihood')
ax.set_ylim(2.1, 2.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
# plt.savefig('improvement_comparison.png', format='png', dpi=1000, bbox_inches='tight')

#%% Compare the likelihood histograms between two models

logs_diff = test_ll_2 - test_ll

fig, ax = plt.subplots()
ax.hist(logs_diff,edgecolor='white', color='darkorange',bins=np.arange(-2,2.25,.25))
ax.vlines(0,0,1600000,linewidth=2,linestyle='--',color='black')
ax.set_xlabel('Difference in log-likelihood per move')
ax.set_ylabel('Number of moves')
ax.set_xlim(-2,2)
ax.set_ylim(0,1600000)
ax.set_yticks([0,500000,1000000,1500000])
ax.set_yticklabels(['0','500k','1M','1.5M'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.75, 0.9,'Evidence favoring \n baseline model',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
ax.text(0.25, 0.9,'Evidence favoring \n defensive model',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)

plt.show()
# plt.savefig('comparison.png', format='png', dpi=1000, bbox_inches='tight')


#%% LOADING IN THE RESULTS FROM A COMPARISON NETWORK

# Load in the text file with the board positions and results
results_path = '../../networks/24/results_file.txt'
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

#%% LOADING IN THE RESULTS FROM A TESTED COGNITIVE MODEL
test_path = '../../fits_1/'

test_ll_1 = np.zeros(num_moves)
moves_1 = np.zeros((num_moves, 36))

counter = 0
for i in tqdm.tqdm(range(55)):
    # print(i, counter)
    curr_ll = np.loadtxt(test_path+'out'+str(i)+'_lltest.csv', delimiter=',')
    curr_moves = np.loadtxt(test_path+'out'+str(i)+'_moves.csv', delimiter=',')
    test_ll_1[counter:counter+np.shape(curr_ll)[0]] = curr_ll
    moves_1[counter:counter+np.shape(curr_moves)[0],:] = curr_moves
    counter += np.shape(curr_ll)[0]

test_path = '../../fits_5/'

test_ll_2 = np.zeros(num_moves)
moves_2 = np.zeros((num_moves, 36))

counter = 0
for i in tqdm.tqdm(range(55)):
    # print(i, counter)
    curr_ll = np.loadtxt(test_path+'out'+str(i)+'_lltest.csv', delimiter=',')
    curr_moves = np.loadtxt(test_path+'out'+str(i)+'_moves.csv', delimiter=',')
    test_ll_2[counter:counter+np.shape(curr_ll)[0]] = curr_ll
    moves_2[counter:counter+np.shape(curr_moves)[0],:] = curr_moves
    counter += np.shape(curr_ll)[0]

#%%
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

#%% FIND LARGEST DIFFERENCES BETWEEN TWO MODELS

# Look at all positions in the dataset and compute the KL divergence
kl_models = []

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
        hist_model_1 = moves_1[ind,:]
        hist_model_2 = moves_2[ind, :]
        prediction_model_1 = np.argmax(hist_model_1)
        prediction_model_2 = np.argmax(hist_model_2)

        # Normalize the nn output and compute the kl divergence
        # output_norm = torch.from_numpy(output) - torch.logsumexp(torch.from_numpy(output), 0)
        kl_divergence = kl_loss(torch.from_numpy(hist_model_1/np.sum(hist_model_1)), torch.from_numpy(hist_model_2/np.sum(hist_model_2)))
        kl_models.append(kl_divergence.item())

#%% Look at board positions between the models

# Sort boards by kl divergence
inds_sort = np.argsort(kl_models)
kl_sort = np.sort(kl_models)

# Look at k boards
k = 10

# Visualize the top or bottom boards with their outputs
for i in range(k):
        curr_ind = inds_sort[len(inds_sort)-1-i]
        board = boards[curr_ind, :]
        output = outputs[curr_ind, :]
        target = targets[curr_ind, :]

        hist_model_1 = moves_1[curr_ind,:]
        hist_model_2 = moves_2[curr_ind, :]
        model_1 = np.argmax(hist_model_1)
        model_2 = np.argmax(hist_model_2)

        visualize_diff(board, output, target, model_1, model_2)