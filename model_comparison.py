"""
Main file to analyze cognitive model fits
"""

# NOTE: use the flips for the original model comparison, but not for any improvements

#%%
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib import colors, patches
import numpy as np
import torch
import tqdm

plt.rcParams.update({'font.size': 16})

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
test_path = '../../fits_5/'

test_ll = np.zeros(num_moves)
moves = np.zeros((num_moves, 36))

counter = 0
for i in tqdm.tqdm(range(55)):
    # print(i, counter)
    curr_ll = np.loadtxt(test_path+'out'+str(i)+'_lltest.csv', delimiter=',')
    curr_moves = np.loadtxt(test_path+'out'+str(i)+'_moves.csv', delimiter=',')
    test_ll[counter:counter+np.shape(curr_ll)[0]] = curr_ll
    moves[counter:counter+np.shape(curr_moves)[0],:] = curr_moves
    counter += np.shape(curr_ll)[0]

#%% COMPARE THE NETWORK AND MODEL OUTPUTS

# Initialize counter for the overall LL
LL_nn = np.zeros(num_moves)
LL_rand = np.zeros(num_moves)

# Initialize a numpy array for each move (number correct, total number)
moves_nn = np.zeros(36)
moves_model = np.zeros(36)
moves_rand = np.zeros(36)
totals = np.zeros(36)

# Define the function to compute losses
loss = nn.CrossEntropyLoss()

# For each board position
for ind in tqdm.tqdm(range(num_moves)):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Get the likelihoods
        loss_size = loss(torch.from_numpy(output.reshape(1, 36)), torch.from_numpy(np.array([target])))
        LL_nn[ind] = loss_size.item()

        # Get the cognitive model predictions
        hist_model = np.flipud(moves[ind,:].reshape(4, 9, order='F')).flatten(order='F')
        prediction_model = np.argmax(hist_model)

        # Get a random prediction
        prediction_rand = np.random.choice(np.where(board == 0)[0])
        probabilities_rand = np.zeros(36)
        probabilities_rand[np.where(board == 0)[0]] = 1/len(np.where(board == 0)[0])
        loss_rand = loss(torch.from_numpy(probabilities_rand.reshape(1, 36)), torch.from_numpy(np.array([target])))
        LL_rand[ind] = loss_rand.item()

        # For the current move number, get the prediction and compare with ground truth
        move_num = np.sum(np.absolute(board))
        moves_nn[move_num] += np.equal(prediction, target)
        moves_model[move_num] += np.equal(prediction_model, target)
        moves_rand[move_num] += np.equal(prediction_rand, target)
        totals[move_num] += 1

# Divide to compute the accuracy, remove the nans and  return
move_accuracy_nn = moves_nn/totals
move_accuracy_nn = move_accuracy_nn[~np.isnan(move_accuracy_nn)]

move_accuracy_model = moves_model/totals
move_accuracy_model = move_accuracy_model[~np.isnan(move_accuracy_model)]

move_accuracy_rand = moves_rand/totals
move_accuracy_rand = move_accuracy_rand[~np.isnan(move_accuracy_rand)]

#%% PLOT

# Plot the log-likelihoods as a histogram
logs_diff = LL_nn - test_ll

fig, ax = plt.subplots()
ax.hist(logs_diff,edgecolor='white', color='lightslategray',bins=np.arange(-4,4.25,.5))
ax.vlines(0,0,1800000,linewidth=2,linestyle='--',color='black')
ax.set_xlabel('Difference in log-likelihood per move')
ax.set_ylabel('Number of moves')
ax.set_xlim(-4,4)
ax.set_ylim(0,1800000)
ax.set_yticks([0,500000,1000000,1500000])
ax.set_yticklabels(['0','500k','1M','1.5M'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.75, 0.9,'Evidence favoring \n planning model',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
ax.text(0.25, 0.9,'Evidence favoring \n neural network',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
# plt.savefig('comparison.png', format='png', dpi=1000, bbox_inches='tight')

# Plot the accuracy per move
fig, ax = plt.subplots(figsize=(4,4))
ax.plot(np.arange(1,37,2), move_accuracy_nn, lw=2, color='darkblue', marker='o',label='neural network')
ax.plot(np.arange(1,37,2), move_accuracy_model, lw=2, color='darkorange', marker='o',label='planning model')
ax.fill_between(np.arange(1,37,2), move_accuracy_nn, move_accuracy_model, color='lightslategray',alpha=0.2)
ax.legend(frameon=False)
ax.set_xlim(0,37)
ax.set_xlabel('Move number')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.05)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('move_comparison.png', format='png', dpi=1000, bbox_inches='tight')

#%% FIND DIFFERENCES BETWEEN THE NETWORK AND MODEL

# Return a subset of the data where the network is correct and the model is incorrect
disagree_inds = []
disagree_KL = []

# For each board position
for ind in range(num_moves):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Get the cognitive model predictions
        hist_model = np.flipud(moves[ind,:].reshape(4, 9, order='F')).flatten(order='F')
        prediction_model = np.argmax(hist_model)

        # If the model and network disagree AND the network is correct
        if prediction_model != prediction and prediction == target:
            # Grab the index
            disagree_inds.append(ind)

#%% VISUALIZING NETWORK OUTPUT

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

#%% MODEL-NN DISAGREEMENT BOARDS

# Look at k boards
k = 50

# Visualize the top or bottom boards with their outputs
for i in range(k):
        curr_ind = disagree_inds[i]
        board = boards[curr_ind, :]
        output = outputs[curr_ind, :]
        target = targets[curr_ind, :]
        hist_model = np.flipud(moves[curr_ind,:].reshape(4, 9, order='F')).flatten(order='F')
        model = np.argmax(hist_model)
        # visualize_pred(board, output, target, model)
        visualize_pred(board, output, target, model, True, True, True, True, str(i)+'.png')

#%% FIND LARGEST DIFFERENCES BETWEEN THE NETWORK AND MODEL

# Look at all positions in the dataset and compute the KL divergence
kl = []

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
        hist_model = moves[ind,:]
        prediction_model = np.argmax(hist_model)

        # Normalize the nn output and compute the kl divergence
        output_norm = torch.from_numpy(output) - torch.logsumexp(torch.from_numpy(output), 0)
        kl_divergence = kl_loss(output_norm, torch.from_numpy(hist_model/np.sum(hist_model)))
        kl.append(kl_divergence.item())

#%% MODEL-NN DIFFERENCE BOARDS

# Sort boards by kl divergence
inds_sort = np.argsort(kl)
kl_sort = np.sort(kl)

# Look at k boards
k = 50

# Visualize the top or bottom boards with their outputs
for i in range(k):
        curr_ind = inds_sort[len(inds_sort)-1-i]
        board = boards[curr_ind, :]
        output = outputs[curr_ind, :]
        target = targets[curr_ind, :]
        # hist_model = np.flipud(moves[curr_ind,:].reshape(4, 9, order='F')).flatten(order='F')
        hist_model = moves[curr_ind,:]
        model = np.argmax(hist_model)
        # visualize_pred(board, output, target, model)
        visualize_pred(board, output, target, model, True, True, True, True, str(i)+'.png')

#%% MODEL-DATA DIFFERENCE BOARDS

# Sort boards by log likelihood of the planning model
inds_ll = np.argsort(test_ll)
model_ll_sort = np.sort(test_ll)

# Look at k boards
k = 10

# Visualize the top or bottom boards with their outputs
for i in range(k):
        curr_ind = inds_ll[len(inds_ll)-1-i]
        board = boards[curr_ind, :]
        output = outputs[curr_ind, :]
        target = targets[curr_ind, :]
        hist_model = np.flipud(moves[curr_ind,:].reshape(4, 9, order='F')).flatten(order='F')
        model = np.argmax(hist_model)
        visualize_pred(board, output, target, model)
        # visualize_pred(board, output, target, model, False, True, True, True, str(i)+'.png')