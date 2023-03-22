"""
Main file to test and analyze networks
"""

#%%
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np
import torch
import torch.nn as nn
import tqdm
from scipy import stats
import json

plt.rcParams.update({'font.size': 16})

#%% Network comparison

layers = [5, 10, 20, 40, 80]
test_loss200 = [1.978, 1.946, 1.922, 1.909, 1.900]
test_loss500 = [1.948, 1.925, 1.906, 1.894, 1.887]
test_loss1000 = [1.933, 1.907, 1.893, 1.883, 1.878]
test_loss2000 = [1.916, 1.894, 1.882, 1.875, 1.872]
test_loss4000 = [1.904, 1.882, 1.873, 1.868, 1.866]

fig, ax = plt.subplots()
ax.plot(layers, test_loss200, lw=2, color='black', marker='o')
ax.plot(layers, test_loss500, lw=2, color='darkblue', marker='o')
ax.plot(layers, test_loss1000, lw=2, color='blue', marker='o')
ax.plot(layers, test_loss2000, lw=2, color='cornflowerblue', marker='o')
ax.plot(layers, test_loss4000, lw=2, color='lightblue', marker='o')
ax.set_xlim(0,85)
ax.set_yticks([1.88,1.9,1.92,1.94,1.96,1.98])
ax.set_xticks([0,20,40,60,80])
ax.set_xlabel('Number of hidden layers')
ax.set_ylabel('Negative log-likelihood')
ax.legend(['200 units', '500 units', '1000 units', '2000 units', '4000 units'],frameon=False,loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.show()
plt.savefig('test_comparison.png', format='png', dpi=1000, bbox_inches='tight')

#%% Learning curves

# Load and plot the learning curves
losses = torch.load('/Volumes/Samsung_T5/Peak/networks/24/losses_9')
train_loss_new = losses['train_loss']
val_loss_new = losses['val_loss']

# Load the older losses if necessary and append
losses = torch.load('/Volumes/Samsung_T5/Peak/networks/24/losses_7')
train_loss = losses['train_loss']
val_loss = losses['val_loss']

train_loss.append(train_loss_new[0])
train_loss.append(train_loss_new[1])
val_loss.append(val_loss_new[0])
val_loss.append(val_loss_new[1])

# Plot the learning curves for train and validation
def plot_learning(train_loss, val_loss, lb=1.85, ub=1.96):
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(np.arange(1,len(train_loss)+1), train_loss, lw=2, color='darkblue', marker='o')
        ax.plot(np.arange(1,len(train_loss)+1), val_loss, lw=2, color='cornflowerblue', marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Negative log-likelihood')
        ax.set_ylim(lb, ub)
        # Add vertical lines for when lr was changed
        # for epoch in lr:
        #     plt.axvline(x=epoch, color='firebrick', linestyle='--')
        ax.legend(['train', 'validation'], frameon=False)
        ax.set_xlim(0, 11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()
        # plt.savefig('learning_24.png', format='png', dpi=1000, bbox_inches='tight')

plot_learning(train_loss, val_loss)

#%% Loading in the results from a network on the test set

# Load in the text file with the board positions and results
results_path = '/Volumes/Samsung_T5/Peak/networks/23/results_file.txt'
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

#%% Sanity check analysis

# Initialize a numpy array with number of guesses
guesses = np.zeros(36)

# Initialize a numpy array for each move (number correct, total number)
moves = np.zeros(36)
totals = np.zeros(36)

# Initialize a list to hold the likelihoods of the human move
# 19: 80 layers, 2000 units, 23: 40 layers, 4000 units, 24: 80 layers, 4000 units
# human_probs19 = []
human_probs23 = []
# human_probs24 = []

# Define the function to compute losses
loss = nn.CrossEntropyLoss()

# For each board position
for ind in tqdm.tqdm(range(num_moves)):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Grab the log probability of the human move to compare between networks
        # logprob = output[target]
        # human_probs24.append(logprob)
        loss_size = loss(torch.from_numpy(output.reshape(1, 36)), torch.from_numpy(np.array([target])))
        # human_probs19.append(loss_size.item())
        human_probs23.append(loss_size.item())
        # human_probs24.append(loss_size.item())

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

#%% Sanity check plots

# Plot accuracy as a function of number of guesses
fig, ax = plt.subplots(figsize=(4,4))
ax.plot(np.arange(1,37), guesses_accuracy, lw=2, color='darkblue', marker='o')
ax.set_xlim(0,37)
ax.set_ylim(0,1.05)
ax.set_xlabel('Number of guesses')
ax.set_ylabel('Accuracy')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('guesses.png', format='png', dpi=1000, bbox_inches='tight')

# Plot accuracy as a function of move number
fig, ax = plt.subplots(figsize=(4,4))
ax.plot(np.arange(1,37,2), move_accuracy, lw=2, color='darkblue', marker='o')
ax.set_xlim(0,37)
ax.set_xlabel('Move number')
ax.set_ylabel('Accuracy')
ax.set_ylim(0,1.05)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('move.png', format='png', dpi=1000, bbox_inches='tight')

# Plot correlation between the predictions of two networks
stats.spearmanr(human_probs24, human_probs19)

fig, ax = plt.subplots()
ax.scatter(human_probs24, human_probs19, color='darkblue', alpha=.5, s=1)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', lw=2, ls='--')
ax.set_xlabel('Negative log-likelihood\n(80 layers, 4000 units)')
ax.set_ylabel('Negative log-likelihood\n(80 layers, 2000 units)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-.5,13)
ax.set_ylim(-.5,13)
ax.set_xticks([0,5,10])
ax.set_yticks([0,5,10])
# ax.text(0.2, 0.95,'$\\rho$ = 0.99, p<$2*10^{-308}$',
#      horizontalalignment='center',
#      verticalalignment='center',
#      transform = ax.transAxes)
# plt.show()
plt.savefig('corr_unit.png', format='png', dpi=1000, bbox_inches='tight')

#%% Expertise and elo analysis

# Load the lists with IDs, final Elo, and number of games played
final20_games = np.load('/Users/ionatankuperwajs/Desktop/MaLab/Peak/Code/peak-analysis/final20_games.npy')
final20_ID = np.load('/Users/ionatankuperwajs/Desktop/MaLab/Peak/Code/peak-analysis/final20_ID.npy')
final20_ratings = np.load('/Users/ionatankuperwajs/Desktop/MaLab/Peak/Code/peak-analysis/final20_ratings.npy')

# Load in the dictionary with number of games per user
with open('/Users/ionatankuperwajs/Desktop/MaLab/Peak/Code/peak-analysis/saved_analyses_updated/numgames_user.txt', 'r') as filehandle:
    num_games = json.load(filehandle)

# Set paths
meta_path = '/Volumes/Samsung_T5/Peak/nn_data/test_meta/%s/test_meta_%d.pt'

# Create lists to hold the log probabilities and elos per move
probs = []
elos = []
games = []

# Create elo lookup dictionary
elo_lookup = dict(zip(final20_ID, final20_ratings))

# For each line in the text file
counter = 0
game_count = 1
move_count = 0

# Define the function to compute losses
loss = nn.CrossEntropyLoss()

# For each board position
for ind in tqdm.tqdm(range(num_moves)):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Grab the log probability of the human move, convert to nll
        # logprob = output[target]
        loss_size = loss(torch.from_numpy(output.reshape(1, 36)), torch.from_numpy(np.array([target])))
        logprob = loss_size.item()

        # Grab the human ID
        meta_string = '%03d' % np.floor(game_count / 10000)
        curr_meta_path = meta_path % (meta_string, game_count)
        meta_data = torch.load(curr_meta_path)
        curr_user = meta_data[0][move_count]

        # Grab the elo if the user has played 20+ games
        if curr_user in elo_lookup:
                curr_elo = elo_lookup[curr_user]

                # Grab the total experience
                curr_games = num_games[curr_user]

                # Add to the final arrays
                probs.append(logprob)
                elos.append(curr_elo)
                games.append(curr_games)

        # Increment the move or game count
        if move_count == len(meta_data[0])-1:
                move_count = 0
                game_count += 1
        else:
                move_count += 1
        counter += 1

 #%% ELO AND EXPERTISE PLOTS

# Sort the elos and probabilities similarly
sort_idxs = np.asarray(elos).argsort()
elos_sorted = np.asarray(elos)[sort_idxs]
probs_sorted = np.asarray(probs)[sort_idxs]

# Now split into 10 bins and compute the mean and sem
elos_binned = np.array_split(elos_sorted, 5)
probs_binned = np.array_split(probs_sorted, 5)

elos_mean = []
probs_mean = []
probs_sem = []
for i in range(5):
        elos_mean.append(np.mean(elos_binned[i]))
        probs_mean.append(np.mean(probs_binned[i]))
        probs_sem.append(stats.sem(probs_binned[i]))

# Find the correlation
stats.spearmanr(elos_sorted, probs_sorted)

# Then plot the bins
fig, ax = plt.subplots(figsize=(4,4))
ax.errorbar(elos_mean, probs_mean, yerr=probs_sem, lw=2, color='darkblue', marker='o', capsize=3)
ax.set_xlabel('Elo rating')
ax.set_ylabel('Negative log-likelihood')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-225,225)
ax.set_ylim(1.79,1.87)
# ax.text(0.2, 0.95,'$\\rho$ = 0.09, p<$2*10^{-308}$',
#      horizontalalignment='center',
#      verticalalignment='center',
#      transform = ax.transAxes)
plt.show()
# plt.savefig('predictability_Elo.png', format='png', dpi=1000, bbox_inches='tight')

# Sort the games and probabilities similarly
sort_idxs = np.asarray(games).argsort()
games_sorted = np.asarray(games)[sort_idxs]
probs_sorted = np.asarray(probs)[sort_idxs]

# Now split into 10 bins and compute the mean and sem
games_binned = np.array_split(games_sorted, 5)
probs_binned = np.array_split(probs_sorted, 5)

games_mean = []
probs_mean = []
probs_sem = []
for i in range(5):
        games_mean.append(np.mean(games_binned[i]))
        probs_mean.append(np.mean(probs_binned[i]))
        probs_sem.append(stats.sem(probs_binned[i]))

# Find the correlation
stats.spearmanr(games_sorted, probs_sorted)

# Then plot the bins
fig, ax = plt.subplots(figsize=(4,4))
ax.errorbar(games_mean, probs_mean, yerr=probs_sem, lw=2, color='darkblue', marker='o', capsize=3)
ax.set_xlabel('Number of games played')
ax.set_ylabel('Negative log-likelihood')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-50,800)
ax.set_xticks([0, 250, 500, 750])
ax.set_ylim(1.79,1.87)
# ax.set_yticks([2.5, 2.55, 2.6, 2.65, 2.7])
# ax.text(0.2, 0.95,'$\\rho$ = -0.02, p<$2*10^{-308}$',
#      horizontalalignment='center',
#      verticalalignment='center',
#      transform = ax.transAxes)
# plt.show()
plt.savefig('predictability_games.png', format='png', dpi=1000, bbox_inches='tight')


#%% Visualizing network output

# Function that takes a board and model output and visualizes it
def visualize_pred(board, output, target, outputOn=True, targetOn=True, save=False, filename=''):
        # Flip to correct dimensions
        board = board.reshape(9,4).T.flatten()
        output = output.reshape(9,4).T.flatten()
        target = np.ravel_multi_index(np.unravel_index(target.astype(int), (9,4)), (9,4), order='F')

        # Preprocessing
        black_pieces = [index for index, element in enumerate(board) if element == 1]
        white_pieces = [index for index, element in enumerate(board) if element == -1]
        model_moves = np.exp(np.asarray(output))
        model_moves /= np.sum(model_moves)

        # Create the figure and colormap
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.vlines(np.arange(-0.5, 9.5, 1), -0.5, 3.5, color='black')
        ax.hlines(np.arange(-0.5, 4.5, 1), -0.5, 8.5, color='black')
        cm = colors.LinearSegmentedColormap.from_list('gray_red_map', [colors.to_rgb('darkgray'),
                                                                colors.to_rgb('red')], N=100)

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

        # Plot with the likelihood mapping
        if outputOn is True:
                plt.imshow(np.flip(np.reshape(model_moves, [4, 9]),axis=0), cmap=cm, interpolation='nearest', origin='lower', vmin=0, vmax=0.5)
        else:
                plt.imshow(np.flip(np.reshape(model_moves, [4, 9]),axis=0), cmap=cm, interpolation='nearest', origin='lower', vmin=0, vmax=100)

        ax.axis('off')
        fig.tight_layout()
        if save is True:
                plt.savefig(filename, format='png', dpi=1000, bbox_inches='tight')
        else:
                plt.show()

# visualize_pred(board, output, target)

#%% Retrieving data for specific board positions

# ** THE INPUT DATA TO THESE FUNCTIONS MUST BE A LIST OF BOARDS RATHER THAN A NUMPY ARRAY **

# Function to return a subset of the data where the board position matches exactly
def board_subset(boards, match_board):

        # Set up a list to return the indices
        data = []

        # Loop through the results looking for matches
        for ind in tqdm.tqdm(range(len(boards))):
                board = boards[ind]
                if np.array_equal(board, match_board):
                        data.append(ind)

        return data

# Function to return a subset of the data where the given pattern is included
def board_subset_pattern(boards, pattern):

        # Set up a list to return the indices
        data = []

        # Grab the indices of the black/white pieces of the pattern
        pattern_black = np.nonzero(np.asarray(pattern) == 1)[0]
        pattern_white = np.where(np.asarray(pattern) == -1)[0]

        # Loop through the results looking for matches
        for ind in tqdm.tqdm(range(len(boards))):
                board = boards[ind]
                board_black = np.where(np.asarray(board) == 1)[0]
                board_white = np.where(np.asarray(board) == -1)[0]
                black_bool = True
                white_bool = True
                if pattern_black.size != 0:
                        black_bool = np.isin(pattern_black, board_black).all()
                if pattern_white.size != 0:
                        white_bool = np.isin(pattern_white, board_white).all()
                if black_bool and white_bool:
                        data.append(ind)

        return data

# Function to return a subset of the data where the number of pieces on the board is fixed
def board_subset_len(boards, num_pieces):

        # Set up a list to return the indices
        data = []

        # Loop through the results looking for matches
        for ind in tqdm.tqdm(range(len(boards))):
                board = boards[ind]
                if np.count_nonzero(board) == num_pieces:
                        data.append(ind)

        return data

# Function to return a ranking of board occurrences in a data set
def board_subset_rank(boards):

        # Set up a dict to return the indices and a count
        data = {}
        first_ind = {}

        # Loop through the results looking for matches
        for ind in tqdm.tqdm(range(len(boards))):

                board = boards[ind]
                pattern_black = np.nonzero(np.asarray(board) == 1)[0].tolist()
                pattern_white = np.where(np.asarray(board) == -1)[0].tolist()
                key_inds = tuple(pattern_black+pattern_white)

                # Check if the target exists in the dict or not
                if key_inds in data.keys():
                        data[key_inds] += 1
                else:
                        data[key_inds] = 1
                        first_ind[key_inds] = ind

        data = {k: v for k, v in sorted(data.items(), key=lambda x: x[1], reverse=True)}

        return data, first_ind

# Function to return a ranking of target occurrences in a data set
def board_target_rank(targets):

        # Set up a dict to return the indices and a count
        data = {}
        first_ind = {}

        # Loop through the results looking for matches
        for ind in tqdm.tqdm(range(len(targets))):

                target = int(targets[ind])

                # Check if the target exists in the dict or not
                if target in data.keys():
                        data[target] += 1
                else:
                        data[target] = 1
                        first_ind[target] = ind

        data = {k: v for k, v in sorted(data.items(), key=lambda x: x[1], reverse=True)}

        return data, first_ind

# Pick a board position and grab the subset
# match_board = board
# board_inds = board_subset(boards, match_board)


 #%% Elo and expertise boards

# Sort the elos and boards similarly
sort_idxs = np.asarray(elos).argsort()
elos_sorted = np.asarray(elos)[sort_idxs]
boards_sorted = boards[sort_idxs, :]
outputs_sorted = outputs[sort_idxs, :]
targets_sorted = targets[sort_idxs, :]

k = 20
for i in range(k):
        board = boards_sorted[i, :]
        output = outputs_sorted[i, :]
        target = targets_sorted[i, :]
        # visualize_pred(board, output, target)
        visualize_pred(board, output, target, True, True, True, str(i)+'.png')

#%% Accuracy boards

# Sort boards by log likelihood of the neural network
inds_ll = np.argsort(human_probs24)
nn_ll_sort = np.sort(human_probs24)

# Look at k boards
k = 20

# Visualize the top or bottom boards with their outputs
for i in range(k):
        curr_ind = inds_ll[i]
        board = boards[curr_ind, :]
        output = outputs[curr_ind, :]
        target = targets[curr_ind, :]
        # visualize_pred(board, output, target)
        visualize_pred(board, output, target, True, True, True, str(i)+'.png')

#%% Entropy analysis

# Compute the entropy for all positions
entropy = np.zeros(num_moves)

# Compute the entropy per move number
entropy_move = np.zeros(36)
entropy_count = np.zeros(36)

# For each board position
for ind in tqdm.tqdm(range(num_moves)):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind,:]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Normalize the output and compute the entropy
        norm_output = np.exp(output)
        norm_output /= np.sum(norm_output)
        curr_entropy = stats.entropy(norm_output)
        entropy[ind] = curr_entropy

        # Also save the entropy based on move number
        num_pieces = np.count_nonzero(board)
        entropy_move[num_pieces] += curr_entropy
        entropy_count[num_pieces] += 1

# Look at the top and bottom k
k = 20
top_inds = np.argpartition(entropy, -k)[-k:]
top_inds_sort = top_inds[np.argsort(entropy[top_inds])]
bottom_inds = np.argpartition(entropy, k)[:k]
bottom_inds_sort = bottom_inds[np.argsort(entropy[bottom_inds])]

# Visualize the top or bottom boards with their outputs
for i in range(k):
        curr_ind = top_inds_sort[i]
        board = boards[curr_ind, :]
        output = outputs[curr_ind, :]
        target = targets[curr_ind, :]
        visualize_pred(board, output, target)
        # visualize_pred(board, output, target, True, True, True, str(i)+'.png')

# Divide to compute the average entropy, remove the nans and  return
move_entropy = entropy_move/entropy_count
move_entropy = move_entropy[~np.isnan(move_entropy)]

# Plot entropy as a function of move number
fig, ax = plt.subplots(figsize=(4,4))
ax.plot(np.arange(1,37,2), move_entropy, lw=2, color='darkblue', marker='o')
ax.set_xlim(0,37)
ax.set_xlabel('Move number')
ax.set_ylabel('Entropy')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.show()
plt.savefig('entropy.png', format='png', dpi=1000, bbox_inches='tight')