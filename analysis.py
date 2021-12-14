"""
Main file to test and analyze networks
"""

#%%
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np
import math
import torch
import re
import tqdm
from scipy import stats
import json

#%% NETWORK COMPARISON

layers = [5, 10, 20, 40, 80]
val_loss200 = [2.088, 2.056, 2.037, 2.023, 2.014]
val_loss500 = [2.055, 2.035, 2.020, 2.007, 1.999]
val_loss1000 = [2.039, 2.020, 2.006, 1.996, 1.989]
val_loss2000 = [2.024, 2.006, 1.993, 1.984, 1.981]
val_loss4000 = [2.013, 1.994, 1.982, 1.977, 1.974]

# 'mistyrose','darksalmon','red','firebrick','maroon'
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(layers, val_loss200, lw=2, color='black', marker='o')
ax.plot(layers, val_loss500, lw=2, color='maroon', marker='o')
ax.plot(layers, val_loss1000, lw=2, color='firebrick', marker='o')
ax.plot(layers, val_loss2000, lw=2, color='red', marker='o')
ax.plot(layers, val_loss4000, lw=2, color='darksalmon', marker='o')
ax.set_xlim(0,85)
ax.set_xlabel('Number of hidden layers')
ax.set_ylabel('Negative log-likelihood')
ax.legend(['200 units', '500 units', '1000 units', '2000 units', '4000 units'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.show()
plt.savefig('test_comparison.png', format='png', dpi=1000, bbox_inches='tight')

#%% LEARNING CURVES

# Load and plot the learning curves
losses = torch.load('/Volumes/Samsung_T5/Peak/networks/21/losses_9')
train_loss_new = losses['train_loss']
val_loss_new = losses['val_loss']

# Load the older losses if necessary and append
losses = torch.load('/Volumes/Samsung_T5/Peak/networks/21/losses_7')
train_loss = losses['train_loss']
val_loss = losses['val_loss']

train_loss.append(train_loss_new[0])
train_loss.append(train_loss_new[1])
val_loss.append(val_loss_new[0])
val_loss.append(val_loss_new[1])

# Plot the learning curves for train and validation
def plot_learning(train_loss, val_loss, lb=1.9, ub=2.4):
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
        ax.set_xlim(0, 11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()
        # plt.savefig('learning_21.png', format='png', dpi=1000, bbox_inches='tight')

plot_learning(train_loss, val_loss)

#%% LOADING IN THE RESULTS FROM A NETWORK ON THE TEST SET

# Load in the text file with the board positions and results
results_path = '/Volumes/Samsung_T5/Peak/networks/21/results_file.txt'
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

#%% SANITY CHECK ANALYSIS

# Initialize a numpy array with number of guesses
guesses = np.zeros(36)

# Initialize a numpy array for each move (number correct, total number)
moves = np.zeros(36)
totals = np.zeros(36)

# Initialize a list to hold the log probabilities of the human move
human_probs21 = []
# human_probs20 = []
# human_probs25 = []

# For each board position
for ind in range(num_moves):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Grab the log probability of the human move to compare between networks
        logprob = output[target]
        human_probs21.append(logprob)

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

#%% SANITY CHECK PLOTS

# Plot accuracy as a function of number of guesses
fig, ax = plt.subplots(figsize=(6,4))
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
fig, ax = plt.subplots(figsize=(6,4))
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
stats.spearmanr(human_probs21, human_probs20)

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(human_probs21, human_probs20, color='darkblue', alpha=.5, s=1)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', lw=2, ls='--')
ax.set_xlabel('Log-likelihood of human move \n (80 layers, 4000 units)')
ax.set_ylabel('Log-likelihood of human move \n (40 layers, 4000 units)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.2, 0.95,'$\\rho$ = 0.99, p<$2*10^{-308}$',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
# plt.savefig('layer_corr.png', format='png', dpi=1000, bbox_inches='tight')

#%% EXPERTISE AND ELO ANALYSIS

# Set the path where the text file with elo ratings is stored
elo_path = '/Users/ionatankuperwajs/Desktop/MaLab/Peak/Code/bayeselo/game_outcomes_peak_paper/ratings.txt'

# Set two regexp patterns to recognize user IDs and ratings in the text file
ID_pattern = re.compile(r"\w{10}")
rating_pattern = re.compile(r"\w{10}\s{3}\d{4}")

# Go through the text file and add all of the matching expressions to two lists
with open (elo_path, 'rt') as myfile:
    contents = myfile.read()
userID = ID_pattern.findall(contents)
user_ratings = rating_pattern.findall(contents)

# For ratings, remove the whitespace and convert from strings to integers
ratings_plot = [int(w[-4:]) for w in user_ratings]

# Remove the average overall AI elo rating from each rating (integer offsets don't matter)
mean_ai_elo = 3104
ratings_plot_avg = [x - mean_ai_elo for x in ratings_plot]

# Load in the dictionary with number of games per user
with open('/Users/ionatankuperwajs/Desktop/MaLab/Peak/Code/peak-analysis/saved_analyses_updated/numgames_user.txt', 'r') as filehandle:
    num_games = json.load(filehandle)

# Set paths
meta_path = '/Volumes/Samsung_T5/Peak/nn_data/test_meta/%s/test_meta_%d.pt'

# Create numpy arrays to hold the log probabilities and elos per move
probs = np.zeros(num_moves)
elos = np.zeros(num_moves)
games = np.zeros(num_moves)

# Create elo lookup dictionary
elo_lookup = dict(zip(userID, ratings_plot_avg))

# For each line in the text file
counter = 0
game_count = 1
move_count = 0
# For each board position
for ind in range(num_moves):
        # Define the components we need
        board = boards[:, ind].astype(np.int)
        output = outputs[:, ind]
        prediction = int(predictions[:, ind][0])
        target = int(targets[:, ind][0])

        # Grab the log probability of the human move
        logprob = output[target]

        # Grab the human ID
        meta_string = '%03d' % np.floor(game_count / 10000)
        curr_meta_path = meta_path % (meta_string, game_count)
        meta_data = torch.load(curr_meta_path)
        curr_user = meta_data[0][move_count]

        # Grab the elo
        curr_elo = elo_lookup[curr_user]

        # Grab the total experience
        curr_games = num_games[curr_user]

        # Add to the final arrays
        probs[counter] = logprob
        elos[counter] = curr_elo
        games[counter] = curr_games

        # Increment the move or game count
        if move_count == len(meta_data[0])-1:
                move_count = 0
                game_count += 1
        else:
                move_count += 1
        counter += 1

 #%% ELO AND EXPERTISE PLOTS

# Sort the elos and probabilities similarly
sort_idxs = elos.argsort()
elos_sorted = elos[sort_idxs]
probs_sorted = probs[sort_idxs]

# Now split into 10 bins and compute the mean and sem
elos_binned = np.array_split(elos_sorted, 10)
probs_binned = np.array_split(probs_sorted, 10)

elos_mean = []
probs_mean = []
probs_sem = []
for i in range(10):
        elos_mean.append(np.mean(elos_binned[i]))
        probs_mean.append(np.mean(probs_binned[i]))
        probs_sem.append(stats.sem(probs_binned[i]))

# Find the correlation
stats.spearmanr(elos_sorted, probs_sorted)

# Then plot the bins
fig, ax = plt.subplots(figsize=(6,4))
ax.errorbar(elos_mean, probs_mean, yerr=probs_sem, lw=2, color='darkblue', marker='o', capsize=3)
ax.set_xlabel('Elo rating')
ax.set_ylabel('Log-likelihood of human move')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-275,275)
ax.text(0.2, 0.95,'$\\rho$ = 0.09, p<$2*10^{-308}$',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
# plt.savefig('predictability_Elo.png', format='png', dpi=1000, bbox_inches='tight')

# Sort the games and probabilities similarly
sort_idxs = games.argsort()
games_sorted = games[sort_idxs]
probs_sorted = probs[sort_idxs]

# Now split into 10 bins and compute the mean and sem
games_binned = np.array_split(games_sorted, 10)
probs_binned = np.array_split(probs_sorted, 10)

games_mean = []
probs_mean = []
probs_sem = []
for i in range(10):
        games_mean.append(np.mean(games_binned[i]))
        probs_mean.append(np.mean(probs_binned[i]))
        probs_sem.append(stats.sem(probs_binned[i]))

# Find the correlation
stats.spearmanr(games_sorted, probs_sorted)

# Then plot the bins
fig, ax = plt.subplots(figsize=(6,4))
ax.errorbar(games_mean, probs_mean, yerr=probs_sem, lw=2, color='darkblue', marker='o', capsize=3)
ax.set_xlabel('Number of games played')
ax.set_ylabel('Log-likelihood of human move')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-50,900)
ax.set_ylim(2.48,2.7)
ax.set_yticks([2.5, 2.55, 2.6, 2.65, 2.7])
ax.text(0.2, 0.95,'$\\rho$ = -0.02, p<$2*10^{-308}$',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.show()
# plt.savefig('predictability_games.png', format='png', dpi=1000, bbox_inches='tight')

#%%
# Sort the games and probabilities similarly
sort_idxs = games.argsort()
games_sorted = games[sort_idxs]
probs_sorted = probs[sort_idxs]
elos_sorted = elos[sort_idxs]

# Now split into 10 bins and compute the mean and sem
games_binned = np.array_split(games_sorted, 10)
probs_binned = np.array_split(probs_sorted, 10)
elos_binned = np.array_split(elos_sorted, 10)

games_mean = []
elos_mean = []
elos_sem = []
probs_mean = []
probs_sem = []
for i in range(10):
        games_mean.append(np.mean(games_binned[i]))
        probs_mean.append(np.mean(probs_binned[i]))
        probs_sem.append(stats.sem(probs_binned[i]))
        elos_mean.append(np.mean(elos_binned[i]))
        elos_sem.append(stats.sem(elos_binned[i]))

fig, ax = plt.subplots(figsize=(6,4))
ax.errorbar(games_mean, elos_mean, yerr=elos_sem, lw=2, color='firebrick', marker='o', capsize=3)
ax.set_xlabel('Number of games played')
ax.set_ylabel('Elo rating')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.set_xlim(-50,900)
# ax.set_ylim(2.48,2.7)
# ax.set_yticks([2.5, 2.55, 2.6, 2.65, 2.7])
# ax.text(0.2, 0.95,'$\\rho$ = -0.02, p<$2*10^{-308}$',
#      horizontalalignment='center',
#      verticalalignment='center',
#      transform = ax.transAxes)
plt.show()


#%% VISUALIZING NETWORK OUTPUT

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

visualize_pred(board, output, target)

#%% RETRIEVING DATA FOR SPECIFIC BOARD POSITIONS

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

 #%% ELO AND EXPERTISE BOARDS

# Sort the elos and boards similarly
sort_idxs = elos.argsort()
elos_sorted = elos[sort_idxs]
boards_sorted = boards[:, sort_idxs]
outputs_sorted = outputs[:, sort_idxs]
targets_sorted = targets[:, sort_idxs]

k = 50
for i in range(k):
        board = boards_sorted[:, -(i+1)]
        output = outputs_sorted[:, -(i+1)]
        target = targets_sorted[:, -(i+1)]
        # visualize_pred(board, output, target)
        visualize_pred(board, output, target, True, str(i)+'.png')

#%% ENTROPY

# Compute the entropy for all positions
entropy = np.zeros(num_moves)

# Compute the entropy per move number
entropy_move = np.zeros(36)
entropy_count = np.zeros(36)

# For each board position
for ind in range(num_moves):
        # Define the components we need
        board = boards[:, ind].astype(np.int)
        output = outputs[:, ind]
        prediction = int(predictions[:, ind][0])
        target = int(targets[:, ind][0])

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
k = 50
top_inds = np.argpartition(entropy, -k)[-k:]
top_inds_sort = top_inds[np.argsort(entropy[top_inds])]
bottom_inds = np.argpartition(entropy, k)[:k]
bottom_inds_sort = bottom_inds[np.argsort(entropy[bottom_inds])]

# Visualize the top or bottom boards with their outputs
for i in range(k):
        curr_ind = top_inds_sort[i]
        board = boards[:, curr_ind]
        output = outputs[:, curr_ind]
        target = targets[:, curr_ind]
        # visualize_pred(board, output, target)
        visualize_pred(board, output, target, True, str(i)+'.png')

# Divide to compute the average entropy, remove the nans and  return
move_entropy = entropy_move/entropy_count
move_entropy = move_entropy[~np.isnan(move_entropy)]

# Plot entropy as a function of move number
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(np.arange(1,37,2), move_entropy, lw=2, color='darkblue', marker='o')
ax.set_xlim(0,37)
ax.set_xlabel('Move number')
ax.set_ylabel('Entropy')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.show()
plt.savefig('entropy.png', format='png', dpi=1000, bbox_inches='tight')

#%% SUMMARY STATS

# Function to convert a board index to a coordinate
def move_to_coordinate(move):
        row = math.floor(move/4)
        column = move-(row*4)
        return (row,column)

# Function that computes the average Manhattan distance of a move from a set of pieces
def distance_from_pieces(move, pieces):
        total_distance = 0
        distance_count = 0
        move = move_to_coordinate(move)
        if pieces.size == 0:
                return 0
        else:
                for piece in pieces:
                        piece = move_to_coordinate(piece)
                        total_distance += np.abs(move[0]-piece[0])+np.abs(move[1]-piece[1])
                        distance_count += 1
                avg_distance = total_distance/distance_count
                return avg_distance

# Function that computes the Manhattan distance from the center of mass of a set of pieces
def distance_from_mass(move, pieces):
        move = move_to_coordinate(move)
        if pieces.size == 0:
                return 0
        else:
                x_sum = 0
                y_sum = 0
                for piece in pieces:
                        piece = move_to_coordinate(piece)
                        x_sum += piece[0]
                        y_sum += piece[1]
                x_bar = x_sum/len(pieces)
                y_bar = y_sum/len(pieces)
                distance = np.abs(move[0]-x_bar)+np.abs(move[1]-y_bar)
                return distance

# Function that computes the number of neighboring pieces to a move
def number_neighbors(move, pieces):
        move = move_to_coordinate(move)
        if pieces.size == 0:
                return 0
        else:
                neighbor_count = 0
                for piece in pieces:
                        row = False
                        column = False
                        piece = move_to_coordinate(piece)
                        if np.abs(move[0]-piece[0]) == 1 or np.abs(move[0]-piece[0]) == 0:
                                row = True
                        if np.abs(move[1]-piece[1]) == 1 or np.abs(move[1]-piece[1]) == 0:
                                column = True
                        if move == piece:
                                row = False
                                column = False
                        if row and column:
                                neighbor_count += 1
                return neighbor_count

# Function that checks if a threat was made with a given move
def check_threat_made(move, pieces):
        threat_made = False
        board = np.copy(pieces)
        board[move] = 1
        coord = move_to_coordinate(move)
        row = coord[0]
        column = coord[1]

        # First check the horizontal orientation
        horizontal = [board[row*4], board[row*4+1], board[row*4+2], board[row*4+3]]
        if horizontal.count(0) == 1 and horizontal.count(1) == 3:
                threat_made = True

        # Then check the vertical orientation
        if row == 0:
                vertical1 = [board[move], board[move+4], board[move+8], board[move+12]]
                if vertical1.count(0) == 1 and vertical1.count(1) == 3:
                        threat_made = True
        elif row == 1:
                vertical1 = [board[move-4], board[move], board[move+4], board[move+8]]
                if vertical1.count(0) == 1 and vertical1.count(1) == 3:
                        threat_made = True
                vertical2 = [board[move], board[move+4], board[move+8], board[move+12]]
                if vertical2.count(0) == 1 and vertical2.count(1) == 3:
                        threat_made = True
        elif row == 2:
                vertical1 = [board[move-8], board[move-4], board[move], board[move+4]]
                if vertical1.count(0) == 1 and vertical1.count(1) == 3:
                        threat_made = True
                vertical2 = [board[move-4], board[move], board[move+4], board[move+8]]
                if vertical2.count(0) == 1 and vertical2.count(1) == 3:
                        threat_made = True
                vertical3 = [board[move], board[move+4], board[move+8], board[move+12]]
                if vertical3.count(0) == 1 and vertical3.count(1) == 3:
                        threat_made = True
        elif row == 3 or row == 4 or row == 5:
                vertical1 = [board[move-12], board[move-8], board[move-4], board[move]]
                if vertical1.count(0) == 1 and vertical1.count(1) == 3:
                        threat_made = True
                vertical2 = [board[move-8], board[move-4], board[move], board[move+4]]
                if vertical2.count(0) == 1 and vertical2.count(1) == 3:
                        threat_made = True
                vertical3 = [board[move-4], board[move], board[move+4], board[move+8]]
                if vertical3.count(0) == 1 and vertical3.count(1) == 3:
                        threat_made = True
                vertical4 = [board[move], board[move+4], board[move+8], board[move+12]]
                if vertical4.count(0) == 1 and vertical4.count(1) == 3:
                        threat_made = True
        elif row == 6:
                vertical1 = [board[move-12], board[move-8], board[move-4], board[move]]
                if vertical1.count(0) == 1 and vertical1.count(1) == 3:
                        threat_made = True
                vertical2 = [board[move-8], board[move-4], board[move], board[move+4]]
                if vertical2.count(0) == 1 and vertical2.count(1) == 3:
                        threat_made = True
                vertical3 = [board[move-4], board[move], board[move+4], board[move+8]]
                if vertical3.count(0) == 1 and vertical3.count(1) == 3:
                        threat_made = True
        elif row == 7:
                vertical1 = [board[move-12], board[move-8], board[move-4], board[move]]
                if vertical1.count(0) == 1 and vertical1.count(1) == 3:
                        threat_made = True
                vertical2 = [board[move-8], board[move-4], board[move], board[move+4]]
                if vertical2.count(0) == 1 and vertical2.count(1) == 3:
                        threat_made = True
        elif row == 8:
                vertical1 = [board[move-12], board[move-8], board[move-4], board[move]]
                if vertical1.count(0) == 1 and vertical1.count(1) == 3:
                        threat_made = True

        # Then check the diagonal orientation
        if column == 0:
                if row != 6 and row != 7 and row != 8:
                        diag1 = [board[move], board[move+5], board[move+10], board[move+15]]
                        if diag1.count(0) == 1 and diag1.count(1) == 3:
                                threat_made = True
                if row != 0 and row != 1 and row != 2:
                        diag2 = [board[move-9], board[move-6], board[move-3], board[move]]
                        if diag2.count(0) == 1 and diag2.count(1) == 3:
                                threat_made = True

        elif column == 1:
                if row != 0 and row != 7 and row != 8:
                        diag1 = [board[move-5], board[move], board[move+5], board[move+10]]
                        if diag1.count(0) == 1 and diag1.count(1) == 3:
                                threat_made = True
                if row != 0 and row != 1 and row != 8:
                        diag2 = [board[move-6], board[move-3], board[move], board[move+3]]
                        if diag2.count(0) == 1 and diag2.count(1) == 3:
                                threat_made = True

        elif column == 2:
                if row != 0 and row != 1 and row != 8:
                        diag1 = [board[move-10], board[move-5], board[move], board[move+5]]
                        if diag1.count(0) == 1 and diag1.count(1) == 3:
                                threat_made = True
                if row != 0 and row != 7 and row != 8:
                        diag2 = [board[move-3], board[move], board[move+3], board[move+6]]
                        if diag2.count(0) == 1 and diag2.count(1) == 3:
                                threat_made = True

        elif column == 3:
                if row != 0 and row != 1 and row != 2:
                        diag1 = [board[move-15], board[move-10], board[move-5], board[move]]
                        if diag1.count(0) == 1 and diag1.count(1) == 3:
                                threat_made = True
                if row != 6 and row != 7 and row != 8:
                        diag2 = [board[move], board[move+3], board[move+6], board[move+9]]
                        if diag2.count(0) == 1 and diag2.count(1) == 3:
                                threat_made = True

        return threat_made

# Function that checks if a threat was defended with a given move
def check_threat_defended(move, pieces):
        threat_defended = False
        board = np.copy(pieces)
        board[move] = 1
        coord = move_to_coordinate(move)
        row = coord[0]
        column = coord[1]

        # First check the horizontal orientation
        horizontal = [board[row*4], board[row*4+1], board[row*4+2], board[row*4+3]]
        if horizontal.count(1) == 1 and horizontal.count(-1) == 3:
                threat_defended = True

        # Then check the vertical orientation
        if row == 0:
                vertical1 = [board[move], board[move+4], board[move+8], board[move+12]]
                if vertical1.count(1) == 1 and vertical1.count(-1) == 3:
                        threat_defended = True
        elif row == 1:
                vertical1 = [board[move-4], board[move], board[move+4], board[move+8]]
                if vertical1.count(1) == 1 and vertical1.count(-1) == 3:
                        threat_defended = True
                vertical2 = [board[move], board[move+4], board[move+8], board[move+12]]
                if vertical2.count(1) == 1 and vertical2.count(-1) == 3:
                        threat_defended = True
        elif row == 2:
                vertical1 = [board[move-8], board[move-4], board[move], board[move+4]]
                if vertical1.count(1) == 1 and vertical1.count(-1) == 3:
                        threat_defended = True
                vertical2 = [board[move-4], board[move], board[move+4], board[move+8]]
                if vertical2.count(1) == 1 and vertical2.count(-1) == 3:
                        threat_defended = True
                vertical3 = [board[move], board[move+4], board[move+8], board[move+12]]
                if vertical3.count(1) == 1 and vertical3.count(-1) == 3:
                        threat_defended = True
        elif row == 3 or row == 4 or row == 5:
                vertical1 = [board[move-12], board[move-8], board[move-4], board[move]]
                if vertical1.count(1) == 1 and vertical1.count(-1) == 3:
                        threat_defended = True
                vertical2 = [board[move-8], board[move-4], board[move], board[move+4]]
                if vertical2.count(1) == 1 and vertical2.count(-1) == 3:
                        threat_defended = True
                vertical3 = [board[move-4], board[move], board[move+4], board[move+8]]
                if vertical3.count(1) == 1 and vertical3.count(-1) == 3:
                        threat_defended = True
                vertical4 = [board[move], board[move+4], board[move+8], board[move+12]]
                if vertical4.count(1) == 1 and vertical4.count(-1) == 3:
                        threat_defended = True
        elif row == 6:
                vertical1 = [board[move-12], board[move-8], board[move-4], board[move]]
                if vertical1.count(1) == 1 and vertical1.count(-1) == 3:
                        threat_defended = True
                vertical2 = [board[move-8], board[move-4], board[move], board[move+4]]
                if vertical2.count(1) == 1 and vertical2.count(-1) == 3:
                        threat_defended = True
                vertical3 = [board[move-4], board[move], board[move+4], board[move+8]]
                if vertical3.count(1) == 1 and vertical3.count(-1) == 3:
                        threat_defended = True
        elif row == 7:
                vertical1 = [board[move-12], board[move-8], board[move-4], board[move]]
                if vertical1.count(1) == 1 and vertical1.count(-1) == 3:
                        threat_defended = True
                vertical2 = [board[move-8], board[move-4], board[move], board[move+4]]
                if vertical2.count(1) == 1 and vertical2.count(-1) == 3:
                        threat_defended = True
        elif row == 8:
                vertical1 = [board[move-12], board[move-8], board[move-4], board[move]]
                if vertical1.count(1) == 1 and vertical1.count(-1) == 3:
                        threat_defended = True

        # Then check the diagonal orientation
        if column == 0:
                if row != 6 and row != 7 and row != 8:
                        diag1 = [board[move], board[move + 5], board[move + 10], board[move + 15]]
                        if diag1.count(1) == 1 and diag1.count(-1) == 3:
                                threat_defended = True
                if row != 0 and row != 1 and row != 2:
                        diag2 = [board[move - 9], board[move - 6], board[move - 3], board[move]]
                        if diag2.count(1) == 1 and diag2.count(-1) == 3:
                                threat_defended = True

        elif column == 1:
                if row != 0 and row != 7 and row != 8:
                        diag1 = [board[move - 5], board[move], board[move + 5], board[move + 10]]
                        if diag1.count(1) == 1 and diag1.count(-1) == 3:
                                threat_defended = True
                if row != 0 and row != 1 and row != 8:
                        diag2 = [board[move - 6], board[move - 3], board[move], board[move + 3]]
                        if diag2.count(1) == 1 and diag2.count(-1) == 3:
                                threat_defended = True

        elif column == 2:
                if row != 0 and row != 1 and row != 8:
                        diag1 = [board[move - 10], board[move - 5], board[move], board[move + 5]]
                        if diag1.count(1) == 1 and diag1.count(-1) == 3:
                                threat_defended = True
                if row != 0 and row != 7 and row != 8:
                        diag2 = [board[move - 3], board[move], board[move + 3], board[move + 6]]
                        if diag2.count(1) == 1 and diag2.count(-1) == 3:
                                threat_defended = True

        elif column == 3:
                if row != 0 and row != 1 and row != 2:
                        diag1 = [board[move - 15], board[move - 10], board[move - 5], board[move]]
                        if diag1.count(1) == 1 and diag1.count(-1) == 3:
                                threat_defended = True
                if row != 6 and row != 7 and row != 8:
                        diag2 = [board[move], board[move + 3], board[move + 6], board[move + 9]]
                        if diag2.count(1) == 1 and diag2.count(-1) == 3:
                                threat_defended = True

        return threat_defended


# Distance to board center
move_total = np.zeros(36)
distance_data = np.zeros(36)
distance_nn = np.zeros(36)
distance_rand = np.zeros(36)

# For each board position
for ind in range(num_moves):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Pick a move according to the network probabilities
        net_prob = np.exp(output)/np.sum(np.exp(output))
        net_move = np.random.choice(np.arange(36), p=net_prob)

        # Grab the number of pieces on the board and the distance from the center
        num_pieces = np.count_nonzero(board)
        move_total[num_pieces] += 1
        distance_data[num_pieces] += distance_from_pieces(17.5, [target])
        distance_nn[num_pieces] += distance_from_pieces(17.5, [net_move])
        distance_rand[num_pieces] += distance_from_pieces(17.5, [np.random.choice(np.where(board == 0)[0])])

# Divide to compute the averages, remove the nans and return
avg_data_center = distance_data/move_total
avg_data_center = avg_data_center[~np.isnan(avg_data_center)]
avg_nn_center = distance_nn/move_total
avg_nn_center = avg_nn_center[~np.isnan(avg_nn_center)]
avg_rand_center = distance_rand/move_total
avg_rand_center = avg_rand_center[~np.isnan(avg_rand_center)]

# Plot
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(np.arange(1,37,2), avg_data_center, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_center, lw=2, color='darkblue', label='network')
ax.plot(np.arange(1,37,2), avg_rand_center, lw=2, ls='--', color='darkgreen', label='random')
ax.set_xlim(0,37)
ax.set_xticks([0,5,10,15,20,25,30,35])
ax.set_xlabel('Move number')
ax.set_ylabel('Distance to board center')
ax.set_ylim(-.1,5.55)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_center.png', format='png', dpi=1000, bbox_inches='tight')

# Distance from own pieces
move_total = np.zeros(36)
distance_data_player = np.zeros(36)
distance_nn_player = np.zeros(36)
distance_rand_player = np.zeros(36)

# Distance from opponent's pieces
distance_data_opponent = np.zeros(36)
distance_nn_opponent = np.zeros(36)
distance_rand_opponent = np.zeros(36)

# For each board position
for ind in range(num_moves):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Pick a move according to the network probabilities
        net_prob = np.exp(output)/np.sum(np.exp(output))
        net_move = np.random.choice(np.arange(36), p=net_prob)

        # Grab the number of pieces on the board and the distance from the player and opponent's pieces
        num_pieces = np.count_nonzero(board)
        move_total[num_pieces] += 1

        player_pieces = np.where(board==1)[0]
        opponent_pieces = np.where(board == -1)[0]
        distance_data_player[num_pieces] += distance_from_pieces(target, player_pieces)
        distance_nn_player[num_pieces] += distance_from_pieces(net_move, player_pieces)
        distance_rand_player[num_pieces] += distance_from_pieces(np.random.choice(np.where(board == 0)[0]), player_pieces)
        distance_data_opponent[num_pieces] += distance_from_pieces(target, opponent_pieces)
        distance_nn_opponent[num_pieces] += distance_from_pieces(net_move, opponent_pieces)
        distance_rand_opponent[num_pieces] += distance_from_pieces(np.random.choice(np.where(board == 0)[0]), opponent_pieces)

# Divide to compute the averages, remove the nans and return
avg_data_player = distance_data_player/move_total
avg_data_player = avg_data_player[~np.isnan(avg_data_player)]
avg_nn_player = distance_nn_player/move_total
avg_nn_player = avg_nn_player[~np.isnan(avg_nn_player)]
avg_rand_player = distance_rand_player/move_total
avg_rand_player = avg_rand_player[~np.isnan(avg_rand_player)]

avg_data_opponent = distance_data_opponent/move_total
avg_data_opponent = avg_data_opponent[~np.isnan(avg_data_opponent)]
avg_nn_opponent = distance_nn_opponent/move_total
avg_nn_opponent = avg_nn_opponent[~np.isnan(avg_nn_opponent)]
avg_rand_opponent = distance_rand_opponent/move_total
avg_rand_opponent = avg_rand_opponent[~np.isnan(avg_rand_opponent)]

# Plot
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(np.arange(1,37,2), avg_data_player, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_player, lw=2, color='darkblue', label='network')
ax.plot(np.arange(1,37,2), avg_rand_player, lw=2, ls='--',color='darkgreen', label='random')
ax.set_xlim(0,37)
ax.set_xticks([0,5,10,15,20,25,30,35])
ax.set_xlabel('Move number')
ax.set_ylabel('Average distance to own pieces')
ax.set_ylim(-.1,5.55)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_own_pieces.png', format='png', dpi=1000, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(np.arange(1,37,2), avg_data_opponent, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_opponent, lw=2, color='darkblue', label='network')
ax.plot(np.arange(1,37,2), avg_rand_opponent, lw=2, ls='--',color='darkgreen', label='random')
ax.set_xlim(0,37)
ax.set_xticks([0,5,10,15,20,25,30,35])
ax.set_xlabel('Move number')
ax.set_ylabel('Average distance to opponent\'s pieces')
ax.set_ylim(-.1,5.55)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_opp_pieces.png', format='png', dpi=1000, bbox_inches='tight')

# Distance from own mass
move_total = np.zeros(36)
distance_data_player_mass = np.zeros(36)
distance_nn_player_mass = np.zeros(36)
distance_rand_player_mass = np.zeros(36)

# Distance from opponent's mass
distance_data_opponent_mass = np.zeros(36)
distance_nn_opponent_mass = np.zeros(36)
distance_rand_opponent_mass = np.zeros(36)

# For each board position
for ind in range(num_moves):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Pick a move according to the network probabilities
        net_prob = np.exp(output)/np.sum(np.exp(output))
        net_move = np.random.choice(np.arange(36), p=net_prob)

        # Grab the number of pieces on the board and the distance from the player and opponent's pieces
        num_pieces = np.count_nonzero(board)
        move_total[num_pieces] += 1

        player_pieces = np.where(board==1)[0]
        opponent_pieces = np.where(board == -1)[0]
        distance_data_player_mass[num_pieces] += distance_from_mass(target, player_pieces)
        distance_nn_player_mass[num_pieces] += distance_from_mass(net_move, player_pieces)
        distance_rand_player_mass[num_pieces] += distance_from_mass(np.random.choice(np.where(board == 0)[0]), player_pieces)
        distance_data_opponent_mass[num_pieces] += distance_from_mass(target, opponent_pieces)
        distance_nn_opponent_mass[num_pieces] += distance_from_mass(net_move, opponent_pieces)
        distance_rand_opponent_mass[num_pieces] += distance_from_mass(np.random.choice(np.where(board == 0)[0]), opponent_pieces)

# Divide to compute the averages, remove the nans and return
avg_data_player_mass = distance_data_player_mass/move_total
avg_data_player_mass = avg_data_player_mass[~np.isnan(avg_data_player_mass)]
avg_nn_player_mass = distance_nn_player_mass/move_total
avg_nn_player_mass = avg_nn_player_mass[~np.isnan(avg_nn_player_mass)]
avg_rand_player_mass = distance_rand_player_mass/move_total
avg_rand_player_mass = avg_rand_player_mass[~np.isnan(avg_rand_player_mass)]

avg_data_opponent_mass = distance_data_opponent_mass/move_total
avg_data_opponent_mass = avg_data_opponent_mass[~np.isnan(avg_data_opponent_mass)]
avg_nn_opponent_mass = distance_nn_opponent_mass/move_total
avg_nn_opponent_mass = avg_nn_opponent_mass[~np.isnan(avg_nn_opponent_mass)]
avg_rand_opponent_mass = distance_rand_opponent_mass/move_total
avg_rand_opponent_mass = avg_rand_opponent_mass[~np.isnan(avg_rand_opponent_mass)]

# Plot
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(np.arange(1,37,2), avg_data_player_mass, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_player_mass, lw=2, color='darkblue', label='network')
ax.plot(np.arange(1,37,2), avg_rand_player_mass, lw=2, ls='--',color='darkgreen', label='random')
ax.set_xlim(0,37)
ax.set_xticks([0,5,10,15,20,25,30,35])
ax.set_xlabel('Move number')
ax.set_ylabel('Distance to own center of mass')
ax.set_ylim(-.1,5.55)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_own_mass.png', format='png', dpi=1000, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(np.arange(1,37,2), avg_data_opponent_mass, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_opponent_mass, lw=2, color='darkblue', label='network')
ax.plot(np.arange(1,37,2), avg_rand_opponent_mass, lw=2, ls='--',color='darkgreen', label='random')
ax.set_xlim(0,37)
ax.set_xticks([0,5,10,15,20,25,30,35])
ax.set_xlabel('Move number')
ax.set_ylabel('Distance to opponent\'s center of mass')
ax.set_ylim(-.1,5.55)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_opp_mass.png', format='png', dpi=1000, bbox_inches='tight')

# Number of neighboring own pieces
move_total = np.zeros(36)
neighbor_data_player = np.zeros(36)
neighbor_nn_player = np.zeros(36)
neighbor_rand_player = np.zeros(36)

# Number of neighboring opponent's pieces
neighbor_data_opponent = np.zeros(36)
neighbor_nn_opponent = np.zeros(36)
neighbor_rand_opponent = np.zeros(36)

# For each board position
for ind in range(num_moves):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Pick a move according to the network probabilities
        net_prob = np.exp(output)/np.sum(np.exp(output))
        net_move = np.random.choice(np.arange(36), p=net_prob)

        # Grab the number of pieces on the board and the distance from the player and opponent's pieces
        num_pieces = np.count_nonzero(board)
        move_total[num_pieces] += 1

        player_pieces = np.where(board==1)[0]
        opponent_pieces = np.where(board == -1)[0]
        neighbor_data_player[num_pieces] += number_neighbors(target, player_pieces)
        neighbor_nn_player[num_pieces] += number_neighbors(net_move, player_pieces)
        neighbor_rand_player[num_pieces] += number_neighbors(np.random.choice(np.where(board == 0)[0]), player_pieces)
        neighbor_data_opponent[num_pieces] += number_neighbors(target, opponent_pieces)
        neighbor_nn_opponent[num_pieces] += number_neighbors(net_move, opponent_pieces)
        neighbor_rand_opponent[num_pieces] += number_neighbors(np.random.choice(np.where(board == 0)[0]), opponent_pieces)

# Divide to compute the averages, remove the nans and return
avg_data_player_neighbor = neighbor_data_player/move_total
avg_data_player_neighbor = avg_data_player_neighbor[~np.isnan(avg_data_player_neighbor)]
avg_nn_player_neighbor = neighbor_nn_player/move_total
avg_nn_player_neighbor = avg_nn_player_neighbor[~np.isnan(avg_nn_player_neighbor)]
avg_rand_player_neighbor = neighbor_rand_player/move_total
avg_rand_player_neighbor = avg_rand_player_neighbor[~np.isnan(avg_rand_player_neighbor)]

avg_data_opponent_neighbor = neighbor_data_opponent/move_total
avg_data_opponent_neighbor = avg_data_opponent_neighbor[~np.isnan(avg_data_opponent_neighbor)]
avg_nn_opponent_neighbor = neighbor_nn_opponent/move_total
avg_nn_opponent_neighbor = avg_nn_opponent_neighbor[~np.isnan(avg_nn_opponent_neighbor)]
avg_rand_opponent_neighbor = neighbor_rand_opponent/move_total
avg_rand_opponent_neighbor = avg_rand_opponent_neighbor[~np.isnan(avg_rand_opponent_neighbor)]

# Plot
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(np.arange(1,37,2), avg_data_player_neighbor, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_player_neighbor, lw=2, color='darkblue', label='network')
ax.plot(np.arange(1,37,2), avg_rand_player_neighbor, lw=2, ls='--',color='darkgreen', label='random')
ax.set_xlim(0,37)
ax.set_xticks([0,5,10,15,20,25,30,35])
ax.set_xlabel('Move number')
ax.set_ylabel('Number of own neighboring pieces')
ax.set_ylim(-.1,2.55)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_own_neighbors.png', format='png', dpi=1000, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(np.arange(1,37,2), avg_data_opponent_neighbor, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_opponent_neighbor, lw=2, color='darkblue', label='network')
ax.plot(np.arange(1,37,2), avg_rand_opponent_neighbor, lw=2, ls='--',color='darkgreen', label='random')
ax.set_xlim(0,37)
ax.set_xticks([0,5,10,15,20,25,30,35])
ax.set_xlabel('Move number')
ax.set_ylabel('Number of opponent\'s neighboring pieces')
ax.set_ylim(-.1,2.55)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_opp_neighbors.png', format='png', dpi=1000, bbox_inches='tight')

# Number of threats made
move_total = np.zeros(36)
threats_data_player = np.zeros(36)
threats_nn_player = np.zeros(36)
threats_rand_player = np.zeros(36)

# Number of threats defended
threats_data_opponent = np.zeros(36)
threats_nn_opponent = np.zeros(36)
threats_rand_opponent = np.zeros(36)

# For each board position
for ind in range(num_moves):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Pick a move according to the network probabilities
        net_prob = np.exp(output)/np.sum(np.exp(output))
        net_move = np.random.choice(np.arange(36), p=net_prob)

        # Grab the number of pieces on the board and if a threat was made
        num_pieces = np.count_nonzero(board)
        move_total[num_pieces] += 1

        if check_threat_made(target,board):
                threats_data_player[num_pieces] += 1
        if check_threat_made(net_move,board):
                threats_nn_player[num_pieces] += 1
        if check_threat_made(np.random.choice(np.where(board == 0)[0]),board):
                threats_rand_player[num_pieces] += 1

        if check_threat_defended(target,board):
                threats_data_opponent[num_pieces] += 1
        if check_threat_defended(net_move,board):
                threats_nn_opponent[num_pieces] += 1
        if check_threat_defended(np.random.choice(np.where(board == 0)[0]),board):
                threats_rand_opponent[num_pieces] += 1

# Divide to compute the averages, remove the nans and return
avg_data_player_threats = threats_data_player/move_total
avg_data_player_threats = avg_data_player_threats[~np.isnan(avg_data_player_threats)]
avg_nn_player_threats = threats_nn_player/move_total
avg_nn_player_threats = avg_nn_player_threats[~np.isnan(avg_nn_player_threats)]
avg_rand_player_threats = threats_rand_player/move_total
avg_rand_player_threats = avg_rand_player_threats[~np.isnan(avg_rand_player_threats)]

avg_data_opponent_threats = threats_data_opponent/move_total
avg_data_opponent_threats = avg_data_opponent_threats[~np.isnan(avg_data_opponent_threats)]
avg_nn_opponent_threats = threats_nn_opponent/move_total
avg_nn_opponent_threats = avg_nn_opponent_threats[~np.isnan(avg_nn_opponent_threats)]
avg_rand_opponent_threats = threats_rand_opponent/move_total
avg_rand_opponent_threats = avg_rand_opponent_threats[~np.isnan(avg_rand_opponent_threats)]

# Plot
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(np.arange(1,37,2), avg_data_player_threats, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_player_threats, lw=2, color='darkblue', label='network')
ax.plot(np.arange(1,37,2), avg_rand_player_threats, lw=2, ls='--',color='darkgreen', label='random')
ax.set_xlim(0,37)
ax.set_xticks([0,5,10,15,20,25,30,35])
ax.set_xlabel('Move number')
ax.set_ylabel('Number of threats made')
ax.set_ylim(-.01,0.52)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_threats_made.png', format='png', dpi=1000, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(np.arange(1,37,2), avg_data_opponent_threats, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_opponent_threats, lw=2, color='darkblue', label='network')
ax.plot(np.arange(1,37,2), avg_rand_opponent_threats, lw=2, ls='--',color='darkgreen', label='random')
ax.set_xlim(0,37)
ax.set_xticks([0,5,10,15,20,25,30,35])
ax.set_xlabel('Move number')
ax.set_ylabel('Number of threats defended')
ax.set_ylim(-.01,0.52)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_threats_defended.png', format='png', dpi=1000, bbox_inches='tight')

#%% OPENING THEORY

# Convert all the arrays we have to lists
boards_list = boards.tolist()
outputs_list = outputs.tolist()
targets_list = targets.tolist()

# Find the most common opening moves in the data and visualize the top 1
opening_inds = board_subset_len(boards_list, 0)
opening_boards = [boards_list[i] for i in opening_inds]
opening_outputs = [outputs_list[i] for i in opening_inds]
opening_targets = [targets_list[i][0] for i in opening_inds]

opening_rank, first_ind = board_target_rank(opening_targets)
op1 = 17
op_ind1 = first_ind[op1]
visualize_pred(np.asarray(opening_boards[op_ind1]), np.asarray(opening_outputs[op_ind1]), np.asarray(opening_targets[op_ind1]), save=True, filename='op1.png')

# Find the most common 2-piece positions for the most common opening
second_inds = board_subset_len(boards_list, 2)
second_boards = [boards_list[i] for i in second_inds]
second_outputs = [outputs_list[i] for i in second_inds]
second_targets = [targets_list[i][0] for i in second_inds]

opening1 = opening_boards[op_ind1]
opening1[op1] = 1
second_inds_op1 = board_subset_pattern(second_boards, opening1)
second_boards_op1 = [second_boards[i] for i in second_inds_op1]
second_outputs_op1 = [second_outputs[i] for i in second_inds_op1]
second_targets_op1 = [second_targets[i] for i in second_inds_op1]

second_op1_rank, second_ind = board_subset_rank(second_boards_op1)
second1 = (17, 18)
second_ind1 = second_ind[second1]
visualize_pred(np.asarray(second_boards_op1[second_ind1]), np.asarray(second_outputs_op1[second_ind1]), np.asarray(second_targets_op1[second_ind1]), outputOn=False, targetOn=False, save=True, filename='second1.png')
second2 = (17, 13)
second_ind2 = second_ind[second2]
visualize_pred(np.asarray(second_boards_op1[second_ind2]), np.asarray(second_outputs_op1[second_ind2]), np.asarray(second_targets_op1[second_ind2]), outputOn=False, targetOn=False, save=True, filename='second2.png')
second3 = (17, 21)
second_ind3 = second_ind[second3]
visualize_pred(np.asarray(second_boards_op1[second_ind3]), np.asarray(second_outputs_op1[second_ind3]), np.asarray(second_targets_op1[second_ind3]), outputOn=False, targetOn=False, save=True, filename='second3.png')
second4 = (17, 22)
second_ind4 = second_ind[second4]
visualize_pred(np.asarray(second_boards_op1[second_ind4]), np.asarray(second_outputs_op1[second_ind4]), np.asarray(second_targets_op1[second_ind4]), outputOn=False, targetOn=False, save=True, filename='second4.png')

# Find the most common third move responses for the most common opening and 2-piece board
second1_inds = board_subset_pattern(second_boards_op1, second_boards_op1[second_ind1])
third_boards_second1 = [second_boards_op1[i] for i in second1_inds]
third_outputs_second1 = [second_outputs_op1[i] for i in second1_inds]
third_targets_second1 = [second_targets_op1[i] for i in second1_inds]
third_second1_rank, third_second1_ind = board_target_rank(third_targets_second1)
third1 = 13
third_ind1 = third_second1_ind[third1]
visualize_pred(np.asarray(third_boards_second1[third_ind1]), np.asarray(third_outputs_second1[third_ind1]), np.asarray(third_targets_second1[third_ind1]), outputOn=True, targetOn=True, save=True, filename='third1.png')
third2 = 14
third_ind2 = third_second1_ind[third2]
visualize_pred(np.asarray(third_boards_second1[third_ind2]), np.asarray(third_outputs_second1[third_ind2]), np.asarray(third_targets_second1[third_ind2]), outputOn=True, targetOn=True, save=True, filename='third2.png')
third3 = 21
third_ind3 = third_second1_ind[third3]
visualize_pred(np.asarray(third_boards_second1[third_ind3]), np.asarray(third_outputs_second1[third_ind3]), np.asarray(third_targets_second1[third_ind3]), outputOn=True, targetOn=True, save=True, filename='third3.png')
third4 = 22
third_ind4 = third_second1_ind[third4]
visualize_pred(np.asarray(third_boards_second1[third_ind4]), np.asarray(third_outputs_second1[third_ind4]), np.asarray(third_targets_second1[third_ind4]), outputOn=True, targetOn=True, save=True, filename='third4.png')

# Find the most common 4-piece positions for the most common opening
fourth_inds = board_subset_len(boards_list, 4)
fourth_boards = [boards_list[i] for i in fourth_inds]
fourth_outputs = [outputs_list[i] for i in fourth_inds]
fourth_targets = [targets_list[i][0] for i in fourth_inds]

opening3 = third_boards_second1[third_ind1]
opening3[third1] = 1
fourth_inds_op1 = board_subset_pattern(fourth_boards, opening3)
fourth_boards_op1 = [fourth_boards[i] for i in fourth_inds_op1]
fourth_outputs_op1 = [fourth_outputs[i] for i in fourth_inds_op1]
fourth_targets_op1 = [fourth_targets[i] for i in fourth_inds_op1]

fourth_op1_rank, fourth_ind = board_subset_rank(fourth_boards_op1)
fourth1 = (13, 17, 18, 21)
fourth_ind1 = fourth_ind[fourth1]
visualize_pred(np.asarray(fourth_boards_op1[fourth_ind1]), np.asarray(fourth_outputs_op1[fourth_ind1]), np.asarray(fourth_targets_op1[fourth_ind1]), outputOn=False, targetOn=False, save=True, filename='fourth1.png')
fourth2 = (13, 17, 9, 18)
fourth_ind2 = fourth_ind[fourth2]
visualize_pred(np.asarray(fourth_boards_op1[fourth_ind2]), np.asarray(fourth_outputs_op1[fourth_ind2]), np.asarray(fourth_targets_op1[fourth_ind2]), outputOn=False, targetOn=False, save=True, filename='fourth2.png')
fourth3 = (13, 17, 14, 18)
fourth_ind3 = fourth_ind[fourth3]
visualize_pred(np.asarray(fourth_boards_op1[fourth_ind3]), np.asarray(fourth_outputs_op1[fourth_ind3]), np.asarray(fourth_targets_op1[fourth_ind3]), outputOn=False, targetOn=False, save=True, filename='fourth3.png')

# Find the most common fifth move responses for the most common opening and 4-piece board
fourth1_inds = board_subset_pattern(fourth_boards_op1, fourth_boards_op1[fourth_ind1])
fifth_boards_fourth1 = [fourth_boards_op1[i] for i in fourth1_inds]
fifth_outputs_fourth1 = [fourth_outputs_op1[i] for i in fourth1_inds]
fifth_targets_fourth1 = [fourth_targets_op1[i] for i in fourth1_inds]
fifth_fourth1_rank, fifth_fourth1_ind = board_target_rank(fifth_targets_fourth1)
fifth1 = 9
fifth_ind1 = fifth_fourth1_ind[fifth1]
visualize_pred(np.asarray(fifth_boards_fourth1[fifth_ind1]), np.asarray(fifth_outputs_fourth1[fifth_ind1]), np.asarray(fifth_targets_fourth1[fifth_ind1]), outputOn=True, targetOn=True, save=True, filename='fifth1.png')
fifth2 = 15
fifth_ind2 = fifth_fourth1_ind[fifth2]
visualize_pred(np.asarray(fifth_boards_fourth1[fifth_ind2]), np.asarray(fifth_outputs_fourth1[fifth_ind2]), np.asarray(fifth_targets_fourth1[fifth_ind2]), outputOn=True, targetOn=True, save=True, filename='fifth2.png')

# Find the most common 6-piece positions for the most common opening
sixth_inds = board_subset_len(boards_list, 6)
sixth_boards = [boards_list[i] for i in sixth_inds]
sixth_outputs = [outputs_list[i] for i in sixth_inds]
sixth_targets = [targets_list[i][0] for i in sixth_inds]

opening5 = fifth_boards_fourth1[fifth_ind1]
opening5[fifth1] = 1
sixth_inds_op1 = board_subset_pattern(sixth_boards, opening5)
sixth_boards_op1 = [sixth_boards[i] for i in sixth_inds_op1]
sixth_outputs_op1 = [sixth_outputs[i] for i in sixth_inds_op1]
sixth_targets_op1 = [sixth_targets[i] for i in sixth_inds_op1]

sixth_op1_rank, sixth_ind = board_subset_rank(sixth_boards_op1)
sixth1 = (9, 13, 17, 5, 18, 21)
sixth_ind1 = sixth_ind[sixth1]
visualize_pred(np.asarray(sixth_boards_op1[sixth_ind1]), np.asarray(sixth_outputs_op1[sixth_ind1]), np.asarray(sixth_targets_op1[sixth_ind1]), outputOn=False, targetOn=False, save=True, filename='sixth1.png')

# Find the most common seventh move responses for the most common opening and 6-piece board
sixth1_inds = board_subset_pattern(sixth_boards_op1, sixth_boards_op1[sixth_ind1])
seventh_boards_sixth1 = [sixth_boards_op1[i] for i in sixth1_inds]
seventh_outputs_sixth1 = [sixth_outputs_op1[i] for i in sixth1_inds]
seventh_targets_sixth1 = [sixth_targets_op1[i] for i in sixth1_inds]
seventh_sixth1_rank, seventh_sixth1_ind = board_target_rank(seventh_targets_sixth1)

seventh1 = 15
seventh_ind1 = seventh_sixth1_ind[seventh1]
visualize_pred(np.asarray(seventh_boards_sixth1[seventh_ind1]), np.asarray(seventh_outputs_sixth1[seventh_ind1]), np.asarray(seventh_targets_sixth1[seventh_ind1]), outputOn=True, targetOn=True, save=True, filename='seventh1.png')
seventh2 = 14
seventh_ind2 = seventh_sixth1_ind[seventh2]
visualize_pred(np.asarray(seventh_boards_sixth1[seventh_ind2]), np.asarray(seventh_outputs_sixth1[seventh_ind2]), np.asarray(seventh_targets_sixth1[seventh_ind2]), outputOn=True, targetOn=True, save=True, filename='seventh2.png')

# Find the most common paths for the second and third 2-piece boards
second2_inds = board_subset_pattern(second_boards_op1, second_boards_op1[second_ind2])
third_boards_second2 = [second_boards_op1[i] for i in second2_inds]
third_outputs_second2 = [second_outputs_op1[i] for i in second2_inds]
third_targets_second2 = [second_targets_op1[i] for i in second2_inds]
third_second2_rank, third_second2_ind = board_target_rank(third_targets_second2)
third_alt1 = 14
third_indalt1 = third_second2_ind[third_alt1]
visualize_pred(np.asarray(third_boards_second2[third_indalt1]), np.asarray(third_outputs_second2[third_indalt1]), np.asarray(third_targets_second2[third_indalt1]), outputOn=True, targetOn=True, save=True, filename='third_alt1.png')

second3_inds = board_subset_pattern(second_boards_op1, second_boards_op1[second_ind3])
third_boards_second3 = [second_boards_op1[i] for i in second3_inds]
third_outputs_second3 = [second_outputs_op1[i] for i in second3_inds]
third_targets_second3 = [second_targets_op1[i] for i in second3_inds]
third_second3_rank, third_second3_ind = board_target_rank(third_targets_second3)
third_alt2 = 14
third_indalt2 = third_second3_ind[third_alt2]
visualize_pred(np.asarray(third_boards_second3[third_indalt2]), np.asarray(third_outputs_second3[third_indalt2]), np.asarray(third_targets_second3[third_indalt2]), outputOn=True, targetOn=True, save=True, filename='third_alt2.png')


opening3_alt1 = third_boards_second2[third_indalt1]
opening3_alt1[third_alt1] = 1
fourth_inds_op2 = board_subset_pattern(fourth_boards, opening3_alt1)
fourth_boards_op2 = [fourth_boards[i] for i in fourth_inds_op2]
fourth_outputs_op2 = [fourth_outputs[i] for i in fourth_inds_op2]
fourth_targets_op2 = [fourth_targets[i] for i in fourth_inds_op2]

fourth_op2_rank, fourth_ind_alt1 = board_subset_rank(fourth_boards_op2)
fourth_alt1 = (14, 17, 13, 18)
fourth_indalt1 = fourth_ind_alt1[fourth_alt1]
visualize_pred(np.asarray(fourth_boards_op2[fourth_indalt1]), np.asarray(fourth_outputs_op2[fourth_indalt1]), np.asarray(fourth_targets_op2[fourth_indalt1]), outputOn=False, targetOn=False, save=True, filename='fourth_alt1.png')

opening3_alt2 = third_boards_second3[third_indalt2]
opening3_alt2[third_alt2] = 1
fourth_inds_op3 = board_subset_pattern(fourth_boards, opening3_alt2)
fourth_boards_op3 = [fourth_boards[i] for i in fourth_inds_op3]
fourth_outputs_op3 = [fourth_outputs[i] for i in fourth_inds_op3]
fourth_targets_op3 = [fourth_targets[i] for i in fourth_inds_op3]

fourth_op3_rank, fourth_ind_alt2 = board_subset_rank(fourth_boards_op3)
fourth_alt2 = (14, 17, 18, 21)
fourth_indalt2 = fourth_ind_alt2[fourth_alt2]
visualize_pred(np.asarray(fourth_boards_op3[fourth_indalt2]), np.asarray(fourth_outputs_op3[fourth_indalt2]), np.asarray(fourth_targets_op3[fourth_indalt2]), outputOn=False, targetOn=False, save=True, filename='fourth_alt2.png')


fourth1_alt1inds = board_subset_pattern(fourth_boards_op2, fourth_boards_op2[fourth_indalt1])
fifth_boards_fourth_alt1 = [fourth_boards_op2[i] for i in fourth1_alt1inds]
fifth_outputs_fourth_alt1 = [fourth_outputs_op2[i] for i in fourth1_alt1inds]
fifth_targets_fourth_alt1 = [fourth_targets_op2[i] for i in fourth1_alt1inds]
fifth_fourthalt1_rank, fifth_fourthalt1_ind = board_target_rank(fifth_targets_fourth_alt1)
fifth_alt1 = 11
fifth_indalt1 = fifth_fourthalt1_ind[fifth_alt1]
visualize_pred(np.asarray(fifth_boards_fourth_alt1[fifth_indalt1]), np.asarray(fifth_outputs_fourth_alt1[fifth_indalt1]), np.asarray(fifth_targets_fourth_alt1[fifth_indalt1]), outputOn=True, targetOn=True, save=True, filename='fifth_alt1.png')

fourth1_alt2inds = board_subset_pattern(fourth_boards_op3, fourth_boards_op3[fourth_indalt2])
fifth_boards_fourth_alt2 = [fourth_boards_op3[i] for i in fourth1_alt2inds]
fifth_outputs_fourth_alt2 = [fourth_outputs_op3[i] for i in fourth1_alt2inds]
fifth_targets_fourth_alt2 = [fourth_targets_op3[i] for i in fourth1_alt2inds]
fifth_fourthalt2_rank, fifth_fourthalt2_ind = board_target_rank(fifth_targets_fourth_alt2)
fifth_alt2 = 20
fifth_indalt2 = fifth_fourthalt2_ind[fifth_alt2]
visualize_pred(np.asarray(fifth_boards_fourth_alt2[fifth_indalt2]), np.asarray(fifth_outputs_fourth_alt2[fifth_indalt2]), np.asarray(fifth_targets_fourth_alt2[fifth_indalt2]), outputOn=True, targetOn=True, save=True, filename='fifth_alt2.png')


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