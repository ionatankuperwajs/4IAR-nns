"""
Main file to compute summary statistics
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import math
import tqdm

plt.rcParams.update({'font.size': 16})

#%% Loading in the results from a network on the test set

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

#%% Loading in the results from the baseline model

test_path = '../../Data/test_fits_baseline/'

test_ll = np.zeros(num_moves)
moves = np.zeros((num_moves, 36))

counter = 0
for i in range(55):
    curr_ll = np.loadtxt(test_path+'out'+str(i)+'_lltest.csv', delimiter=',')
    curr_moves = np.loadtxt(test_path+'out'+str(i)+'_moves.csv', delimiter=',')
    test_ll[counter:counter+np.shape(curr_ll)[0]] = curr_ll
    moves[counter:counter+np.shape(curr_moves)[0],:] = curr_moves
    counter += np.shape(curr_ll)[0]

#%% Summary statistic functions

# Function to convert a board index to a coordinate
def move_to_coordinate(move):
        row = math.floor(move/4)
        column = move-(row*4)
        return (row,column)

# Function that checks if a move was made on the left side of the board
def check_left(move):
        move = move_to_coordinate(move)
        column = move[1]
        if column == 0 or column == 1:
                return True
        elif column == 2 or column == 3:
                return False

# Function that computes the average Manhattan distance of a move from a set of pieces
def distance_from_pieces(move, pieces):
        total_distance = 0
        distance_count = 0
        move = move_to_coordinate(move)
        if len(pieces) == 0:
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
        if len(pieces) == 0:
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
        if len(pieces) == 0:
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

#%% Distance to board center

move_total = np.zeros(36)
distance_data = np.zeros(36)
distance_nn = np.zeros(36)
distance_model = np.zeros(36)
distance_rand = np.zeros(36)

# For each board position
for ind in tqdm.tqdm(range(num_moves)):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Pick a move according to the network probabilities
        net_prob = np.exp(output)/np.sum(np.exp(output))
        net_move = np.random.choice(np.arange(36), p=net_prob)

        # Get the cognitive model predictions
        # hist_model = np.flipud(moves[ind,:].reshape(4, 9, order='F')).flatten(order='F')
        hist_model = moves[ind,:]
        prediction_model = np.argmax(hist_model)

        # Grab the number of pieces on the board and the distance from the center
        num_pieces = np.count_nonzero(board)
        move_total[num_pieces] += 1
        distance_data[num_pieces] += distance_from_pieces(17.5, [target])
        distance_nn[num_pieces] += distance_from_pieces(17.5, [net_move])
        distance_model[num_pieces] += distance_from_pieces(17.5, [prediction_model])
        distance_rand[num_pieces] += distance_from_pieces(17.5, [np.random.choice(np.where(board == 0)[0])])

# Divide to compute the averages, remove the nans and return
avg_data_center = distance_data/move_total
avg_data_center = avg_data_center[~np.isnan(avg_data_center)]
avg_nn_center = distance_nn/move_total
avg_nn_center = avg_nn_center[~np.isnan(avg_nn_center)]
avg_model_center = distance_model/move_total
avg_model_center = avg_model_center[~np.isnan(avg_model_center)]
avg_rand_center = distance_rand/move_total
avg_rand_center = avg_rand_center[~np.isnan(avg_rand_center)]

# Plot
fig, ax = plt.subplots(figsize=(4,4))
# ax.scatter(np.arange(1,37,2), avg_data_center, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_center, lw=2, marker='o', color='darkblue', label='neural network')
ax.plot(np.arange(1,37,2), avg_model_center, lw=2, marker='o', color='darkorange', label='baseline model')
# ax.fill_between(np.arange(1,37,2), avg_nn_center, avg_model_center, color='lightslategray',alpha=0.2)
# ax.plot(np.arange(1,37,2), avg_rand_center, lw=2, ls='--', color='darkgreen', label='random')
# ax.set_xlim(0,37)
# ax.set_xticks([0,5,10,15,20,25,30,35])
ax.set_xlabel('Move number')
ax.set_ylabel('Distance to\nboard center')
ax.set_ylim(-.1,5.55)
ax.legend(frameon=False,bbox_to_anchor=(1.05, 1),loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_center.png', format='png', dpi=1000, bbox_inches='tight')

#%% Distance from pieces

# Distance from own pieces
move_total = np.zeros(36)
distance_data_player = np.zeros(36)
distance_nn_player = np.zeros(36)
distance_model_player = np.zeros(36)
distance_rand_player = np.zeros(36)

# Distance from opponent's pieces
distance_data_opponent = np.zeros(36)
distance_nn_opponent = np.zeros(36)
distance_model_opponent = np.zeros(36)
distance_rand_opponent = np.zeros(36)

# For each board position
for ind in tqdm.tqdm(range(num_moves)):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Pick a move according to the network probabilities
        net_prob = np.exp(output)/np.sum(np.exp(output))
        net_move = np.random.choice(np.arange(36), p=net_prob)

        # Get the cognitive model predictions
        # hist_model = np.flipud(moves[ind,:].reshape(4, 9, order='F')).flatten(order='F')
        hist_model = moves[ind,:]
        prediction_model = np.argmax(hist_model)

        # Grab the number of pieces on the board and the distance from the player and opponent's pieces
        num_pieces = np.count_nonzero(board)
        move_total[num_pieces] += 1

        player_pieces = np.where(board==1)[0]
        opponent_pieces = np.where(board == -1)[0]
        distance_data_player[num_pieces] += distance_from_pieces(target, player_pieces)
        distance_nn_player[num_pieces] += distance_from_pieces(net_move, player_pieces)
        distance_model_player[num_pieces] += distance_from_pieces(prediction_model, player_pieces)
        distance_rand_player[num_pieces] += distance_from_pieces(np.random.choice(np.where(board == 0)[0]), player_pieces)
        distance_data_opponent[num_pieces] += distance_from_pieces(target, opponent_pieces)
        distance_nn_opponent[num_pieces] += distance_from_pieces(net_move, opponent_pieces)
        distance_model_opponent[num_pieces] += distance_from_pieces(prediction_model, opponent_pieces)
        distance_rand_opponent[num_pieces] += distance_from_pieces(np.random.choice(np.where(board == 0)[0]), opponent_pieces)

# Divide to compute the averages, remove the nans and return
avg_data_player = distance_data_player/move_total
avg_data_player = avg_data_player[~np.isnan(avg_data_player)]
avg_nn_player = distance_nn_player/move_total
avg_nn_player = avg_nn_player[~np.isnan(avg_nn_player)]
avg_model_player = distance_model_player/move_total
avg_model_player = avg_model_player[~np.isnan(avg_model_player)]
avg_rand_player = distance_rand_player/move_total
avg_rand_player = avg_rand_player[~np.isnan(avg_rand_player)]

avg_data_opponent = distance_data_opponent/move_total
avg_data_opponent = avg_data_opponent[~np.isnan(avg_data_opponent)]
avg_nn_opponent = distance_nn_opponent/move_total
avg_nn_opponent = avg_nn_opponent[~np.isnan(avg_nn_opponent)]
avg_model_opponent = distance_model_opponent/move_total
avg_model_opponent = avg_model_opponent[~np.isnan(avg_model_opponent)]
avg_rand_opponent = distance_rand_opponent/move_total
avg_rand_opponent = avg_rand_opponent[~np.isnan(avg_rand_opponent)]

# Plot
fig, ax = plt.subplots(figsize=(4,4))
# ax.scatter(np.arange(1,37,2), avg_data_player, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_player, lw=2, marker='o', color='darkblue', label='neural network')
ax.plot(np.arange(1,37,2), avg_model_player, lw=2, marker='o', color='darkorange', label='planning model')
# ax.fill_between(np.arange(1,37,2), avg_nn_player, avg_model_player, color='lightslategray',alpha=0.2)
# ax.plot(np.arange(1,37,2), avg_rand_player, lw=2, ls='--',color='darkgreen', label='random')
# ax.set_xlim(0,37)
# ax.set_xticks([0,5,10,15,20,25,30,35])
ax.set_xlabel('Move number')
ax.set_ylabel('Distance to\nown pieces')
ax.set_ylim(-.1,5.55)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_own_pieces.png', format='png', dpi=1000, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(4,4))
# ax.scatter(np.arange(1,37,2), avg_data_opponent, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_opponent, lw=2, marker='o', color='darkblue', label='neural network')
ax.plot(np.arange(1,37,2), avg_model_opponent, lw=2, marker='o', color='darkorange', label='planning model')
# ax.fill_between(np.arange(1,37,2), avg_nn_opponent, avg_model_opponent, color='lightslategray',alpha=0.2)
# ax.plot(np.arange(1,37,2), avg_rand_opponent, lw=2, ls='--',color='darkgreen', label='random')
# ax.set_xlim(0,37)
# ax.set_xticks([0,5,10,15,20,25,30,35])
ax.set_xlabel('Move number')
ax.set_ylabel('Distance to\nopponent\'s pieces')
ax.set_ylim(-.1,5.55)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_opp_pieces.png', format='png', dpi=1000, bbox_inches='tight')

#%% Distance from center of mass

# Distance from own mass
move_total = np.zeros(36)
distance_data_player_mass = np.zeros(36)
distance_nn_player_mass = np.zeros(36)
distance_model_player_mass = np.zeros(36)
distance_rand_player_mass = np.zeros(36)

# Distance from opponent's mass
distance_data_opponent_mass = np.zeros(36)
distance_nn_opponent_mass = np.zeros(36)
distance_model_opponent_mass = np.zeros(36)
distance_rand_opponent_mass = np.zeros(36)

# For each board position
for ind in tqdm.tqdm(range(num_moves)):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Pick a move according to the network probabilities
        net_prob = np.exp(output)/np.sum(np.exp(output))
        net_move = np.random.choice(np.arange(36), p=net_prob)

        # Get the cognitive model predictions
        # hist_model = np.flipud(moves[ind,:].reshape(4, 9, order='F')).flatten(order='F')
        hist_model = moves[ind,:]
        prediction_model = np.argmax(hist_model)

        # Grab the number of pieces on the board and the distance from the player and opponent's pieces
        num_pieces = np.count_nonzero(board)
        move_total[num_pieces] += 1

        player_pieces = np.where(board==1)[0]
        opponent_pieces = np.where(board == -1)[0]
        distance_data_player_mass[num_pieces] += distance_from_mass(target, player_pieces)
        distance_nn_player_mass[num_pieces] += distance_from_mass(net_move, player_pieces)
        distance_model_player_mass[num_pieces] += distance_from_mass(prediction_model, player_pieces)
        distance_rand_player_mass[num_pieces] += distance_from_mass(np.random.choice(np.where(board == 0)[0]), player_pieces)
        distance_data_opponent_mass[num_pieces] += distance_from_mass(target, opponent_pieces)
        distance_nn_opponent_mass[num_pieces] += distance_from_mass(net_move, opponent_pieces)
        distance_model_opponent_mass[num_pieces] += distance_from_mass(prediction_model, opponent_pieces)
        distance_rand_opponent_mass[num_pieces] += distance_from_mass(np.random.choice(np.where(board == 0)[0]), opponent_pieces)

# Divide to compute the averages, remove the nans and return
avg_data_player_mass = distance_data_player_mass/move_total
avg_data_player_mass = avg_data_player_mass[~np.isnan(avg_data_player_mass)]
avg_nn_player_mass = distance_nn_player_mass/move_total
avg_nn_player_mass = avg_nn_player_mass[~np.isnan(avg_nn_player_mass)]
avg_model_player_mass = distance_model_player_mass/move_total
avg_model_player_mass = avg_model_player_mass[~np.isnan(avg_model_player_mass)]
avg_rand_player_mass = distance_rand_player_mass/move_total
avg_rand_player_mass = avg_rand_player_mass[~np.isnan(avg_rand_player_mass)]

avg_data_opponent_mass = distance_data_opponent_mass/move_total
avg_data_opponent_mass = avg_data_opponent_mass[~np.isnan(avg_data_opponent_mass)]
avg_nn_opponent_mass = distance_nn_opponent_mass/move_total
avg_nn_opponent_mass = avg_nn_opponent_mass[~np.isnan(avg_nn_opponent_mass)]
avg_model_opponent_mass = distance_model_opponent_mass/move_total
avg_model_opponent_mass = avg_model_opponent_mass[~np.isnan(avg_model_opponent_mass)]
avg_rand_opponent_mass = distance_rand_opponent_mass/move_total
avg_rand_opponent_mass = avg_rand_opponent_mass[~np.isnan(avg_rand_opponent_mass)]

# Plot
fig, ax = plt.subplots(figsize=(4,4))
# ax.scatter(np.arange(1,37,2), avg_data_player_mass, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_player_mass, lw=2, marker='o', color='darkblue', label='neural network')
ax.plot(np.arange(1,37,2), avg_model_player_mass, lw=2, marker='o', color='darkorange', label='planning model')
# ax.fill_between(np.arange(1,37,2), avg_nn_player_mass, avg_model_player_mass, color='lightslategray',alpha=0.2)
# ax.plot(np.arange(1,37,2), avg_rand_player_mass, lw=2, ls='--',color='darkgreen', label='random')
# ax.set_xlim(0,37)
# ax.set_xticks([0,5,10,15,20,25,30,35])
ax.set_xlabel('Move number')
ax.set_ylabel('Distance to\nown center of mass')
ax.set_ylim(-.1,5.55)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_own_mass.png', format='png', dpi=1000, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(4,4))
# ax.scatter(np.arange(1,37,2), avg_data_opponent_mass, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_opponent_mass, lw=2, marker='o', color='darkblue', label='neural network')
ax.plot(np.arange(1,37,2), avg_model_opponent_mass, lw=2, marker='o', color='darkorange', label='planning model')
# ax.fill_between(np.arange(1,37,2), avg_nn_opponent_mass, avg_model_opponent_mass, color='lightslategray',alpha=0.2)
# ax.plot(np.arange(1,37,2), avg_rand_opponent_mass, lw=2, ls='--',color='darkgreen', label='random')
# ax.set_xlim(0,37)
# ax.set_xticks([0,5,10,15,20,25,30,35])
ax.set_xlabel('Move number')
ax.set_ylabel('Distance to\nopponent\'s center of mass')
ax.set_ylim(-.1,5.55)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_opp_mass.png', format='png', dpi=1000, bbox_inches='tight')

#%% Neighboring pieces

# Number of neighboring own pieces
move_total = np.zeros(36)
neighbor_data_player = np.zeros(36)
neighbor_nn_player = np.zeros(36)
neighbor_model_player = np.zeros(36)
neighbor_rand_player = np.zeros(36)

# Number of neighboring opponent's pieces
neighbor_data_opponent = np.zeros(36)
neighbor_nn_opponent = np.zeros(36)
neighbor_model_opponent = np.zeros(36)
neighbor_rand_opponent = np.zeros(36)

# For each board position
for ind in tqdm.tqdm(range(num_moves)):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Pick a move according to the network probabilities
        net_prob = np.exp(output)/np.sum(np.exp(output))
        net_move = np.random.choice(np.arange(36), p=net_prob)

        # Get the cognitive model predictions
        # hist_model = np.flipud(moves[ind,:].reshape(4, 9, order='F')).flatten(order='F')
        hist_model = moves[ind,:]
        prediction_model = np.argmax(hist_model)

        # Grab the number of pieces on the board and the distance from the player and opponent's pieces
        num_pieces = np.count_nonzero(board)
        move_total[num_pieces] += 1

        player_pieces = np.where(board==1)[0]
        opponent_pieces = np.where(board == -1)[0]
        neighbor_data_player[num_pieces] += number_neighbors(target, player_pieces)
        neighbor_nn_player[num_pieces] += number_neighbors(net_move, player_pieces)
        neighbor_model_player[num_pieces] += number_neighbors(prediction_model, player_pieces)
        neighbor_rand_player[num_pieces] += number_neighbors(np.random.choice(np.where(board == 0)[0]), player_pieces)
        neighbor_data_opponent[num_pieces] += number_neighbors(target, opponent_pieces)
        neighbor_nn_opponent[num_pieces] += number_neighbors(net_move, opponent_pieces)
        neighbor_model_opponent[num_pieces] += number_neighbors(prediction_model, opponent_pieces)
        neighbor_rand_opponent[num_pieces] += number_neighbors(np.random.choice(np.where(board == 0)[0]), opponent_pieces)

# Divide to compute the averages, remove the nans and return
avg_data_player_neighbor = neighbor_data_player/move_total
avg_data_player_neighbor = avg_data_player_neighbor[~np.isnan(avg_data_player_neighbor)]
avg_nn_player_neighbor = neighbor_nn_player/move_total
avg_nn_player_neighbor = avg_nn_player_neighbor[~np.isnan(avg_nn_player_neighbor)]
avg_model_player_neighbor = neighbor_model_player/move_total
avg_model_player_neighbor = avg_model_player_neighbor[~np.isnan(avg_model_player_neighbor)]
avg_rand_player_neighbor = neighbor_rand_player/move_total
avg_rand_player_neighbor = avg_rand_player_neighbor[~np.isnan(avg_rand_player_neighbor)]

avg_data_opponent_neighbor = neighbor_data_opponent/move_total
avg_data_opponent_neighbor = avg_data_opponent_neighbor[~np.isnan(avg_data_opponent_neighbor)]
avg_nn_opponent_neighbor = neighbor_nn_opponent/move_total
avg_nn_opponent_neighbor = avg_nn_opponent_neighbor[~np.isnan(avg_nn_opponent_neighbor)]
avg_model_opponent_neighbor = neighbor_model_opponent/move_total
avg_model_opponent_neighbor = avg_model_opponent_neighbor[~np.isnan(avg_model_opponent_neighbor)]
avg_rand_opponent_neighbor = neighbor_rand_opponent/move_total
avg_rand_opponent_neighbor = avg_rand_opponent_neighbor[~np.isnan(avg_rand_opponent_neighbor)]

# Plot
fig, ax = plt.subplots(figsize=(4,4))
# ax.scatter(np.arange(1,37,2), avg_data_player_neighbor, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_player_neighbor, lw=2, marker='o', color='darkblue', label='neural network')
ax.plot(np.arange(1,37,2), avg_model_player_neighbor, lw=2, marker='o', color='darkorange', label='planning model')
# ax.fill_between(np.arange(1,37,2), avg_nn_player_neighbor, avg_model_player_neighbor, color='lightslategray',alpha=0.2)
# ax.plot(np.arange(1,37,2), avg_rand_player_neighbor, lw=2, ls='--',color='darkgreen', label='random')
# ax.set_xlim(0,37)
ax.set_xticks([0,10,20,30])
ax.set_xlabel('Move number')
ax.set_ylabel('Number of\nown neighbors')
ax.set_ylim(-.1,2.55)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_own_neighbors.png', format='png', dpi=1000, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(4,4))
# ax.scatter(np.arange(1,37,2), avg_data_opponent_neighbor, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_opponent_neighbor, lw=2, marker='o', color='darkblue', label='network')
ax.plot(np.arange(1,37,2), avg_model_opponent_neighbor, lw=2, marker='o', color='darkorange', label='planning model')
# ax.fill_between(np.arange(1,37,2), avg_nn_opponent_neighbor, avg_model_opponent_neighbor, color='lightslategray',alpha=0.2)
# ax.plot(np.arange(1,37,2), avg_rand_opponent_neighbor, lw=2, ls='--',color='darkgreen', label='random')
# ax.set_xlim(0,37)
ax.set_xticks([0,10,20,30])
ax.set_xlabel('Move number')
ax.set_ylabel('Number of\nopponent\'s neighbors')
ax.set_ylim(-.1,2.55)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_opp_neighbors.png', format='png', dpi=1000, bbox_inches='tight')

#%% Number of threats

# Number of threats made
move_total = np.zeros(36)
threats_data_player = np.zeros(36)
threats_nn_player = np.zeros(36)
threats_model_player = np.zeros(36)
threats_rand_player = np.zeros(36)

# Number of threats defended
threats_data_opponent = np.zeros(36)
threats_nn_opponent = np.zeros(36)
threats_model_opponent = np.zeros(36)
threats_rand_opponent = np.zeros(36)

# For each board position
for ind in tqdm.tqdm(range(num_moves)):
        # Define the components we need
        board = boards[ind, :].astype(np.int)
        output = outputs[ind, :]
        prediction = int(predictions[ind, :][0])
        target = int(targets[ind, :][0])

        # Pick a move according to the network probabilities
        net_prob = np.exp(output)/np.sum(np.exp(output))
        net_move = np.random.choice(np.arange(36), p=net_prob)

        # Get the cognitive model predictions
        # hist_model = np.flipud(moves[ind,:].reshape(4, 9, order='F')).flatten(order='F')
        hist_model = moves[ind,:]
        prediction_model = np.argmax(hist_model)

        # Grab the number of pieces on the board and if a threat was made
        num_pieces = np.count_nonzero(board)
        move_total[num_pieces] += 1

        if check_threat_made(target,board):
                threats_data_player[num_pieces] += 1
        if check_threat_made(net_move,board):
                threats_nn_player[num_pieces] += 1
        if check_threat_made(prediction_model,board):
                threats_model_player[num_pieces] += 1
        if check_threat_made(np.random.choice(np.where(board == 0)[0]),board):
                threats_rand_player[num_pieces] += 1

        if check_threat_defended(target,board):
                threats_data_opponent[num_pieces] += 1
        if check_threat_defended(net_move,board):
                threats_nn_opponent[num_pieces] += 1
        if check_threat_defended(prediction_model, board):
            threats_model_opponent[num_pieces] += 1
        if check_threat_defended(np.random.choice(np.where(board == 0)[0]),board):
                threats_rand_opponent[num_pieces] += 1

# Divide to compute the averages, remove the nans and return
avg_data_player_threats = threats_data_player/move_total
avg_data_player_threats = avg_data_player_threats[~np.isnan(avg_data_player_threats)]
avg_nn_player_threats = threats_nn_player/move_total
avg_nn_player_threats = avg_nn_player_threats[~np.isnan(avg_nn_player_threats)]
avg_model_player_threats = threats_model_player/move_total
avg_model_player_threats = avg_model_player_threats[~np.isnan(avg_model_player_threats)]
avg_rand_player_threats = threats_rand_player/move_total
avg_rand_player_threats = avg_rand_player_threats[~np.isnan(avg_rand_player_threats)]

avg_data_opponent_threats = threats_data_opponent/move_total
avg_data_opponent_threats = avg_data_opponent_threats[~np.isnan(avg_data_opponent_threats)]
avg_nn_opponent_threats = threats_nn_opponent/move_total
avg_nn_opponent_threats = avg_nn_opponent_threats[~np.isnan(avg_nn_opponent_threats)]
avg_model_opponent_threats = threats_model_opponent/move_total
avg_model_opponent_threats = avg_model_opponent_threats[~np.isnan(avg_model_opponent_threats)]
avg_rand_opponent_threats = threats_rand_opponent/move_total
avg_rand_opponent_threats = avg_rand_opponent_threats[~np.isnan(avg_rand_opponent_threats)]

# Plot
fig, ax = plt.subplots(figsize=(4,4))
# ax.scatter(np.arange(1,37,2), avg_data_player_threats, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_player_threats, lw=2, marker='o', color='darkblue',  label='neural network')
ax.plot(np.arange(1,37,2), avg_model_player_threats, lw=2, marker='o', color='darkorange', label='planning model')
# ax.fill_between(np.arange(1,37,2), avg_nn_player_threats, avg_model_player_threats, color='lightslategray',alpha=0.2)
# ax.plot(np.arange(1,37,2), avg_rand_player_threats, lw=2, ls='--',color='darkgreen', label='random')
# ax.set_xlim(0,37)
ax.set_xticks([0,10,20,30])
ax.set_xlabel('Move number')
ax.set_ylabel('Number of\nthreats made')
ax.set_ylim(-.01,0.62)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_threats_made.png', format='png', dpi=1000, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(4,4))
# ax.scatter(np.arange(1,37,2), avg_data_opponent_threats, lw=1, color='black', label='data')
ax.plot(np.arange(1,37,2), avg_nn_opponent_threats, lw=2, marker='o', color='darkblue', label='neural network')
ax.plot(np.arange(1,37,2), avg_model_opponent_threats, lw=2, marker='o', color='darkorange', label='planning')
# ax.fill_between(np.arange(1,37,2), avg_nn_opponent_threats, avg_model_opponent_threats, color='lightslategray',alpha=0.2)
# ax.plot(np.arange(1,37,2), avg_rand_opponent_threats, lw=2, ls='--',color='darkgreen', label='random')
# ax.set_xlim(0,37)
ax.set_xticks([0,10,20,30])
ax.set_xlabel('Move number')
ax.set_ylabel('Number of\nthreats defended')
ax.set_ylim(-.01,0.62)
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# plt.savefig('summary_threats_defended.png', format='png', dpi=1000, bbox_inches='tight')