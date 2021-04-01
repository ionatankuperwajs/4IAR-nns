"""
# Pre-processing the Peak data
"""

import json
import math
import torch
import numpy as np

#%% HELPER FUNCTIONS

# Function to get all of the moves in a game and return them in a list
def get_moves_from_json(data):
    return [d['positionValue'] for d in data['data']['rounds'][-1]['analytic']['events'] if
            d['stateString'] == 'Turn']

# Function to generate a tensor representing the game board from a list of moves
def create_tensor_from_list(move_list):
    # Loop through the moves and place them into a tensor
    game_tensor = torch.zeros(2, 9, 4)
    if len(move_list) == 0:
        return game_tensor
    for idx, move in enumerate(move_list):
        # Compute the position of the current move
        column = math.floor(move/9)
        row = abs(move-8-(column*8)-(column*1))
        # Place a 1 in that position based on who's turn it is (0 == user, 1 == AI)
        if idx % 2 == 0:
            game_tensor[0, row, column] = 1
        else:
            game_tensor[1, row, column] = 1
    return game_tensor

# Function that maps moves to the correct class labels
def map_move_to_label(move):
    return abs(move%9-8)*4+math.floor(move/9)

#%% PRE-PROCESSING

# Read in the filtered list of games
with open('../../Data/all_paths.txt', 'r') as filehandle:
    all_paths = json.load(filehandle)

# Split the paths into train, test, val
train_size = math.ceil(len(all_paths)*.9)
val_size = math.ceil(len(all_paths)*.05)
test_size = math.floor(len(all_paths)*.05)

train_paths = all_paths[:train_size]
val_paths = all_paths[train_size:train_size+val_size]
test_paths = all_paths[train_size+val_size:]

# List to store the number of moves in each game
num_moves = []

# Counter for game number
game_count = 1

# Paths for saving out preprocessed data
moves_path = '../../Data/train_moves.pt'
games_path = '../../Data/train/train_%d.pt'

# Loop through all the games (note: change references in loop to generate data for train, val, test)
for game_path in train_paths:

    # Initialize lists to hold the board state tensors and next move labels
    tensors = []
    labels = []

    # Open the datafile
    with(open(game_path)) as f:
        data = json.load(f)

    # Get all the moves in a list
    game_list = get_moves_from_json(data)

    # Take every user move in the game, create its board tensor representation, and grab the label
    for idx in range(0, len(game_list) - 1, 2):
        move_list = game_list[0:idx]
        move_tensor = create_tensor_from_list(move_list)
        move_label = map_move_to_label(game_list[idx])

        # Store the (tensor, label) tuple
        tensors.append(move_tensor)
        labels.append(move_label)

    # Save out the number of moves in the game
    num_moves.append(len(tensors))

    # Now store the list as a tensor and save it out
    tensors_stacked = torch.stack(tensors)
    labels_stacked = np.asarray(labels)

    # torch.save([tensors_stacked, labels_stacked], games_path % game_count)
    game_count += 1

# Save out the numpy array with total number of moves per game at the end
torch.save(np.asarray(num_moves), moves_path)
