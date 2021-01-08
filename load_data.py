"""
# Custom Dataset class to hold the Peak files and do pre-processing
"""

import json
import math
import torch
from torch.utils.data import Dataset

#%%
class PeakDataset(Dataset):

    # Create the dataset from a list of filtered game paths
    # The dataset is a list, and each entry is a (board state tensor, next move tensor) tuple
    def __init__(self, data_path_list):
        self.samples = []

        # Loop through all the games
        for game_path in data_path_list:

            # Open the datafile
            with(open(game_path)) as f:
                data = json.load(f)

            # Get all the moves in a list
            game_list = self.get_moves_from_json(data)

            # Take every user move in the game, create its board tensor representation, and grab the label
            for idx in range(0, len(game_list) - 1, 2):
                move_list = game_list[0:idx]
                move_tensor = self.create_tensor_from_list(move_list)
                move_label = self.map_move_to_label(game_list[idx])

                # Store the (tensor, label) tuple
                self.samples.append((move_tensor, move_label))

    # Returns the number of moves in the dataset
    def __len__(self):
        return len(self.samples)

    # Returns the (move tensor, label) tuple at idx
    def __getitem__(self, idx):
        return self.samples[idx]

    # Function to get all of the moves in a game and return them in a list
    def get_moves_from_json(self, data):
        return [d['positionValue'] for d in data['data']['rounds'][-1]['analytic']['events'] if
                d['stateString'] == 'Turn']

    # Function to generate a tensor representing the game board from a list of moves
    def create_tensor_from_list(self, move_list):
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
    def map_move_to_label(self, move):
        return abs(move%9-8)*4+math.floor(move/9)

