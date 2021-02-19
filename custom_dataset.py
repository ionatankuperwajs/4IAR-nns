"""
# Custom Dataset class for the Peak data
"""

import torch
import numpy as np
from torch.utils.data import Dataset

#%% CUSTOM DATASET

class PeakDataset(Dataset):

    # Load the path into the PeakDataset class
    def __init__(self, moves_path, folder_path):
        self.moves_path = moves_path
        self.folder_path = folder_path
        # Find the starting move number for each game
        moves = torch.load(self.moves_path)
        self.gamestart_index = np.cumsum(moves)

    # Returns the number of moves in the path
    def __len__(self):
        return self.gamestart_index[-1]

    # Returns the (move tensor, label) tuple at index
    def __getitem__(self, index):

        # Find the game of the move at index and load the game
        game_index = np.argmax(self.gamestart_index > index)
        game = torch.load(self.folder_path % (game_index+1))

        # Find the move and label in the game
        move_index = index - self.gamestart_index[game_index]
        board = game[0][move_index]
        label = game[1][move_index]

        return board, label