"""
# Custom Dataset class for the Peak data
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import bisect

#%% CUSTOM DATASET

class PeakDataset(Dataset):

    # Load the path into the PeakDataset class
    def __init__(self, moves_path, folder_path, train):
        self.moves_path = moves_path
        self.folder_path = folder_path
        # Find the starting move number for each game
        moves = torch.load(self.moves_path)
        self.gamestart_index = np.cumsum(moves)
        # Boolean for train or val data
        self.train = train

    # Returns the number of moves in the path
    def __len__(self):
        return self.gamestart_index[-1]

    # Returns the (move tensor, label) tuple at index
    def __getitem__(self, index):

        # Find the game of the move at index and load the game
        game_index = bisect(self.gamestart_index, index)
        if self.train == 1:
            train_folder = f'{int(np.floor(game_index/10000)):03}'
            game = torch.load(self.folder_path % (train_folder, game_index+1))
        elif self.train == 0:
            game = torch.load(self.folder_path % (game_index+1))

        # Find the move and label in the game
        move_index = index - self.gamestart_index[game_index]
        board = game[0][move_index]
        label = game[1][move_index]

        return board, label