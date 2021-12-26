"""
# Pre-processing the data for cognitive model fits
"""

import csv
import torch
import math
import numpy as np
import tqdm

#%% HELPER FUNCTIONS

# Function that maps class labels back to the correct move
def map_label_to_move(label):
    temp_tensor = np.zeros((9,4))
    row = math.floor(label / 4)
    column = label - (row * 4)
    temp_tensor[row, column] = 1
    return np.argwhere(np.flip(np.fliplr(temp_tensor)).flatten('F') == 1).transpose()[0]

# Function to return integer code for a list of moves
def encode_move_list(moves):
    return np.sum([2**m for m in moves])

#%% PRE-PROCESSING

# List of paths
games_path = '/Volumes/Samsung_T5/Peak/nn_data/train/%s/train_%d.pt'
meta_path = '/Volumes/Samsung_T5/Peak/nn_data/train_meta/%s/train_meta_%d.pt'

# Open a csv to write out to
player = 'network'
num_folders = 979

for sub_folder in range(num_folders):
    folder_string = f'{sub_folder:03}'
    if sub_folder == 0:
        count_init = 1
        num_iters = 9999
    elif sub_folder == 978:
        count_init = sub_folder*10000
        num_iters = 7094
    else:
        count_init = sub_folder*10000
        num_iters = 10000
    with open('/Volumes/Samsung_T5/Peak/nn_data/cognitive_model/train/' + player + '_' + str(sub_folder) + '.csv', 'w') as f1:
        writer = csv.writer(f1, delimiter='\t', lineterminator='\n', )
        for ii in tqdm.tqdm(range(num_iters)):
            game_count = count_init + ii
            games = torch.load(games_path % (folder_string, game_count))
            meta = torch.load(meta_path % (folder_string, game_count))

            # For each move, get the info we need: board state, chosen move, RT
            for move in range(len(games[0])):
                black_pieces = np.argwhere(np.flip(np.fliplr(games[0][move][0].numpy())).flatten('F') == 1).transpose()[0]
                black_code = int(encode_move_list(black_pieces))
                white_pieces = np.argwhere(np.flip(np.fliplr(games[0][move][1].numpy())).flatten('F') == 1).transpose()[0]
                white_code = int(encode_move_list(white_pieces))
                label = int(encode_move_list([map_label_to_move(games[1][move])[0]]))
                response_time = meta[2][move]/1000

                writer.writerow([black_code, white_code, 'Black', label, response_time, player])