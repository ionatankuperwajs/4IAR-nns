"""
Main file to test networks
"""

from custom_dataset import PeakDataset
from network import Linear, LinearSkip, CNN
from testing import test_performance, test_performance_minimal
import torch

#%% Run from the command line

def main(model_name, model_version, moves_path, data_path):

    # Grab the testing data as a DataLoader
    test_set = PeakDataset(moves_path+'/test_moves.pt', data_path+'/test/%s/test_%d.pt')

    # Load the saved network
    hparams = torch.load('../networks/'+str(model_version)+'/hparams')
    model = torch.load('../networks/'+str(model_version)+'/model_9')
    net = LinearSkip(num_layers=hparams['layers'], num_units=hparams['units'])
    net.load_state_dict(model['model_state_dict'])

    if model_name == 'linear':
        net = Linear(num_layers=hparams['layers'], num_units=hparams['units'])
        net.load_state_dict(model['model_state_dict'])
    elif model_name == 'linearskip':
        net = LinearSkip(num_layers=hparams['layers'], num_units=hparams['units'], bottleneck=hparams['bottleneck'])
        net.load_state_dict(model['model_state_dict'])
    elif model_name == 'conv':
        net = CNN(num_layers=hparams['layers'], num_filters=hparams['filters'], filter_size=hparams['size'], stride=hparams['stride'], pad=hparams['padding'])
        net.load_state_dict(model['model_state_dict'])

    # Test the network
    test_performance(net, test_set, model_version)

if __name__ == '__main__':
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('-m', '--model_name',
                       help="model type to be trained",
                       choices=['linear', 'linearskip', 'conv'],
                       default='linear')
   parser.add_argument('-v', '--model_version',
                       help="model version number for saving",
                       type=int, default=1)
   parser.add_argument('-mp', '--moves_path',
                       help="path for the move index data",
                       default='..')
   parser.add_argument('-d', '--data_path',
                       help="path for the training and validation data",
                       default='/nn_peakdata')
   args = parser.parse_args()
   main(model_name=args.model_name, model_version=args.model_version,
        moves_path=args.moves_path, data_path=args.data_path)