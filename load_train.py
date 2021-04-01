"""
Main file to train networks
"""

from custom_dataset import PeakDataset
from network import Linear, CNN
from training import train
import torch
import os

#%% RUN FROM THE COMMAND LINE

def main(model_name, model_version, num_layers, num_units, num_filters, filter_size, stride, padding, batch_size, n_epochs, learning_rate,
         moves_path, data_path):

    # Grab the training and validation data as a DataLoader
    train_set = PeakDataset(moves_path+'/train_moves.pt', data_path+'/train/%s/train_%d.pt', 1)
    val_set = PeakDataset(moves_path+'/val_moves.pt', data_path+'/val/val_%d.pt', 0)

    # Check if a folder exists for this network version number, if not create it
    folder_path = '../networks/' + str(model_version)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    else:
        raise Exception("Model version already exists")

    # Initialize the network
    if model_name == 'linear':
        net = Linear(num_layers=num_layers, num_units=num_units)
        # Save out the hyperparameters
        torch.save({
            'model': model_name,
            'version': model_version,
            'layers': num_layers,
            'units': num_units,
            'batch': batch_size,
            'epoch': n_epochs,
            'lr': learning_rate,
        }, '../networks/' + str(model_version) + '/hparams')

    elif model_name == 'conv':
        net = CNN(num_layers=num_layers, num_filters=num_filters, filter_size=filter_size,  stride=stride, pad=padding)
        # Save out the hyperparameters
        torch.save({
            'model': model_name,
            'version': model_version,
            'layers': num_layers,
            'filters': num_filters,
            'size': filter_size,
            'stride': stride,
            'padding': padding,
            'batch': batch_size,
            'epoch': n_epochs,
            'lr': learning_rate,
        }, '../networks/' + str(model_version) + '/hparams')

    # Train the network
    train(net, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate,
                                 train_set=train_set, val_set=val_set, L2=0, model_name=model_name, model_version=model_version)

if __name__ == '__main__':
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('-m', '--model_name',
                       help="model type to be trained",
                       choices=['linear', 'conv'],
                       default='linear')
   parser.add_argument('-v', '--model_version',
                       help="model version number for saving",
                       type=int, default=1)
   parser.add_argument('-hl', '--num_layers',
                       help="number of hidden layers",
                       type=int,default=1)
   parser.add_argument('-u', '--num_units',
                       help="number of hidden units for fc network",
                       type=int,default=200)
   parser.add_argument('-f', '--num_filters',
                       help="number of filters for conv network",
                       type=int,default=4)
   parser.add_argument('-fs', '--filter_size',
                       help="filter size for conv network",
                       type=int,default=3)
   parser.add_argument('-s', '--stride',
                       help="stride size for conv network",
                       type=int,default=1)
   parser.add_argument('-p', '--padding',
                       help="padding size for conv network",
                       type=int,default=1)
   parser.add_argument('-b', '--batch_size',
                       help="size of a batch",
                       type=int, default=12)
   parser.add_argument('-e', '--n_epochs',
                       help='number of epochs',
                       type=int, default=1)
   parser.add_argument('-lr', '--learning_rate',
                       help="learning rate",
                       type=float, default=10**-3)
   parser.add_argument('-d', '--moves_path',
                       help="path for the move index data",
                       default='..')
   parser.add_argument('-d', '--data_path',
                       help="path for the training and validation data",
                       default='/nn_peakdata')
   args = parser.parse_args()

main(model_name=args.model_name, model_version=args.model_version, num_layers=args.num_layers, num_units=args.num_units,
     num_filters=args.num_filters, filter_size=args.filter_size, stride=args.stride, padding=args.padding,
     batch_size=args.batch_size, n_epochs=args.n_epochs, learning_rate=args.learning_rate, train_path=args.moves_path,
     val_path=args.data_path)