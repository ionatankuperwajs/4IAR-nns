"""
Test a network's performance in terms of accuracy and NLL
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from custom_dataset import PeakDataset

#%% Function to test the network
def test(net, test_set):

    # Initialize the test set
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # Set loss function
    loss = nn.CrossEntropyLoss(reduction='none')

    test_output = np.zeros((len(test_set),3))
    count = 0
    for data, target in test_loader:
        # Run the data through the network
        output = net(data)
        loss_size = loss(output, target)

        # Save out the loss size, prediction, and target
        test_output[count*38:(count+1)*38, 0] = loss_size.detach()
        test_output[count*38:(count+1)*38, 1] = torch.max(output, dim=1)[1]
        test_output[count*38:(count+1)*38, 2] = target
        count += 1

    return test_output


#%% Function to run a model on the test set, returns the percent correct and nll
def test_performance(net, test_set):
    # Initialize the test set
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    # Set loss function
    loss = nn.CrossEntropyLoss()

    correct = 0
    running_loss = 0
    for data, target in test_loader:
        # Run the data through the network
        output = net(data)
        loss_size = loss(output, target)
        running_loss += loss_size.item()
        # Compare prediction to ground truth (get the index of the max log-probability)
        pred = torch.max(output, dim=1)[1]
        correct += torch.eq(pred,target)
    perc_correct = 100. * correct / len(test_loader)
    nll = running_loss / len(test_loader)
    return perc_correct, nll

#%% Function to run a model on the test set per user, returns the percent correct and nll for each as a vector

def test_performance_user(net, user_paths):

    # Initialize lists to return for each user
    perc_correct_all = []
    nll_all = []

    # Loop over all users
    for user in user_paths.keys():

        # Initialize the test set for the user
        test_set = PeakDataset(user_paths[user])
        test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

        # Set loss function
        loss = nn.CrossEntropyLoss()

        correct = 0
        running_loss = 0

        for data, target in test_loader:

            # Run the data through the network
            output = net(data)
            loss_size = loss(output, target)
            running_loss += loss_size.item()
            # Compare prediction to ground truth (get the index of the max log-probability)
            pred = torch.max(output, dim=1)[1]
            correct += torch.eq(pred,target)

        perc_correct = 100. * correct / len(test_loader)
        nll = running_loss / len(test_loader)

        perc_correct_all.append(perc_correct.item())
        nll_all.append(nll)

    return perc_correct_all, nll_all

#%% Function to run a model on the test set and return the number of guesses it took to predict the right move
def test_with_guesses(net, test_set):

    # Initialize a numpy array with number of guesses
    guesses = np.zeros(36)

    # Initialize the test set
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    for data, target in test_loader:
        # Run the data through the network
        output = net(data)

        # Get the top 36 values and their indices from the output
        preds, idxs = torch.topk(output, 36, dim=1, sorted=True)

        # Now iterate through the sorted values until one matches the ground truth
        for guess in range(len(preds[0])):
            if torch.eq(idxs[0][guess], target):
                guesses[guess] += 1
                break

    # Convert to cumulative percent accuracy
    guesses_accuracy = np.zeros(36)
    for i in range(len(guesses)):
        guesses_accuracy[i] = np.sum(guesses[0:i+1])/np.sum(guesses)

    return guesses_accuracy

#%% Function to run a model on the test set and return the accuracy by move number in the game
def test_by_move(net, test_set):

    # Initialize a numpy array for each move (number correct, total number)
    moves = np.zeros(36)
    totals = np.zeros(36)

    # Initialize the test set
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    for data, target in test_loader:

        # Run the data through the network
        output = net(data)

        # For the current move number, get the prediction and compare with ground truth
        move_num = int(data.sum().item())
        pred = torch.max(output, dim=1)[1]
        moves[move_num] += torch.eq(pred, target)
        totals[move_num] += 1

    # Divide to compute the accuracy, remove the nans and  return
    move_accuracy = moves/totals
    return move_accuracy[~np.isnan(move_accuracy)]

