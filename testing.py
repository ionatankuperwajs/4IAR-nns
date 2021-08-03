"""
Test a network's performance in terms of accuracy and NLL
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

#%% Function to test the network
# def test(net, test_set, batch_size):
#
#     # Initialize the test set
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=40)
#
#     # Set loss function
#     loss = nn.CrossEntropyLoss(reduction='none')
#
#     test_output = np.zeros((len(test_set),3))
#     count = 0
#     for data, target in test_loader:
#         # Run the data through the network
#         output = net(data)
#         loss_size = loss(output, target)
#
#         # Save out the loss size, prediction, and target
#         test_output[count*38:(count+1)*38, 0] = loss_size.detach()
#         test_output[count*38:(count+1)*38, 1] = torch.max(output, dim=1)[1]
#         test_output[count*38:(count+1)*38, 2] = target
#         count += 1
#
#     return test_output


#%% Function to run a model on the test set, returns the percent correct and nll and save out every board position,
#   prediction, and target for the test data set
def test_performance(net, test_set, batch_size, model_version):

    # Open a file to save out the test boards with model predictions and human moves
    results_file = open('../networks/' + str(model_version) + '/results_file.txt', "w")

    # Initialize the test set
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=40)

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
        np.savetxt(results_file, data, delimiter=',', newline=',')
        results_file.write('%f,%f \n' % (pred,target))
    perc_correct = 100. * correct / len(test_loader)
    nll = running_loss / len(test_loader)

    # Save out the accuracy and nll to a text file
    with open('../networks/' + str(model_version) + '/final_stats.txt', "w") as f:
        f.write('%f \n %f' % (perc_correct,nll))

#%% Same function as above, but returns only the percent correct and nll
def test_performance_minimal(net, test_set, batch_size, model_version):

    # Initialize the test set
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=40)

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

    # Save out the accuracy and nll to a text file
    with open('../networks/' + str(model_version) + '/final_stats.txt', "w") as f:
        f.write('%f \n %f' % (perc_correct,nll))

#%% Function to run a model on the test set per user, returns the percent correct and nll for each as a vector
# def test_performance_user(net, user_paths):
#
#     # Initialize lists to return for each user
#     perc_correct_all = []
#     nll_all = []
#
#     # Loop over all users
#     for user in user_paths.keys():
#
#         # Initialize the test set for the user
#         test_set = PeakDataset(user_paths[user])
#         test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
#
#         # Set loss function
#         loss = nn.CrossEntropyLoss()
#
#         correct = 0
#         running_loss = 0
#
#         for data, target in test_loader:
#
#             # Run the data through the network
#             output = net(data)
#             loss_size = loss(output, target)
#             running_loss += loss_size.item()
#             # Compare prediction to ground truth (get the index of the max log-probability)
#             pred = torch.max(output, dim=1)[1]
#             correct += torch.eq(pred,target)
#
#         perc_correct = 100. * correct / len(test_loader)
#         nll = running_loss / len(test_loader)
#
#         perc_correct_all.append(perc_correct.item())
#         nll_all.append(nll)
#
#     return perc_correct_all, nll_all

#%% Function to run a model on the test set and return the accuracy by move number in the game
# def test_by_move(net, test_set):
#
#     # Initialize a numpy array for each move (number correct, total number)
#     moves = np.zeros(36)
#     totals = np.zeros(36)
#
#     # Initialize the test set
#     test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
#
#     for data, target in test_loader:
#
#         # Run the data through the network
#         output = net(data)
#
#         # For the current move number, get the prediction and compare with ground truth
#         move_num = int(data.sum().item())
#         pred = torch.max(output, dim=1)[1]
#         moves[move_num] += torch.eq(pred, target)
#         totals[move_num] += 1
#
#     # Divide to compute the accuracy, remove the nans and  return
#     move_accuracy = moves/totals
#     return move_accuracy[~np.isnan(move_accuracy)]

