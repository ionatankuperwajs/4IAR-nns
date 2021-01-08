"""
Functions to train a network, save it, and return test/val loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

#%% Function to train the network
def train(net, batch_size, n_epochs, learning_rate, train_set, val_set, L2, model_name):

    # Print all of the hyperparameters of the training iteration
    print("===== HYPERPARAMETERS =====")
    print("batch_size =", batch_size)
    print("epochs =", n_epochs)
    print("learning_rate =", learning_rate)
    print("=" * 27)

    # Initialize training and validation sets
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=True, num_workers=2)

    # Cross-entropy loss combines log softmax and NLL
    loss = nn.CrossEntropyLoss()

    # Stochastic gradient descent with L2 regularization
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=L2)

    # Set up a scheduler to decrease the learning rate if validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1, verbose=True)

    # Training start time
    training_start_time = time.time()

    # Set up lists for test and validation losses to plot later
    train_loss = []
    val_loss = []

    # Loop for n_epochs
    for epoch in range(n_epochs):

        # Set running loss and start time for training epoch
        running_loss = 0.0
        start_time = time.time()

        # Loop through the data loader
        for data in train_loader:

            # Get inputs and labels
            inputs, labels = data

            # Set parameter gradients to 0
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            # Update the training loss for the epoch
            running_loss += loss_size.item()

        # At the end of each epoch, do a forward pass on the validation set
        total_val_loss = 0
        for inputs, labels in val_loader:

            # Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.item()

        # Update the scheduler
        scheduler.step(total_val_loss/len(val_loader))

        # Print and save training and validation loss for the epoch
        print("Epoch {}\nTraining Loss = {:.2f}, Validation Loss = {:.2f}, took {:.2f}s".format(
            epoch + 1, running_loss / len(train_loader), total_val_loss / len(val_loader),
            time.time() - start_time))

        train_loss.append(running_loss/len(train_loader))
        val_loss.append(total_val_loss/len(val_loader))
        print('-' * 20)

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

    # Save the trained model
    torch.save(net.state_dict(), model_name)

    # Save the entire model, not just the state dict if desired
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'running_loss': running_loss,
        'train_loss': train_loss,
        'val_loss': val_loss
    }, 'full'+model_name)

    return train_loss, val_loss

