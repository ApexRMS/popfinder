import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os

from _neural_networks import ClassifierNet

class PopClassifier(object):
    """
    A class to represent a classifier neural network object for population assignment.
    """
    def __init__(self, random_state, output_folder=None):
        self.random_state = random_state
        if output_folder is None:
            output_folder = os.getcwd()
        self.output_folder = output_folder
        self.train_history = None
        self.best_model = None

    def train(self, X_train, y_train, X_valid, y_valid, epochs=100):
        
        train = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train, batch_size=16, shuffle=True)
        valid = TensorDataset(X_valid, y_valid)
        valid_loader = DataLoader(valid, batch_size=16, shuffle=True)

        net = ClassifierNet(X_train.shape[1], 16, len(y_train.unique()))
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loss_func = nn.CrossEntropyLoss()
        loss_dict = {"epoch": [], "train": [], "valid": []}
        lowest_val_loss = 1

        for epoch in range(epochs):

            train_loss = 0
            valid_loss = 0

            for _, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = net(data)
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item
        
            # Calculate average train loss
            avg_train_loss = train_loss / len(train_loader)

            for _, (data, target) in enumerate(valid_loader):
                output = net(data)
                loss = loss_func(output, target)
                valid_loss += loss.item

                if valid_loss < lowest_val_loss:
                    lowest_val_loss = valid_loss
                    torch.save(net, os.path.join(self.output_folder, "best_model.pt"))

            # Calculate average validation loss
            avg_valid_loss = valid_loss / len(valid_loader)

            loss_dict["epoch"].append(epoch)
            loss_dict["train"].append(avg_train_loss)
            loss_dict["valid"].append(avg_valid_loss)

        self.train_history = loss_dict
        self.best_model = torch.load(os.path.join(self.output_folder, "best_model.pt"))

    def test(self, X_test, y_test):
        
        y_pred = self.best_mod(X_test)

        correct = (y_pred == y_test)
        accuracy = correct.sum() / correct.size

        return accuracy


    def assign_unknown(self, unknown_data):
        
        y_assign = self.best_mod(unknown_data["alleles"]).argmax(axis=1)

        return y_assign