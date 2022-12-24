import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import spatial
import numpy as np
import os

from preprocess import _normalize_locations
from _neural_networks import RegressorNet

class PopRegressor(object):
    """
    A class to represent a regressor neural network object for population assignment.
    """
    def __init__(self, random_state, output_folder=None):
        self.random_state = random_state
        if output_folder is None:
            output_folder = os.getcwd()
        self.output_folder = output_folder
        self.train_history = None
        self.best_model = None
        self.median_distance = None
        self.mean_distance = None
        self.r2_lat = None
        self.r2_long = None


    def train(self, X_train, y_train, X_valid, y_valid, epochs=100):
        
        train = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train, batch_size=16, shuffle=True)
        valid = TensorDataset(X_valid, y_valid)
        valid_loader = DataLoader(valid, batch_size=16, shuffle=True)

        net = RegressorNet(X_train.shape[1], 16, len(y_train.unique()))
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loss_func = self._euclidean_dist_loss()
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
        normalized_preds = self._normalize_preds(y_pred)

        r2_long = np.corrcoef(normalized_preds[:, 0], y_test[:, 0])[0][1] ** 2
        r2_lat = np.corrcoef(normalized_preds[:, 1], y_test[:, 1])[0][1] ** 2

        return {"r2_long": r2_long, "r2_lat": r2_lat}


    def assign_unknown(self, unknown_data):
        
        pred = self.best_mod(unknown_data["alleles"]).argmax(axis=1)
        normalized_preds = self._normalize_preds(pred)

        dists = [
            spatial.distance.euclidean(
                normalized_preds[x, :], actual_locs[x, :]
            ) for x in range(len(normalized_preds))
        ]

        self.median_distance = np.median(dists)
        self.mean_distance = np.mean(dists)

        return dists

    def get_assignment_summary(self):

        summary = {
            "median_distance": self.median_distance,
            "mean_distance": self.mean_distance,
            "r2_long": self.r2_long,
            "r2_lat": self.r2_lat
        }

        return summary

    def _euclidean_dist_loss(self, y_pred, y_true):

        return np.sqrt(np.sum(np.square(y_pred - y_true), axis=-1))
        