import torch
from torch.autograd import Variable
from scipy import spatial
import numpy as np
import pandas as pd
import os

from popfinder._neural_networks import RegressorNet
from popfinder._helper import _generate_train_inputs
from popfinder._helper import _generate_data_loaders
from popfinder._helper import _data_converter
from popfinder._helper import _split_input_regressor

class PopRegressor(object):
    """
    A class to represent a regressor neural network object for population assignment.
    """
    def __init__(self, data, random_state=123, output_folder=None):

        self.data = data # GeneticData object
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


    def train(self, epochs=100, valid_size=0.2, cv_splits=1, cv_reps=1):
        
        inputs = _generate_train_inputs(self.data, valid_size, cv_splits,
                                        cv_reps, seed=self.random_state)
        loss_dict = {"rep": [], "split": [], "epoch": [], "train": [], "valid": []}
        lowest_val_loss = 9999

        for i, input in enumerate(inputs):

            X_train, y_train, X_valid, y_valid = _split_input_regressor(input)
            train_loader, valid_loader = _generate_data_loaders(X_train, y_train,
                                                                X_valid, y_valid)

            net = RegressorNet(X_train.shape[1], 16, len(y_train.unique()))
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
            loss_func = self._euclidean_dist_loss

            for epoch in range(epochs):

                train_loss = 0
                valid_loss = 0

                for _, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = net(data)
                    loss = loss_func(output.squeeze(), target.squeeze().long())
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.data.item()
            
                # Calculate average train loss
                avg_train_loss = train_loss / len(train_loader)

                for _, (data, target) in enumerate(valid_loader):
                    output = net(data)
                    loss = loss_func(output.squeeze(), target.squeeze().long())
                    valid_loss += loss.data.item()

                    if valid_loss < lowest_val_loss:
                        lowest_val_loss = valid_loss
                        torch.save(net, os.path.join(self.output_folder, "best_model.pt"))

                # Calculate average validation loss
                avg_valid_loss = valid_loss / len(valid_loader)

                split = i % cv_splits + 1
                rep = int(i / cv_splits) + 1

                loss_dict["rep"].append(rep)
                loss_dict["split"].append(split)
                loss_dict["epoch"].append(epoch)
                loss_dict["train"].append(avg_train_loss)
                loss_dict["valid"].append(avg_valid_loss)

        self.train_history = pd.DataFrame(loss_dict)
        self.best_model = torch.load(os.path.join(self.output_folder, "best_model.pt"))

    def test(self):
        
        test_input = self.data.test

        X_test = test_input["alleles"]
        y_test = test_input[["x", "y"]]
        X_test, y_test = _data_converter(X_test, y_test)

        y_pred = self.best_model(X_test).detach().numpy()
        y_pred = self._unnormalize_locations(y_pred)

        dists = [
            spatial.distance.euclidean(
                y_pred[x, :], y_test[x, :]
            ) for x in range(len(y_pred))
        ]

        self.median_distance = np.median(dists) # won't be accurate, need to unnormalize
        self.mean_distance = np.mean(dists)
        self.r2_long = np.corrcoef(y_pred[:, 0], y_test[:, 0])[0][1] ** 2
        self.r2_lat = np.corrcoef(y_pred[:, 1], y_test[:, 1])[0][1] ** 2

        summary = self.get_assignment_summary()

        return pd.DataFrame(summary)

    def assign_unknown(self):
        
        unknown_data = self.data.unknowns

        X_unknown = unknown_data["alleles"]
        X_unknown = _data_converter(X_unknown, None)

        y_pred = self.best_model(X_unknown).detach().numpy()
        y_pred = self._unnormalize_locations(y_pred)
        unknown_data.loc[:, "x"] = y_pred[:, 0]
        unknown_data.loc[:, "y"] = y_pred[:, 1]

        return unknown_data

    def get_assignment_summary(self):

        summary = {
            "median_distance": [self.median_distance],
            "mean_distance": [self.mean_distance],
            "r2_long": [self.r2_long],
            "r2_lat": [self.r2_lat]
        }

        return summary

    def _euclidean_dist_loss(self, y_pred, y_true):

        y_pred = y_pred.detach().numpy()
        y_true = y_true.detach().numpy()

        loss = np.sqrt(np.sum(np.square(y_pred - y_true)))
        loss = Variable(torch.tensor(loss), requires_grad=True)

        return loss

    def _unnormalize_locations(self, y_pred):

        y_pred_norm = np.array(
            [[x[0] * self.data.sdlong + self.data.meanlong,
              x[1] * self.data.sdlat + self.data.meanlat
            ] for x in y_pred])

        return y_pred_norm
        