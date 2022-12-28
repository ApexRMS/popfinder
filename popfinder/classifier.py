import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn import preprocessing
from scipy import spatial
import numpy as np
import pandas as pd
import os

# from popfinder.preprocess import _normalize_locations
from popfinder.preprocess import split_train_test
from popfinder.preprocess import split_kfcv
from popfinder._neural_networks import ClassifierNet

pd.options.mode.chained_assignment = None

class PopClassifier(object):
    """
    A class to represent a classifier neural network object for population assignment.
    """
    def __init__(self, random_state=123, output_folder=None):
        self.random_state = random_state
        if output_folder is None:
            output_folder = os.getcwd()
        self.output_folder = output_folder
        self.label_enc = None
        self.train_history = None
        self.best_model = None
        self.accuracy = None

    def train(self, train_input, epochs=100, valid_size=0.2, cv=None, cv_reps=1):

        # Split into train and valid
        if cv is None:
            train_input, valid_input = split_train_test(train_input, test_size=valid_size)
            inputs = [(train_input, valid_input)]
        else:
            inputs = split_kfcv(train_input, n_splits=cv, n_reps=cv_reps, seed=self.random_state)

        loss_dict = {"rep": [], "split": [], "epoch": [], "train": [], "valid": []}
        lowest_val_loss = 9999
        rep = 1 
        split = 1

        for input in inputs:

            X_train, y_train, X_valid, y_valid = self._split_input(input)
            train_loader, valid_loader = self._generate_data_loaders(X_train, y_train, X_valid, y_valid)

            net = ClassifierNet(X_train.shape[1], 16, len(y_train.unique()))
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
            loss_func = nn.CrossEntropyLoss()

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

                loss_dict["rep"].append(rep)
                loss_dict["split"].append(split)
                loss_dict["epoch"].append(epoch)
                loss_dict["train"].append(avg_train_loss)
                loss_dict["valid"].append(avg_valid_loss)

            while split < cv:
                split += 1
            if split == cv:
                split = 1
                rep += 1

        self.train_history = pd.DataFrame(loss_dict)
        self.best_model = torch.load(os.path.join(self.output_folder, "best_model.pt"))

    def test(self, test_input):
        
        X_test = test_input["alleles"]
        y_test = test_input["pop"]

        y_test = self.label_enc.transform(y_test)
        X_test, y_test = self._data_converter(X_test, y_test)

        y_pred = self.best_model(X_test).argmax(axis=1)
        correct = (y_pred == y_test.squeeze())
        accuracy = correct.sum() / len(correct)

        self.accuracy = np.round(accuracy.data.item(), 3)

        return self.accuracy

    def assign_unknown(self, unknown_data):
        
        X_unknown = unknown_data["alleles"]
        X_unknown = self._data_converter(X_unknown, None)

        preds = self.best_model(X_unknown).argmax(axis=1)
        preds = self.label_enc.inverse_transform(preds)
        unknown_data.loc[:, "assigned_pop"] = preds
        
        return unknown_data

    def get_assignment_summary(self):

        summary = {
            "accuracy": self.accuracy,
        }

        return summary

    def _split_input(self, input):
            
        train_input, valid_input = input
    
        X_train = train_input["alleles"]
        X_valid = valid_input["alleles"]
        y_train = train_input["pop"] # one hot encode
        y_valid = valid_input["pop"] # one hot encode

        # Label encode y values
        self.label_enc = preprocessing.LabelEncoder()
        y_train = self.label_enc.fit_transform(y_train)
        y_valid = self.label_enc.transform(y_valid)

        X_train, y_train = self._data_converter(X_train, y_train)
        X_valid, y_valid = self._data_converter(X_valid, y_valid)

        return X_train, y_train, X_valid, y_valid

    def _generate_data_loaders(self, X_train, y_train, X_valid, y_valid):

        train = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train, batch_size=16, shuffle=True)
        valid = TensorDataset(X_valid, y_valid)
        valid_loader = DataLoader(valid, batch_size=16, shuffle=True)

        return train_loader, valid_loader

    def _data_converter(self, x, y, variable=False):

        features = torch.from_numpy(np.vstack(np.array(x)).astype(np.float32))
        if torch.isnan(features).sum() != 0:
            print("Remove NaNs from features")        
        if variable:
            features = Variable(features)

        if y is not None:
            targets = torch.from_numpy(np.vstack(np.array(y)))
            if torch.isnan(targets).sum() != 0:
                print("remove NaNs from target")
            if variable:
                targets = Variable(targets)
               
            return features, targets

        else:
            return features
