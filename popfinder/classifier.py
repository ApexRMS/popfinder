import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import pickle
import numpy as np
import pandas as pd
import os

from popfinder._neural_networks import ClassifierNet
from popfinder._helper import _generate_train_inputs
from popfinder._helper import _generate_data_loaders
from popfinder._helper import _data_converter
from popfinder._helper import _split_input_classifier
from popfinder._helper import _save, _load
from popfinder._visualize import _plot_assignment
from popfinder._visualize import _plot_training_curve
from popfinder._visualize import _plot_confusion_matrix
from popfinder._visualize import _plot_structure

pd.options.mode.chained_assignment = None

class PopClassifier(object):
    """
    A class to represent a classifier neural network object for population assignment.
    """
    def __init__(self, data, random_state=123, output_folder=None):

        self.data = data # GeneticData object
        self.random_state = random_state
        if output_folder is None:
            output_folder = os.getcwd()
        self.output_folder = output_folder
        self.label_enc = None
        self.train_history = None
        self.best_model = None
        self.test_results = None # use for cm and structure plot
        self.classification = None # use for assignment plot
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.confusion_matrix = None

        self._nn_type = "classifier"

    def train(self, epochs=100, valid_size=0.2, cv_splits=1, cv_reps=1):

        inputs = _generate_train_inputs(self.data, valid_size, cv_splits,
                                        cv_reps, seed=self.random_state)
        loss_dict = {"rep": [], "split": [], "epoch": [], "train": [], "valid": []}
        lowest_val_loss = 9999

        for i, input in enumerate(inputs):

            X_train, y_train, X_valid, y_valid = _split_input_classifier(self, input)
            train_loader, valid_loader = _generate_data_loaders(X_train, y_train,
                                                                X_valid, y_valid)

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
        y_test = test_input["pop"]

        y_test = self.label_enc.transform(y_test)
        X_test, y_test = _data_converter(X_test, y_test)

        y_pred = self.best_model(X_test).argmax(axis=1)
        y_true = y_test.squeeze()

        # revert from label encoder
        y_pred_pops = self.label_enc.inverse_transform(y_pred)
        y_true_pops = self.label_enc.inverse_transform(y_true)

        self.test_results = pd.DataFrame({"true_pop": y_true_pops,
                                          "pred_pop": y_pred_pops})
        self.confusion_matrix = np.round(
            confusion_matrix(self.test_results["true_pop"],
                             self.test_results["pred_pop"], 
                             labels=np.unique(y_true_pops).tolist(),
                             normalize="true"), 3)
        self.accuracy = np.round(accuracy_score(y_true, y_pred), 3)
        self.precision = np.round(precision_score(y_true, y_pred, average="weighted"), 3)
        self.recall = np.round(recall_score(y_true, y_pred, average="weighted"), 3)
        self.f1 = np.round(f1_score(y_true, y_pred, average="weighted"), 3)

    def assign_unknown(self):
        
        unknown_data = self.data.unknowns

        X_unknown = unknown_data["alleles"]
        X_unknown = _data_converter(X_unknown, None)

        preds = self.best_model(X_unknown).argmax(axis=1)
        preds = self.label_enc.inverse_transform(preds)
        unknown_data.loc[:, "assigned_pop"] = preds

        self.classification = unknown_data
        
        return unknown_data
  
    # Reporting functions below
    def get_classification_summary(self):

        summary = {
            "accuracy": [self.accuracy],
            "precision": [self.precision],
            "recall": [self.recall],
            "f1": [self.f1],
            "confusion_matrix": [self.confusion_matrix]
        }

        return summary

    # Plotting functions below
    def plot_training_curve(self, save=True):

        _plot_training_curve(self.train_history, self._nn_type,
            self.output_folder, save)

    def plot_confusion_matrix(self, save=True):

        _plot_confusion_matrix(self.test_results, self.confusion_matrix,
            self._nn_type, self.output_folder, save)

    def plot_roc_curve():

        pass # add later

    def plot_assignment(self, save=True, col_scheme="Spectral"):

        if self.classification is None:
            raise ValueError("No classification results to plot.")

        e_preds = self.classification.copy()

        _plot_assignment(e_preds, col_scheme, self.output_folder,
            self._nn_type, save)

    def plot_structure(self, save=True, col_scheme="Spectral"):
        """
        Plots the proportion of times individuals from the
        test data were assigned to the correct population. 
        Used for determining the accuracy of the classifier.
        """
        preds = pd.DataFrame(self.confusion_matrix,
                             columns=self.label_enc.classes_,
                             index=self.label_enc.classes_)

        _plot_structure(preds, col_scheme, self._nn_type, 
            self.output_folder, save)

    def save(self, save_path=None, filename="classifier.pkl"):
        """
        Saves the current instance of the class to a pickle file.
        """
        _save(self, save_path, filename)

    def load(self, load_path=None, filename="classifier.pkl"):
        """
        Loads a saved instance of the class from a pickle file.
        """
        _save(self, load_path, filename)
