import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sn
import numpy as np
import pandas as pd
import os
import itertools

from popfinder._neural_networks import ClassifierNet
from popfinder._helper import _generate_train_inputs
from popfinder._helper import _generate_data_loaders
from popfinder._helper import _data_converter
from popfinder._helper import _split_input_classifier

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

        self.test_results = pd.DataFrame({"y_test": y_true,
                                          "y_pred": y_pred})
        self.confusion_matrix = np.round(
            confusion_matrix(y_true, y_pred, normalize="true"), 3)
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

        plt.switch_backend("agg")
        fig = plt.figure(figsize=(3, 1.5), dpi=200)
        plt.rcParams.update({"font.size": 7})
        ax1 = fig.add_axes([0, 0, 1, 1])
        ax1.plot(self.train_history["valid"][3:], "--", color="black",
            lw=0.5, label="Validation Loss")
        ax1.plot(self.train_history["train"][3:], "-", color="black",
            lw=0.5, label="Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        if save:
            fig.savefig(self.output_folder + "/training_history.png",
                bbox_inches="tight")

        plt.close()

    def plot_confusion_matrix(self, save=True):

        true_labels = self.test_results["y_test"]

        cm = np.round(self.confusion_matrix, 2)
        plt.style.use("default")
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.ylabel("True Population")
        plt.xlabel("Predicted Population")
        plt.title("Confusion Matrix")
        tick_marks = np.arange(len(np.unique(true_labels)))
        plt.xticks(tick_marks, np.unique(true_labels))
        plt.yticks(tick_marks, np.unique(true_labels))
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()

        if save:
            plt.savefig(self.output_folder + "/confusion_matrix.png")

        plt.close()

    def plot_roc_curve():

        pass # add later

    def plot_assignment(self, save=True, col_scheme="Spectral"):

        if self.classification is None:
            raise ValueError("No classification results to plot.")

        e_preds = self.classification.copy()
        e_preds.set_index("sampleID", inplace=True)

        # One hot encode assigned populations
        e_preds = pd.get_dummies(e_preds["assigned_pop"])
        num_classes = len(e_preds.columns) # will need to double check

        sn.set()
        sn.set_style("ticks")
        e_preds.plot(kind="bar", stacked=True,
            colormap=ListedColormap(sn.color_palette(col_scheme, num_classes)),
            figsize=(12, 6), grid=None)
        legend = plt.legend(
            loc="center right",
            bbox_to_anchor=(1.2, 0.5),
            prop={"size": 15},
            title="Predicted Population",
        )
        plt.setp(legend.get_title(), fontsize="x-large")
        plt.xlabel("Sample ID", fontsize=20)
        plt.ylabel("Frequency of Assignment", fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        if save:
            plt.savefig(self.output_folder + "/assignment_plot.png",
                bbox_inches="tight")

        plt.close()

    def plot_structure(self, preds, save=True, col_scheme="Spectral"):
        """
        Plots the proportion of times individuals from the
        test data were assigned to the correct population. 
        Used for determining the accuracy of the classifier.
        """
        preds = preds.drop(preds.columns[0], axis=1) # replace preds
        npreds = preds.groupby(["true_pops"]).agg("mean")
        npreds = npreds.sort_values("true_pops", ascending=True)
        npreds = npreds / np.sum(npreds, axis=1)

        # Make sure values are correct
        if not np.round(np.sum(npreds, axis=1), 2).eq(1).all():
            raise ValueError("Incorrect input values")

        # Find number of unique classes
        num_classes = len(npreds.index)

        if not len(npreds.index) == len(npreds.columns):
            raise ValueError(
                "Number of pops does not \
                match number of predicted pops"
            )

        sn.set()
        sn.set_style("ticks")
        npreds.plot(kind="bar", stacked=True,
            colormap=ListedColormap(sn.color_palette(col_scheme, num_classes)),
            figsize=(12, 6), grid=None)
        legend = plt.legend(loc="center right", bbox_to_anchor=(1.2, 0.5),
            prop={"size": 15}, title="Predicted Pop")
        plt.setp(legend.get_title(), fontsize="x-large")
        plt.xlabel("Actual Pop", fontsize=20)
        plt.ylabel("Frequency of Assignment", fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if save:
            plt.savefig(self.output_folder + "/structure_plot.png",
                bbox_inches="tight")
