import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
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

    def train(self, epochs=100, valid_size=0.2, cv_splits=1, cv_reps=1,
              learning_rate=0.001, batch_size=16, dropout_prop=0):
        """
        Trains the classification neural network.

        Parameters
        ----------
        epochs : int, optional
            Number of epochs to train the neural network. The default is 100.
        valid_size : float, optional
            Proportion of data to use for validation. The default is 0.2.
        cv_splits : int, optional
            Number of cross-validation splits. The default is 1.
        cv_reps : int, optional
            Number of cross-validation repetitions. The default is 1.
        learning_rate : float, optional
            Learning rate for the neural network. The default is 0.001.
        batch_size : int, optional
            Batch size for the neural network. The default is 16.
        dropout_prop : float, optional
            Dropout proportion for the neural network. The default is 0.
        
        Returns
        -------
        None.
        """

        inputs = _generate_train_inputs(self.data, valid_size, cv_splits,
                                        cv_reps, seed=self.random_state)
        loss_dict = {"rep": [], "split": [], "epoch": [], "train": [], "valid": []}
        lowest_val_loss = 9999

        for i, input in enumerate(inputs):

            X_train, y_train, X_valid, y_valid = _split_input_classifier(self, input)
            train_loader, valid_loader = _generate_data_loaders(X_train, y_train,
                                                                X_valid, y_valid)

            net = ClassifierNet(input_size=X_train.shape[1], hidden_size=16,
                                output_size=len(y_train.unique()),
                                batch_size=batch_size, dropout_prop=dropout_prop)
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
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
        """
        Tests the classification neural network.
        
        Returns
        -------
        None.
        """
        
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

    def assign_unknown(self, save=True):
        """
        Assigns unknown samples to populations using the trained neural network.

        Parameters
        ----------
        save : bool, optional
            Whether to save the results to a csv file. The default is True.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing the unknown samples and their assigned populations.
        """
        
        unknown_data = self.data.unknowns

        X_unknown = unknown_data["alleles"]
        X_unknown = _data_converter(X_unknown, None)

        preds = self.best_model(X_unknown).argmax(axis=1)
        preds = self.label_enc.inverse_transform(preds)
        unknown_data.loc[:, "assigned_pop"] = preds

        self.classification = unknown_data

        if save:
            unknown_data.to_csv(os.path.join(self.output_folder,
                                "classifier_assignment_results.csv"),
                                index=False)
        
        return unknown_data
  
    # Reporting functions below
    def get_classification_summary(self, save=True):
        """
        Get a summary of the classification results.

        Parameters
        ----------
        save : bool, optional
            Whether to save the results to a csv file. The default is True.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing the classification summary.
        """

        summary = {
            "accuracy": [self.accuracy],
            "precision": [self.precision],
            "recall": [self.recall],
            "f1": [self.f1],
            "confusion_matrix": [self.confusion_matrix]
        }

        if save:
            summary = pd.DataFrame(summary)
            summary.to_csv(os.path.join(self.output_folder,
                          "classifier_classification_summary.csv"),
                           index=False)

        return summary

    def rank_site_importance(self, save=True):
        """
        Rank sites (SNPs) by importance in model performance.

        Parameters
        ----------
        save : bool, optional
            Whether to save the results to a csv file. The default is True.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing the ranked sites.
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. " + 
            "Please run the train() method first.")

        X = self.data.knowns["alleles"].to_numpy()
        X = np.stack(X)
        Y = self.data.knowns["pop"]
        enc = OneHotEncoder(handle_unknown="ignore")
        Y_enc = enc.fit_transform(Y.values.reshape(-1, 1)).toarray()
        snp_names = np.arange(1, X.shape[1] + 1)
        errors = []

        for i in range(X.shape[1]):
            X_temp = X.copy()
            X_temp[:, i] = np.random.choice(X_temp[:, i], X_temp.shape[0])
            X_temp = torch.from_numpy(X_temp).float()
            preds = self.best_model(X_temp).argmax(axis=1)
            errors.append(np.sum(preds != Y_enc.argmax(axis=1)) / len(Y))

        max_error = np.max(errors)
        importance = [1 - (e / max_error) for e in errors]
        importance_data = {"snp": snp_names, "error": errors,
                           "importance": importance}
        ranking = pd.DataFrame(importance_data).sort_values("importance",
                                                            ascending=False)

        if save:
            ranking.to_csv(os.path.join(self.output_folder,
                          "classifier_site_importance_ranking.csv"),
                           index=False)

        return ranking

    # Plotting functions below
    def plot_training_curve(self, save=True):
        """
        Plots the training curve.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the plot to a png file. The default is True.
            
        Returns
        -------
        None
        """

        _plot_training_curve(self.train_history, self._nn_type,
            self.output_folder, save)

    def plot_confusion_matrix(self, save=True):
        """
        Plots the confusion matrix.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the plot to a png file. The default is True.
        
        Returns
        -------
        None
        """

        _plot_confusion_matrix(self.test_results, self.confusion_matrix,
            self._nn_type, self.output_folder, save)

    def plot_assignment(self, save=True, col_scheme="Spectral"):
        """
        Plots the proportion of times each individual from the
        unknown data was assigned to each population.

        Parameters
        ----------
        save : bool, optional
            Whether to save the plot to a png file. The default is True.
        col_scheme : str, optional
            The colour scheme to use for the plot. The default is "Spectral".

        Returns
        -------
        None
        """

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

        Parameters
        ----------
        save : bool, optional
            Whether to save the plot to a png file. The default is True.
        col_scheme : str, optional
            The colour scheme to use for the plot. The default is "Spectral".
        
        Returns
        -------
        None
        """
        preds = pd.DataFrame(self.confusion_matrix,
                             columns=self.label_enc.classes_,
                             index=self.label_enc.classes_)

        _plot_structure(preds, col_scheme, self._nn_type, 
            self.output_folder, save)

    def save(self, save_path=None, filename="classifier.pkl"):
        """
        Saves the current instance of the class to a pickle file.

        Parameters
        ----------
        save_path : str, optional
            The path to save the file to. The default is None.
        filename : str, optional
            The name of the file to save. The default is "classifier.pkl".

        Returns
        -------
        None
        """
        _save(self, save_path, filename)

    def load(self, load_path=None, filename="classifier.pkl"):
        """
        Loads a saved instance of the class from a pickle file.

        Parameters
        ----------
        load_path : str, optional
            The path to load the file from. The default is None.
        filename : str, optional
            The name of the file to load. The default is "classifier.pkl".
        
        Returns
        -------
        None
        """
        _load(self, load_path, filename)
