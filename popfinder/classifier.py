import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import numpy as np
import pandas as pd
import os

from popfinder.dataloader import GeneticData
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

        self._validate_init_inputs(data, random_state, output_folder)

        self.__data = data # GeneticData object
        self.__random_state = random_state
        if output_folder is None:
            output_folder = os.getcwd()
        self.__output_folder = output_folder
        self.__cv_output_folder = os.path.join(output_folder, "cv_results")
        if not os.path.exists(self.__cv_output_folder):
            os.makedirs(self.__cv_output_folder)
        self.__label_enc = None
        self.__train_history = None
        self.__best_model = None
        self.__test_results = None # use for cm and structure plot
        self.__cv_test_results = None
        self.__classification = None # use for assignment plot
        self.__accuracy = None
        self.__precision = None
        self.__recall = None
        self.__f1 = None
        self.__confusion_matrix = None
        self.__cv_accuracy = None
        self.__cv_precision = None
        self.__cv_recall = None
        self.__cv_f1 = None
        self.__cv_confusion_matrix = None
        self.__nn_type = "classifier"

    @property
    def data(self):
        return self.__data

    @property
    def random_state(self):
        return self.__random_state
    
    @property
    def output_folder(self):
        return self.__output_folder

    @property
    def label_enc(self):
        return self.__label_enc

    @label_enc.setter
    def label_enc(self, value):
        self.__label_enc = value

    @property
    def train_history(self):
        return self.__train_history

    @property
    def best_model(self):
        return self.__best_model

    @property
    def test_results(self):
        return self.__test_results
    
    @property
    def cv_test_results(self):
        return self.__cv_test_results

    @property
    def classification(self):
        return self.__classification

    @property
    def accuracy(self):
        return self.__accuracy

    @property
    def precision(self):
        return self.__precision

    @property
    def recall(self):
        return self.__recall

    @property
    def f1(self):
        return self.__f1
    
    @property
    def confusion_matrix(self):
        return self.__confusion_matrix

    @property
    def cv_accuracy(self):
        return self.__cv_accuracy

    @property
    def cv_precision(self):
        return self.__cv_precision

    @property
    def cv_recall(self):
        return self.__cv_recall

    @property
    def cv_f1(self):
        return self.__cv_f1
    
    @property
    def cv_confusion_matrix(self):
        return self.__cv_confusion_matrix

    @property
    def nn_type(self):
        return self.__nn_type

    def train(self, epochs=100, valid_size=0.2, cv_splits=1, cv_reps=1,
              learning_rate=0.001, batch_size=16, dropout_prop=0, bootstraps=None):
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
        bootstraps : int, optional
            Number of bootstraps to perform. The default is None.
        
        Returns
        -------
        None.
        """
        self._validate_train_inputs(epochs, valid_size, cv_splits, cv_reps,
                                    learning_rate, batch_size, dropout_prop)
        
        self.__lowest_val_loss_total = 9999

        if bootstraps is not None:
            self.__bootstrap_results = os.path.join(self.output_folder, "bootstrap_results")
            if not os.path.exists(self.__bootstrap_results):
                os.makedirs(self.__bootstrap_results)

            loss_df = pd.DataFrame()

            for i in range(bootstraps):
                boot_folder = os.path.join(self.__bootstrap_results, f"bootstrap_{i + 1}")
                if not os.path.exists(boot_folder):
                    os.makedirs(boot_folder)

                inputs = _generate_train_inputs(self.data, valid_size, cv_splits,
                                cv_reps, seed=self.random_state, bootstrap=True)
                boot_loss_df = self.__train_on_inputs(inputs, cv_splits, epochs, learning_rate,
                                    batch_size, dropout_prop, result_folder = boot_folder)
                
                boot_loss_df.to_csv(os.path.join(boot_folder, "loss.csv"), index=False)
                boot_loss_df["bootstrap"] = i + 1
                loss_df = pd.concat([loss_df, boot_loss_df], axis=0, ignore_index=True)
        else:
            inputs = _generate_train_inputs(self.data, valid_size, cv_splits,
                                            cv_reps, seed=self.random_state, bootstrap=False)
            loss_df = self.__train_on_inputs(inputs, cv_splits, epochs, learning_rate,
                                                batch_size, dropout_prop, result_folder = self.__cv_output_folder)

        self.__train_history = loss_df
        self.__best_model = torch.load(os.path.join(self.output_folder, "best_model.pt"))

    def __train_on_inputs(self, inputs, cv_splits, epochs, learning_rate, batch_size, 
                          dropout_prop, result_folder):

        loss_dict = {"rep": [], "split": [], "epoch": [], "train": [], "valid": []}

        for i, input in enumerate(inputs):

            lowest_val_loss_rep = 9999
            split = i % cv_splits + 1
            rep = int(i / cv_splits) + 1

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

                    if valid_loss < lowest_val_loss_rep:
                        lowest_val_loss_rep = valid_loss
                        torch.save(net, os.path.join(result_folder, 
                                                     f"best_model_split{split}_rep{rep}.pt"))

                    if valid_loss < self.__lowest_val_loss_total:
                        self.__lowest_val_loss_total = valid_loss
                        torch.save(net, os.path.join(self.output_folder, "best_model.pt"))

                # Calculate average validation loss
                avg_valid_loss = valid_loss / len(valid_loader)

                loss_dict["rep"].append(rep)
                loss_dict["split"].append(split)
                loss_dict["epoch"].append(epoch)
                loss_dict["train"].append(avg_train_loss)
                loss_dict["valid"].append(avg_valid_loss)

        return pd.DataFrame(loss_dict)

    def test(self, best_model_only=True, save=True):
        """
        Tests the classification neural network.

        Parameters
        ----------
        best_model_only : bool, optional
            Whether to only test the best model only. If set to False, then will use all
            models generated from all training repeats and cross-validation splits.
            The default is True.
        save : bool, optional
            Whether to save the test results to the output folder. The default is True.
        
        Returns
        -------
        None.
        """
        # Find unique reps/splits from cross validation
        reps = self.train_history["rep"].unique()
        splits = self.train_history["split"].unique()

        if "bootstrap" in self.train_history.columns:
            bootstraps = self.train_history["bootstrap"].unique()
        else:
            bootstraps = None
        
        test_input = self.data.test

        X_test = test_input["alleles"]
        y_test = test_input["pop"]

        y_test = self.label_enc.transform(y_test)
        X_test, y_test = _data_converter(X_test, y_test)

        y_true = y_test.squeeze()
        y_true_pops = self.label_enc.inverse_transform(y_true)

        # Generate predictions for each cross validation model
        if not best_model_only and bootstraps is None:
            self.__cv_test_results = self.__test_on_multiple_models(reps, splits, X_test, y_true_pops, 
                                                                    self.__cv_output_folder)

        elif not best_model_only:
            # TODO: multiprocess this
            self.__bootstrap_test_results = pd.DataFrame()
            for bootstrap in bootstraps:
                bootstrap_folder = os.path.join(
                    self.output_folder, "bootstrap_results", f"bootstrap_{bootstrap}")
                boot_result = self.__test_on_multiple_models(reps, splits, X_test, y_true_pops, bootstrap_folder)
                self.__bootstrap_test_results = pd.concat([self.__bootstrap_test_results,
                                                    boot_result])

        # Predict using the best model and revert from label encoder
        y_pred = self.best_model(X_test).argmax(axis=1)
        y_pred_pops = self.label_enc.inverse_transform(y_pred)

        self.__test_results = pd.DataFrame({"true_pop": y_true_pops,
                                            "pred_pop": y_pred_pops})

        if save:
            self.test_results.to_csv(os.path.join(self.output_folder,
                                     "classifier_test_results.csv"), index=False)

        self.__calculate_performance(y_true, y_pred, y_true_pops, best_model_only, bootstraps)


    def __test_on_multiple_models(self, reps, splits, X_test, y_true_pops, folder):

        result_df = pd.DataFrame()
        for rep in reps:
            for split in splits:
                model = torch.load(os.path.join(
                    folder, f"best_model_split{split}_rep{rep}.pt"))
                y_pred = model(X_test).argmax(axis=1)
                y_pred_pops = self.label_enc.inverse_transform(y_pred)
                cv_test_results_temp = pd.DataFrame(
                    {"rep": rep, "split": split, 
                     "true_pop": y_true_pops, "pred_pop": y_pred_pops})
                result_df = pd.concat([result_df, cv_test_results_temp])

        return result_df
                                    

    def __calculate_performance(self, y_true, y_pred, y_true_pops, best_model_only, bootstraps):

        # TODO: Make this DRY
        # Calculate performance metrics for best model                     
        self.__confusion_matrix = np.round(
            confusion_matrix(self.test_results["true_pop"],
                             self.test_results["pred_pop"], 
                             labels=np.unique(y_true_pops).tolist(),
                             normalize="true"), 3)
        self.__accuracy = np.round(accuracy_score(y_true, y_pred), 3)
        self.__precision = np.round(precision_score(y_true, y_pred, average="weighted"), 3)
        self.__recall = np.round(recall_score(y_true, y_pred, average="weighted"), 3)
        self.__f1 = np.round(f1_score(y_true, y_pred, average="weighted"), 3)

        # Calculate ensemble performance metrics if not best model only
        if not best_model_only and bootstraps is None:
            y_pred_cv = self.label_enc.transform(self.cv_test_results["pred_pop"])
            y_true_cv = self.label_enc.transform(self.cv_test_results["true_pop"])
            # TODO: put all cross validation results in single attribute?
            self.__cv_confusion_matrix = np.round(
                confusion_matrix(self.cv_test_results["true_pop"],
                                 self.cv_test_results["pred_pop"], 
                                 labels=np.unique(y_true_pops).tolist(),
                                 normalize="true"), 3)
            self.__cv_accuracy = np.round(accuracy_score(y_true_cv, y_pred_cv), 3)
            self.__cv_precision = np.round(precision_score(y_true_cv, y_pred_cv, average="weighted"), 3)
            self.__cv_recall = np.round(recall_score(y_true_cv, y_pred_cv, average="weighted"), 3)
            self.__cv_f1 = np.round(f1_score(y_true_cv, y_pred_cv, average="weighted"), 3)

        elif not best_model_only:
            y_pred_bs = self.label_enc.transform(self.__bootstrap_test_results["pred_pop"])
            y_true_bs = self.label_enc.transform(self.__bootstrap_test_results["true_pop"])
            # TODO: put all bootstrap results in single attribute?
            self.__bs_confusion_matrix = np.round(
                confusion_matrix(self.__bootstrap_test_results["true_pop"],
                                 self.__bootstrap_test_results["pred_pop"], 
                                 labels=np.unique(y_true_pops).tolist(),
                                 normalize="true"), 3)
            self.__bs_accuracy = np.round(accuracy_score(y_true_bs, y_pred_bs), 3)
            self.__bs_precision = np.round(precision_score(y_true_bs, y_pred_bs, average="weighted"), 3)
            self.__bs_recall = np.round(recall_score(y_true_bs, y_pred_bs, average="weighted"), 3)
            self.__bs_f1 = np.round(f1_score(y_true_bs, y_pred_bs, average="weighted"), 3)


    def assign_unknown(self, best_model_only=True, save=True):
        """
        Assigns unknown samples to populations using the trained neural network.

        Parameters
        ----------
        best_model_only : bool, optional
            Whether to only assign samples to populations using the best model 
            (lowest validation loss during training). If set to False, then will also use all
            models generated from all training repeats and cross-validation splits to
            identify the most commonly assigned population and the frequency of assignment
            to this population. The default is True.
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

        if "bootstrap" in self.train_history.columns:
            bootstraps = self.train_history["bootstrap"].unique()
        else:
            bootstraps = None

        if not best_model_only and bootstraps is None:
            self.__pred_array = self.__assign_on_multiple_models(
                X_unknown, self.__cv_output_folder)
            
            unknown_data = self.__get_most_common_preds(unknown_data)

        elif not best_model_only:
            reps = self.train_history["rep"].unique()
            splits = self.train_history["split"].unique()
            array_width_total = len(bootstraps) * splits.max() * reps.max()
            self.__pred_array = np.zeros(shape=(len(X_unknown), array_width_total))

            for bootstrap in bootstraps:
                bootstrap_folder = os.path.join(
                    self.output_folder, "bootstrap_results", f"bootstrap_{bootstrap}")
                array_width_bootstrap = splits.max() * reps.max()
                array_end_position = bootstrap * array_width_bootstrap
                array_start_position = array_end_position - array_width_bootstrap
                new_array = self.__assign_on_multiple_models(X_unknown, bootstrap_folder)
                self.__pred_array[:, array_start_position:array_end_position] = new_array

            unknown_data = self.__get_most_common_preds(unknown_data)

        self.__classification = unknown_data

        if save:
            unknown_data.to_csv(os.path.join(self.output_folder,
                                "classifier_assignment_results.csv"),
                                index=False)
        
        return unknown_data
    
    def __assign_on_multiple_models(self, X_unknown, folder):
        reps = self.train_history["rep"].unique()
        splits = self.train_history["split"].unique()

        # Create empty array to fill
        array_width_total = splits.max() * reps.max()
        array = np.zeros(shape=(len(X_unknown), array_width_total))
        pos = 0

        for rep in reps:
            for split in splits:
                model = torch.load(os.path.join(
                    folder, f"best_model_split{split}_rep{rep}.pt"))
                preds = model(X_unknown).argmax(axis=1)
                array[:, pos] = preds
                pos += 1

        return array

    def __get_most_common_preds(self, unknown_data):
        """
        Want to retrieve the most common prediction across all reps / splits
        for each unknown sample - give estimate of confidence based on how
        many times a sample is assigned to a population
        """
        most_common = np.array([Counter(sorted(row, reverse=True)).\
                                most_common(1)[0][0] for row in self.__pred_array])
        most_common_count = np.count_nonzero(self.__pred_array == most_common[:, None], axis=1)
        frequency = np.round(most_common_count / self.__pred_array.shape[1], 3)
        most_common = self.label_enc.inverse_transform(most_common.astype(int))
        unknown_data.loc[:, "most_assigned_pop"] = most_common    
        unknown_data.loc[:, "frequency"] = frequency

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
            num_mismatches = [i for i, j in zip(preds, Y_enc.argmax(axis=1)) if i != j]
            errors.append(np.round(len(num_mismatches) / len(Y), 3))

        max_error = np.max(errors)

        if max_error == 0:
            importance = [1 for e in errors]
        else:
            importance = [1 - (1 - np.round(e / max_error, 3)) for e in errors]

        importance_data = {"snp": snp_names, "error": errors,
                           "importance": importance}
        ranking = pd.DataFrame(importance_data).sort_values("importance",
                                                            ascending=False)
        ranking.reset_index(drop=True, inplace=True)

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

        _plot_training_curve(self.train_history, self.__nn_type,
            self.output_folder, save)

    def plot_confusion_matrix(self, best_model_only=True, save=True):
        """
        Plots the confusion matrix.
        
        Parameters
        ----------
        best_model_only : bool, optional
            Whether to create the confusion matrix from results from running the
            best modely only or from results from running models for all splits
            and reps. The default is True.
        save : bool, optional
            Whether to save the plot to a png file. The default is True.
        
        Returns
        -------
        None
        """
        bootstraps = "bootstrap" in self.train_history.columns

        if best_model_only:
            _plot_confusion_matrix(self.test_results, self.confusion_matrix,
                self.nn_type, self.output_folder, save)
        elif bootstraps:
            _plot_confusion_matrix(self.__bootstrap_test_results, self.__bs_confusion_matrix,
                self.nn_type, self.__bootstrap_results, save)
        else:
            _plot_confusion_matrix(self.cv_test_results, self.cv_confusion_matrix,
                self.nn_type, self.__cv_output_folder, save)

    def plot_assignment(self, best_model_only=True, save=True, col_scheme="Spectral"):
        """
        Plots the proportion of times each individual from the
        unknown data was assigned to each population.

        Parameters
        ----------
        best_model_only : bool, optional
            Whether to create the assignment plot from results from running the
            best modely only or from results from running models for all splits
            and reps. The default is True.
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

        if best_model_only:
            e_preds = self.classification.copy()
            folder = self.output_folder

        else:
            pred_df = pd.DataFrame(self.__pred_array)
            for col in pred_df.columns:
                pred_df[col] = self.__label_enc.inverse_transform(pred_df[col].astype(int))

            classifications = self.classification.copy()
            classifications.reset_index(inplace=True)
            classifications = classifications[["id"]]
            classifications = pd.concat([classifications, pred_df], axis=1)

            e_preds = pd.melt(classifications, id_vars=["id"], 
                    value_vars=pred_df.columns, 
                    value_name="assigned_pop")
            e_preds.rename(columns={"id": "sampleID"}, inplace=True)

            if "bootstrap" in self.train_history.columns:
                folder = self.__bootstrap_results
            else:
                folder = self.__cv_output_folder

        _plot_assignment(e_preds, col_scheme, folder, self.__nn_type, save, best_model_only)

    def plot_structure(self, best_model_only=True, save=True, col_scheme="Spectral"):
        """
        Plots the proportion of times individuals from the
        test data were assigned to the correct population. 
        Used for determining the accuracy of the classifier.

        Parameters
        ----------
        best_model_only : bool, optional
            Whether to create the structure plot from results from running the
            best modely only or from results from running models for all splits
            and reps. The default is True.
        save : bool, optional
            Whether to save the plot to a png file. The default is True.
        col_scheme : str, optional
            The colour scheme to use for the plot. The default is "Spectral".
        
        Returns
        -------
        None
        """
        bootstraps = "bootstrap" in self.train_history.columns

        if best_model_only:
            preds = pd.DataFrame(self.confusion_matrix,
                                columns=self.label_enc.classes_,
                                index=self.label_enc.classes_)
            folder = self.output_folder
        elif bootstraps:
            preds = pd.DataFrame(self.__bs_confusion_matrix,
                                columns=self.label_enc.classes_,
                                index=self.label_enc.classes_)
            folder = self.__bootstrap_results
        else:
            preds = pd.DataFrame(self.cv_confusion_matrix,
                                columns=self.label_enc.classes_,
                                index=self.label_enc.classes_)
            folder = self.__cv_output_folder

        _plot_structure(preds, col_scheme, self.__nn_type, folder, save)

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

    @staticmethod
    def load(load_path=None):
        """
        Loads a saved instance of the class from a pickle file.

        Parameters
        ----------
        load_path : str, optional
            The path to load the file from. The default is None.
        
        Returns
        -------
        None
        """
        return _load(load_path)

    def _validate_init_inputs(self, data, random_state, output_folder):

        if not isinstance(data, GeneticData):
            raise TypeError("data must be an instance of GeneticData")

        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer")

        if output_folder is not None:
            if not isinstance(output_folder, str):
                raise TypeError("output_folder must be a string")

            if not os.path.isdir(output_folder):
                raise ValueError("output_folder must be a valid directory")

    def _validate_train_inputs(self, epochs, valid_size, cv_splits, cv_reps,
                               learning_rate, batch_size, dropout_prop):

        if not isinstance(epochs, int):
            raise TypeError("epochs must be an integer")
        
        if not isinstance(valid_size, float):
            raise TypeError("valid_size must be a float")

        if valid_size > 1 or valid_size < 0:
            raise ValueError("valid_size must be between 0 and 1")
        
        if not isinstance(cv_splits, int):
            raise TypeError("cv_splits must be an integer")

        if not isinstance(cv_reps, int):
            raise TypeError("cv_reps must be an integer")

        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be a float")

        if learning_rate > 1 or learning_rate < 0:
            raise ValueError("learning_rate must be between 0 and 1")

        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")

        if not isinstance(dropout_prop, float) and not isinstance(dropout_prop, int):
            raise TypeError("dropout_prop must be a float")

        if dropout_prop > 1 or dropout_prop < 0:
            raise ValueError("dropout_prop must be between 0 and 1")


