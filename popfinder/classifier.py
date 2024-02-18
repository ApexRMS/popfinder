import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import numpy as np
import pandas as pd
import os
import shutil
from subprocess import call
import tempfile

import popfinder as pf
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
            output_folder = os.join.path(os.getdcwd(), "popfinder_results")
        self.__output_folder = output_folder
        self.__cv_output_folder = os.path.join(output_folder, "cv_results")
        self.__label_enc = data.label_enc
        self.__train_history = None
        self.__best_model = None
        self.__test_results = None # use for cm and structure plot
        self.__classification = None # use for assignment plot
        self.__accuracy = None
        self.__precision = None
        self.__recall = None
        self.__f1 = None
        self.__confusion_matrix = None
        self.__nn_type = "classifier"
        self.__mp_run = False
        self.__lowest_val_loss_total = 9999

    @property
    def data(self):
        return self.__data

    @property
    def random_state(self):
        return self.__random_state
    
    @property
    def output_folder(self):
        return self.__output_folder

    @output_folder.setter
    def output_folder(self, output_folder):
        self.__output_folder = output_folder
        self.__cv_output_folder = os.path.join(output_folder, "cv_results")

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

    def train(self, epochs=100, valid_size=0.2, cv_splits=1, nreps=1,
              learning_rate=0.001, batch_size=16, dropout_prop=0, bootstraps=None,
              jobs=1, patience=None, min_delta=0, overwrite_results=True):
        """
        Trains the classification neural network.

        Parameters
        ----------
        epochs : int, optional
            Number of epochs to train the neural network. The default is 100.
        valid_size : float, optional
            Proportion of data to use for validation. The default is 0.2.
        cv_splits : int, optional
            Number of cross-validation splits. If set to 1, then no cross-
            validation is applied. The default is 1.
        nreps : int, optional
            Number of repetitions. The default is 1.
        learning_rate : float, optional
            Learning rate for the neural network. The default is 0.001.
        batch_size : int, optional
            Batch size for the neural network. The default is 16.
        dropout_prop : float, optional
            Dropout proportion for the neural network. The default is 0.
        bootstraps : int, optional
            Number of bootstraps to perform. The default is None.
        jobs : int, optional
            If greater than 1, will use multiprocessing to train the neural network. 
            The default is 1.
        patience : int, optional
            Number of epochs to wait before stopping training if validation loss   
            does not decrease. The default is None.
        min_delta : float, optional
            Minimum change in validation loss to be considered an improvement.
            The default is 0.
        overwrite_results : boolean, optional
            If True, then will clear the output folder before training the new 
            model. The default is True.
        
        Returns
        -------
        None.
        """
        self._validate_train_inputs(epochs, valid_size, cv_splits, nreps,
                                    learning_rate, batch_size, dropout_prop)
        
        self.__prepare_result_folder(self.output_folder, overwrite_results)

        files = os.listdir(self.output_folder)
        if (overwrite_results) or (len(files) == 0) or (self.train_history is None):
            nrep_begin = 0
            self.__lowest_val_loss_total = 9999 # reset lowest val loss
            self.__train_history = None # reset train history
        else:
            existing_reps = [int(f.split("_")[-2].replace("rep", "")) for f in files if "rep" in f]
            nrep_begin = max(existing_reps)
            nreps = nrep_begin + nreps 

        multi_output = (bootstraps is not None) or (nreps is not None)

        if multi_output:

            if bootstraps is None:
                bootstraps = 1
            if nreps is None:
                nreps = nrep_begin + 1

            loss_df = pd.DataFrame()

            if jobs == 1:
                for i in range(bootstraps):
                    for j in range(nrep_begin, nreps):
                        #TODO: how does this affect mp results
                        if not self.__mp_run:
                            boot_folder = os.path.join(self.output_folder, f"rep{j+1}_boot{i+1}")
                            if not os.path.exists(boot_folder):
                                os.makedirs(boot_folder)
                        else:
                            boot_folder = self.output_folder

                        inputs = _generate_train_inputs(self.data, valid_size, cv_splits,
                                        nreps, seed=self.random_state, bootstrap=True)
                        boot_loss_df = self.__train_on_inputs(inputs, cv_splits, epochs, learning_rate,
                                            batch_size, dropout_prop, result_folder = boot_folder, 
                                            patience=patience, min_delta=min_delta,
                                            overwrite_results=overwrite_results)
                        
                        boot_loss_df.to_csv(os.path.join(boot_folder, "loss.csv"), index=False)
                        boot_loss_df["rep"] = j + 1
                        boot_loss_df["bootstrap"] = i + 1
                        loss_df = pd.concat([loss_df, boot_loss_df], axis=0, ignore_index=True)
            elif jobs > 1:
                # Create tempfolder
                tempfolder = tempfile.mkdtemp()

                # Let popfinder know this is a multiprocessing run (affects output folder creation)
                self.__mp_run = True
                self.save(save_path=tempfolder)

                # Find path to _mp_training
                filepath = pf.__file__
                folderpath = os.path.dirname(filepath)

                # Instead of looping through bootstrap iteration, run in parallel
                # to speed up training
                call(["python", folderpath + "/_mp_training.py", "-p", tempfolder,
                    "-n", str(bootstraps), "--r_start", str(nrep_begin), "-r", str(nreps), 
                    "-e", str(epochs), "-v", str(valid_size), "-s", str(cv_splits), 
                    "-l", str(learning_rate), "-b", str(batch_size), "-d", str(dropout_prop),
                    "-j", str(jobs)])
                loss_df = pd.read_csv(os.path.join(tempfolder, "train_history.csv"))

        # Save training history
        if self.__train_history is None:
            self.__train_history = loss_df
        else:
            self.__train_history = pd.concat([self.__train_history, loss_df], ignore_index=True)
       
       # Determine best model
        if (jobs == 1) or (not multi_output):
            self.__best_model = torch.load(os.path.join(self.output_folder, "best_model.pt"))
        else:
            best_model_folder, min_split = self.__find_best_model_folder_from_mp()
            self.__best_model = torch.load(os.path.join(best_model_folder, f"best_model_split{min_split}.pt"))
            torch.save(self.__best_model, os.path.join(self.output_folder, "best_model.pt"))
            self.__clean_mp_folders(nrep_begin, nreps, bootstraps)


    def test(self, use_best_model=True, save=True):
        """
        Tests the classification neural network.

        Parameters
        ----------
        use_best_model : bool, optional
            Whether to test using the best model only. If set to False, then will use all
            models generated from all training repeats and cross-validation splits and
            provide an ensemble frequency of assignments. The default is True.
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

        # If not using just the best model, then test using all models
        if not use_best_model:
            if bootstraps is None: 
                bootstraps = 1
            if reps is None:
                reps = 1

            # Tests on all reps, bootstraps, and cv splits
            self.__test_results = self.__test_on_multiple_models(reps, bootstraps, splits, X_test, y_true_pops)
            y_pred = self.label_enc.transform(self.__test_results["pred_pop"])
            y_true = self.label_enc.transform(self.__test_results["true_pop"])
            y_true_pops = self.label_enc.inverse_transform(y_true)

        elif use_best_model:
            # Predict using the best model and revert from label encoder
            y_pred = self.best_model(X_test).argmax(axis=1)
            y_pred_pops = self.label_enc.inverse_transform(y_pred)

            self.__test_results = pd.DataFrame({"true_pop": y_true_pops,
                                                "pred_pop": y_pred_pops})

        if save:
            self.test_results.to_csv(os.path.join(self.output_folder,
                                    "classifier_test_results.csv"), index=False)

        self.__calculate_performance(y_true, y_pred, y_true_pops, use_best_model, bootstraps)

    def assign_unknown(self, use_best_model=True, save=True):
        """
        Assigns unknown samples to populations using the trained neural network.

        Parameters
        ----------
        use_best_model : bool, optional
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

        if use_best_model:
            assign_array = self.best_model(X_unknown).argmax(axis=1)
            preds = self.label_enc.inverse_transform(assign_array)
            unknown_data.loc[:, "assigned_pop"] = preds
            assign_array = assign_array.numpy().reshape(-1, 1)

        if "bootstrap" in self.train_history.columns:
            bootstraps = self.train_history["bootstrap"].unique()
        else:
            bootstraps = None

        # TODO: I don't think this chunk of code ever executes now
            # because bootstraps is 1, not None, if not set
        # if not use_best_model and bootstraps is None:
        #     assign_array = self.__assign_on_multiple_models(
        #         X_unknown, self.__cv_output_folder)
            
        #     unknown_data = self.__get_most_common_preds(unknown_data)

        if not use_best_model:
            reps = self.train_history["rep"].unique()
            splits = self.train_history["split"].unique()
            array_width_total = len(bootstraps) * splits.max() * reps.max()
            assign_array = np.zeros(shape=(len(X_unknown), array_width_total))
            array_start_position = 0

            for rep in reps:
                for boot in bootstraps:
                    bootstrap_folder = os.path.join(self.output_folder, f"rep{rep}_boot{boot}")
                    # array_width_bootstrap = splits.max() * reps.max()
                    array_width_bootstrap = splits.max() * rep * boot
                    array_end_position = boot * array_width_bootstrap
                    new_array = self.__assign_on_multiple_models(X_unknown, bootstrap_folder)
                    assign_array[:, array_start_position:array_end_position] = new_array
                    array_start_position = array_end_position


            unknown_data = self.__get_most_common_preds(unknown_data, assign_array)

        self.__classification = unknown_data
        self.__pred_array = assign_array

        # Creates a table of assignment results for each model in ensemble
        self.__mod_assign = pd.DataFrame(
            data = assign_array, 
            index = unknown_data.index, 
            columns = ["mod_" + str(i) for i in range(0, assign_array.shape[1])])
        
        # Creates a table of sample assignment frequency for each population
        # TODO: maybe doesn't work for use_best_model=True
        self.__freq_assign = self.__get_assignment_frequency(assign_array)

        if save:
            unknown_data.to_csv(os.path.join(self.output_folder,
                                "classifier_assignment_results.csv"),
                                index=False)
        
        return unknown_data

    def __get_assignment_frequency(self, assign_array):

        pred_df = pd.DataFrame(assign_array)
        for col in pred_df.columns:
            pred_df[col] = self.label_enc.inverse_transform(pred_df[col].astype(int))

        classifications = self.classification.copy()
        classifications.reset_index(inplace=True)
        classifications = classifications[["id"]]
        classifications = pd.concat([classifications, pred_df], axis=1)

        e_preds = pd.melt(classifications, id_vars=["id"], 
                value_vars=pred_df.columns, 
                value_name="assigned_pop")
        e_preds.rename(columns={"id": "sampleID"}, inplace=True)
        e_preds.set_index("sampleID", inplace=True)
        e_preds = pd.get_dummies(e_preds["assigned_pop"], dtype=float)
        e_preds = e_preds.reset_index().groupby("sampleID").mean()

        return e_preds

    def update_unknown_samples(self, new_genetic_data, new_sample_data):
        """
        Updates the unknown samples in the classifier object.

        Parameters
        ----------
        new_genetic_data : str
            Path to the new genetic data file.
        new_sample_data : str
            Path to the new sample data file.
        
        Returns
        -------
        None.
        """
        self.__data.update_unknowns(new_genetic_data, new_sample_data)

    # Reporting functions below
    def get_test_summary(self, save=True):
        """
        Get a summary of the classification performance metrics from running
        the test() function, including accuracy, precision, recall, and f1 
        score. Metrics are either based on the best classifier model
        (use_best_model set to True), or are averaged across the ensemble of 
        models if tested across all bootstraps, repetitions, and cross 
        validation splits (use_best_model set to False).

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
            "metric": ["accuracy", "precision", "recall", "f1"],
            "value": [self.accuracy, self.precision, self.recall, self.f1]
        }
        summary = pd.DataFrame(summary)

        if save:
            summary.to_csv(os.path.join(self.output_folder,
                          "classifier_classification_summary.csv"),
                           index=False)

        return summary
    
    def get_assignment_summary(self, save=True):
        """
        Get a summary of the assignment results from running the assign_unknown()
        function, including the most commonly assigned population and the
        frequency of assignment across all bootstraps, repetitions, and cross
        validation splits.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the results to a csv file. The default is True.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing the assignment summary.
        """
        if self.classification is None:
            raise ValueError("No classification results to summarize. " + 
            "Please run the assign_unknown() method first.")

        summary = self.__freq_assign

        if save:
            summary.to_csv(os.path.join(self.output_folder,
                          "classifier_assignment_summary.csv"),
                           index=False)

        return summary
    
    def get_confusion_matrix(self):
        """
        Get the confusion matrix for the classification results.

        Returns
        -------
        numpy.ndarray
            Confusion matrix based on the results of running the test() function.
        """           
        return self.confusion_matrix

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
                          "rank_site_importance.csv"),
                           index=False)

        return ranking

    # Plotting functions below
    def plot_training_curve(self, save=True, facet_by_split_rep=False, 
                            y_axis_zero=False):
        """
        Plots the training curve.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the plot to a png file. The default is True.
        facet_by_split_rep : bool, optional
            Whether to facet the plot by split and rep. If False and more than
            1 split and rep have been used during training, then the training
            plot will contain variability corresponding to the multiple runs.
            The default is False.
        y_axis_zero : bool, optional
            Whether to set the y-axis to start at 0. The default is False.
            
        Returns
        -------
        None
        """

        _plot_training_curve(self.train_history, self.__nn_type,
            self.output_folder, save, facet_by_split_rep, y_axis_zero)

    def plot_confusion_matrix(self, save=True):
        """
        Plots the confusion matrix based on the results from running the test() 
        function.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the plot to a png file. The default is True.
        
        Returns
        -------
        None
        """
        _plot_confusion_matrix(self.test_results, self.confusion_matrix,
            self.nn_type, self.output_folder, save)

    def plot_assignment(self, save=True, col_scheme="Spectral"):
        """
        Plots the results from running the assign_unknown() function. If the 
        assign_unknown() function is run with use_best_model set to False, then plots 
        the proportion of times each sample from the unknown data was assigned to each 
        population across all bootstraps, repetitions, and cross validation splits.
        If the assign_unknown() function is run with use_best_model set to True, 
        only plots the assignment based on the results from running the data through 
        the best classifier model (all assignment frequencies will be 1).

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

        # if len(np.unique(self.classification.index)) == len(self.classification):
        # if self.__pred_array.shape[1] == 1:
        #     e_preds = self.classification.copy()
        #     use_best_model = True

        # else:
        e_preds = self.__freq_assign
        use_best_model = False

        _plot_assignment(e_preds, col_scheme, self.output_folder, self.__nn_type, save, use_best_model)

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
        folder = self.output_folder

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

    def _validate_train_inputs(self, epochs, valid_size, cv_splits, nreps,
                               learning_rate, batch_size, dropout_prop):

        if not isinstance(epochs, int):
            raise TypeError("epochs must be an integer")
        
        if not isinstance(valid_size, float):
            raise TypeError("valid_size must be a float")

        if valid_size > 1 or valid_size < 0:
            raise ValueError("valid_size must be between 0 and 1")
        
        if not isinstance(cv_splits, int):
            raise TypeError("cv_splits must be an integer")
        
        if cv_splits < 1:
            raise ValueError("cv_splits must be greater than 0")

        if not isinstance(nreps, int):
            raise TypeError("nreps must be an integer")

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

    # Hidden functions below   
    def __train_on_inputs(self, inputs, cv_splits, epochs, learning_rate, batch_size, 
                          dropout_prop, result_folder, patience, min_delta,
                          overwrite_results):

        self.__prepare_result_folder(result_folder, overwrite_results)

        loss_dict = {"split": [], "epoch": [], "train": [], "valid": []}

        for i, input in enumerate(inputs):

            lowest_val_loss_rep = 9999
            split = i % cv_splits + 1

            X_train, y_train, X_valid, y_valid = _split_input_classifier(self, input)
            train_loader, valid_loader = _generate_data_loaders(X_train, y_train,
                                                                X_valid, y_valid)

            net = ClassifierNet(input_size=X_train.shape[1], hidden_size=16, #TODO: make hidden size a parameter
                                output_size=len(y_train.unique()),
                                batch_size=batch_size, dropout_prop=dropout_prop)
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
            loss_func = nn.CrossEntropyLoss()
            patience_counter = 0

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
                        torch.save(net, os.path.join(result_folder, f"best_model_split{split}.pt"))

                    elif ((valid_loss - lowest_val_loss_rep) > min_delta) & (patience is not None): 
                        patience_counter += 1
                        if patience_counter > patience:
                            break

                    if valid_loss < self.__lowest_val_loss_total:
                        self.__lowest_val_loss_total = valid_loss
                        torch.save(net, os.path.join(self.output_folder, "best_model.pt"))

                # Calculate average validation loss
                avg_valid_loss = valid_loss / len(valid_loader)

                loss_dict["split"].append(split)
                loss_dict["epoch"].append(epoch)
                loss_dict["train"].append(avg_train_loss)
                loss_dict["valid"].append(avg_valid_loss)

        return pd.DataFrame(loss_dict)
    
    def __prepare_result_folder(self, result_folder, overwrite_results=True):

        # Make result folder if it doesn't exist
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        elif overwrite_results:
            shutil.rmtree(result_folder)
            os.mkdir(result_folder)

    def __test_on_multiple_models(self, reps, bootstraps, splits, X_test, y_true_pops):

        result_df = pd.DataFrame()
        for rep in reps:
            for boot in bootstraps:
                for split in splits:
                    folder = os.path.join(self.output_folder, f"rep{rep}_boot{boot}")
                    model = torch.load(os.path.join(folder, f"best_model_split{split}.pt"))
                    y_pred = model(X_test).argmax(axis=1)
                    y_pred_pops = self.label_enc.inverse_transform(y_pred)
                    cv_test_results_temp = pd.DataFrame(
                        {"rep": rep, "bootstrap": boot, "split": split, 
                        "true_pop": y_true_pops, "pred_pop": y_pred_pops})
                    result_df = pd.concat([result_df, cv_test_results_temp])

        return result_df
                                    

    def __calculate_performance(self, y_true, y_pred, y_true_pops, use_best_model, bootstraps):

        # Calculate ensemble performance metrics if not best model only
        if not use_best_model and bootstraps is None:
            self.__test_on_cv_splits = True

        elif not use_best_model:
            self.__test_on_bootstraps = True

        results = self.__organize_performance_metrics(self.test_results, y_true_pops, y_true, y_pred)
        self.__confusion_matrix, self.__accuracy, self.__precision, self.__recall, self.__f1 = results

    def __organize_performance_metrics(self, result_df, y_true_pops, y_true, y_pred):
        cf = np.round(confusion_matrix(
            result_df["true_pop"], result_df["pred_pop"], 
            labels=np.unique(y_true_pops).tolist(), normalize="true"), 3)
        accuracy = np.round(accuracy_score(y_true, y_pred), 3)
        precision = np.round(precision_score(y_true, y_pred, average="weighted"), 3)
        recall = np.round(recall_score(y_true, y_pred, average="weighted"), 3)
        f1 = np.round(f1_score(y_true, y_pred, average="weighted"), 3)

        return cf, accuracy, precision, recall, f1

    def __assign_on_multiple_models(self, X_unknown, folder):
        # reps = self.train_history["rep"].unique()
        splits = self.train_history["split"].unique()

        # Create empty array to fill
        array_width_total = splits.max() # * reps.max()
        array = np.zeros(shape=(len(X_unknown), array_width_total))
        pos = 0

        #for rep in reps:
        for split in splits:
            model = torch.load(os.path.join(folder, f"best_model_split{split}.pt"))
            preds = model(X_unknown).argmax(axis=1)
            array[:, pos] = preds
            pos += 1

        return array

    def __get_most_common_preds(self, unknown_data, assign_array):
        """
        Want to retrieve the most common prediction across all reps / splits
        for each unknown sample - give estimate of confidence based on how
        many times a sample is assigned to a population
        """
        most_common = np.array([Counter(sorted(row, reverse=True)).\
                                most_common(1)[0][0] for row in assign_array])
        most_common_count = np.count_nonzero(assign_array == most_common[:, None], axis=1)
        frequency = np.round(most_common_count / assign_array.shape[1], 3)
        most_common = self.label_enc.inverse_transform(most_common.astype(int))
        unknown_data.loc[:, "most_assigned_pop_across_models"] = most_common    
        unknown_data.loc[:, "frequency_of_assignment_across_models"] = frequency

        return unknown_data

    def __find_best_model_folder_from_mp(self):
        min_loss = self.train_history.iloc[self.train_history[["valid"]].idxmin()]
        min_rep = min_loss["rep"].values[0]
        min_boot = min_loss["bootstrap"].values[0]
        min_split = min_loss["split"].values[0]
        best_model_folder = os.path.join(self.output_folder, f"rep{min_rep}_boot{min_boot}")
        return best_model_folder, min_split
    
    def __clean_mp_folders(self, nrep_begin, nreps, bootstraps):
        for rep in range(nrep_begin, nreps):
            for boot in range(bootstraps):
                folder = os.path.join(self.output_folder, f"rep{rep+1}_boot{boot+1}")
                os.remove(os.path.join(folder, "best_model.pt"))

