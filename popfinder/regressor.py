import torch
from torch.autograd import Variable
from scipy import spatial
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
import os

from popfinder.dataloader import GeneticData
from popfinder._neural_networks import RegressorNet
from popfinder._helper import _generate_train_inputs
from popfinder._helper import _generate_data_loaders
from popfinder._helper import _data_converter
from popfinder._helper import _split_input_regressor
from popfinder._helper import _save, _load
from popfinder._visualize import _plot_assignment
from popfinder._visualize import _plot_training_curve
from popfinder._visualize import _plot_confusion_matrix
from popfinder._visualize import _plot_structure

class PopRegressor(object):
    """
    A class to represent a regressor neural network object for population assignment.

    Parameters
    ----------
    data : GeneticData object
        GeneticData object containing the training data.
    nboots : int, optional
        Number of bootstrap samples to generate. The default is 20.
    random_state : int, optional
        Random seed for reproducibility. The default is 123.
    output_folder : str, optional
        Path to output folder. The default is None.
    
    Attributes
    ----------
    data : GeneticData object
        GeneticData object containing the training data.
    nboots : int
        Number of bootstrap samples to generate.
    random_state : int
        Random seed for reproducibility.
    output_folder : str
        Path to output folder.
    train_history : list
        List of training history objects.
    best_model : torch.nn.Module
        Best model from training.
    regression : pandas.DataFrame
        Dataframe containing the regression results.
    median_distance : float
        Median distance between true and predicted coordinates.
    mean_distance : float
        Mean distance between true and predicted coordinates.
    r2_lat : float
        R-squared value for latitude.
    r2_long : float
        R-squared value for longitude.
    summary : pandas.DataFrame
        Dataframe containing the summary of the regression results.
    contour_classification : pandas.DataFrame
        Dataframe containing the contour classification results.
    classification_test_results : pandas.DataFrame
        Dataframe containing the classification test results.
    classification_accuracy : float
        Classification accuracy.
    classification_precision : float
        Classification precision.
    classification_recall : float
        Classification recall.
    classification_f1 : float
        Classification F1 score.
    classification_confusion_matrix : numpy.ndarray
        Classification confusion matrix.

    Methods
    -------
    train(epochs=100, valid_size=0.2, cv_splits=1, cv_reps=1, learning_rate=0.001, batch_size=16, dropout_prop=0, boot_data=None)
        Train the regressor.
    test()
        Test the regressor.
    assign_unknown()
        Assign unknown samples.
    rank_site_importance()
        Rank the importance of each site.
    plot_training_curve()
        Plot the training curve.
    plot_location()
        Plot the predicted and true locations.  
    plot_contour_map()
        Plot the contour map.
    plot_confusion_matrix()
        Plot the confusion matrix.
    plot_structure()
        Plot the neural network structure.  
    save()
        Save the regressor object.
    load()
        Load the regressor object.  
    """
    def __init__(self, data, nboots=20, random_state=123, output_folder=None):

        self._validate_init_inputs(data, nboots, random_state, output_folder)

        self.__data = data # GeneticData object
        self.__nboots = nboots
        self.__random_state = random_state
        if output_folder is None:
            output_folder = os.getcwd()
        self.__output_folder = output_folder
        self.__train_history = None
        self.__best_model = None
        self.__test_results = None
        self.__regression = None
        self.__median_distance = None
        self.__mean_distance = None
        self.__r2_lat = None
        self.__r2_long = None
        self.__summary = None
        self.__contour_classification = None
        self.__classification_test_results = None
        self.__classification_accuracy = None
        self.__classification_precision = None
        self.__classification_recall = None
        self.__classification_f1 = None
        self.__classification_confusion_matrix = None
        self.__nn_type = "regressor"

    @property
    def data(self):
        return self.__data

    @property
    def nboots(self):
        return self.__nboots

    @property
    def random_state(self):
        return self.__random_state

    @property
    def output_folder(self):
        return self.__output_folder

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
    def regression(self):
        return self.__regression

    @property
    def median_distance(self):
        return self.__median_distance

    @property
    def mean_distance(self):
        return self.__mean_distance

    @property
    def r2_lat(self):
        return self.__r2_lat

    @property
    def r2_long(self):
        return self.__r2_long

    @property
    def summary(self):
        return self.__summary

    @property
    def contour_classification(self):
        return self.__contour_classification

    @property
    def classification_test_results(self):
        return self.__classification_test_results

    @property
    def classification_accuracy(self):
        return self.__classification_accuracy

    @property
    def classification_precision(self): 
        return self.__classification_precision

    @property
    def classification_recall(self):
        return self.__classification_recall

    @property
    def classification_f1(self):
        return self.__classification_f1

    @property
    def classification_confusion_matrix(self):
        return self.__classification_confusion_matrix

    @property
    def nn_type(self):
        return self.__nn_type

    def train(self, epochs=100, valid_size=0.2, cv_splits=1, cv_reps=1,
              learning_rate=0.001, batch_size=16, dropout_prop=0, boot_data=None):
        """
        Trains the regression neural network to estimate xy coordinates of 
        a sample's origin.
        
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
        boot_data : GeneticData, optional
            GeneticData object to use for bootstrapping. The default is None.
            
        Returns
        -------
        None.
        """
        self._validate_train_inputs(epochs, valid_size, cv_splits, cv_reps,
                            learning_rate, batch_size, dropout_prop, boot_data)

        if boot_data is None:
            inputs = _generate_train_inputs(self.data, valid_size, cv_splits,
                                            cv_reps, seed=self.random_state)
        else:
            inputs = _generate_train_inputs(boot_data, valid_size, cv_splits,
                                            cv_reps, seed=self.random_state)

        loss_df_final = pd.DataFrame({"rep": [], "split": [], "epoch": [],
                                      "train": [], "valid": []})
        self.lowest_val_loss = 9999

        for i, input in enumerate(inputs):

            X_train, y_train, X_valid, y_valid = _split_input_regressor(input)
            net = RegressorNet(input_size=X_train.shape[1], hidden_size=16,
                               batch_size=batch_size, dropout_prop=dropout_prop)
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
            loss_func = self._euclidean_dist_loss

            train_loader, valid_loader = _generate_data_loaders(X_train, y_train,
                                                                X_valid, y_valid)

            loss_df = self._fit_regressor_model(epochs, train_loader,
                    valid_loader, net, optimizer, loss_func)

            split = i % cv_splits + 1
            rep = int(i / cv_splits) + 1

            loss_df["rep"] = rep
            loss_df["split"] = split

        loss_df_final = pd.concat([loss_df_final, loss_df])

        self.__train_history = loss_df_final
        self.__best_model = torch.load(os.path.join(self.output_folder, "best_model.pt"))

    def test(self, boot_data=None, verbose=False):
        """
        Tests the regression neural network on the test data.
        
        Parameters
        ----------
        boot_data : GeneticData, optional
            GeneticData object to use for bootstrapping. The default is None.
        verbose : bool, optional
            Whether to print the test results. The default is False.

        Returns
        -------
        None.
        """
        
        if boot_data is None:
            test_input = self.data.test
        else:
            test_input = boot_data.test

        X_test = test_input["alleles"]    
        y_test = test_input[["x", "y"]]
        X_test, y_test = _data_converter(X_test, y_test)
        y_test = self._unnormalize_locations(y_test)

        y_pred = self.best_model(X_test).detach().numpy()
        y_pred = self._unnormalize_locations(y_pred)

        test_input = test_input.assign(x_pred=y_pred[:, 0], y_pred=y_pred[:, 1])

        dists = [
            spatial.distance.euclidean(
                y_pred[x, :], y_test[x, :]
            ) for x in range(len(y_pred))
        ]

        # true coordinates vs predicted coordinates
        self.__test_results = test_input[["x", "y", "x_pred", "y_pred"]]

        self.__median_distance = np.median(dists)
        self.__mean_distance = np.mean(dists)
        self.__r2_long = np.corrcoef(y_pred[:, 0], y_test[:, 0])[0][1] ** 2
        self.__r2_lat = np.corrcoef(y_pred[:, 1], y_test[:, 1])[0][1] ** 2

        self.__summary = self.get_assignment_summary()

        if verbose:
            print(self.summary)


    def assign_unknown(self, save=True, boot_data=None):
        """
        Assigns unknown samples to their predicted origin.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the results to a csv file. The default is True.
        boot_data : GeneticData, optional
            GeneticData object to use for bootstrapping. The default is None.
        
        Returns
        -------
        None.
        """
        
        if boot_data is None:
            unknown_data = self.data.unknowns
        else:
            unknown_data = boot_data.unknowns

        X_unknown = unknown_data["alleles"]
        X_unknown = _data_converter(X_unknown, None)

        y_pred = self.best_model(X_unknown).detach().numpy()
        y_pred = self._unnormalize_locations(y_pred)
        unknown_data.loc[:, "x_pred"] = y_pred[:, 0]
        unknown_data.loc[:, "y_pred"] = y_pred[:, 1]

        self.__regression = unknown_data

        if save:
            unknown_data.to_csv(os.path.join(self.output_folder,
                                "regressor_assignment_results.csv"))

        return unknown_data

    def classify_by_contours(self, nboots=5, num_contours=5,
        save_plots=True, save=True):
        """
        Classifies unknown samples by kernel density estimates (contours).
        This function uses 2D kernel density estimation to create contour 
        plots from many predictions on the same sample, then assigns the sample
        to the contour with the highest probability.

        Parameters
        ----------
        nboots : int, optional
            Number of bootstraps to perform. The default is 5.
        num_contours : int, optional
            Number of contours to generate. The default is 5.
        save_plots : bool, optional
            Whether to save the contour plots. The default is True.
        save : bool, optional
            Whether to save the classification results. The default is True.
        
        Returns
        -------
        None.
        """
        self._validate_contour_inputs(nboots, num_contours, save_plots, save)

        self._generate_bootstraps(nboots=nboots) # also add option for multiple reps

        test_locs = self.test_locs_final
        pred_locs = self.pred_locs_final

        self.__classification_test_results = self._test_classification(test_locs,
            num_contours, save_plots)
        self.__contour_classification = self._classify_unknowns(pred_locs,
            test_locs, num_contours, save_plots)

        if save:
            self.classification_test_results.to_csv(os.path.join(self.output_folder,
                "contour_classification_test_report.csv"), index=False)
            self.contour_classification.to_csv(os.path.join(self.output_folder,
                "contour_classification_results.csv"), index=False)

        # Generate classification summary stats from test_report
        y_pred = self.classification_test_results["pred_pop"]
        y_true = self.classification_test_results["true_pop"]
        self.__classification_confusion_matrix = np.round(
            confusion_matrix(y_true, y_pred, normalize="true"), 3)
        self.__classification_accuracy = accuracy_score(y_true, y_pred)
        self.__classification_precision = precision_score(y_true, y_pred, average="weighted")
        self.__classification_recall = recall_score(y_true, y_pred, average="weighted")
        self.__classification_f1 = f1_score(y_true, y_pred, average="weighted")

    # Reporting functions below
    def get_assignment_summary(self, save=True):
        """
        Returns a summary of the assignment results.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the summary to a csv file. The default is True.
            
        Returns
        -------
        dict
            Dictionary containing the summary statistics median distance,
            mean distance, r2_long, r2_lat.
        """
        summary = {
            "median_distance": [self.median_distance],
            "mean_distance": [self.mean_distance],
            "r2_long": [self.r2_long],
            "r2_lat": [self.r2_lat]
        }

        if save:
            pd.DataFrame(summary).to_csv(os.path.join(self.output_folder,
                "regressor_assignment_summary.csv"), index=False)

        return summary

    def get_classification_summary(self, save=True):
        """
        Returns a summary of the classification results.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the summary to a csv file. The default is True.
        
        Returns
        -------
        dict
            Dictionary containing the summary statistics accuracy, precision,
            recall, f1, and confusion matrix.
        """
        if self.classification_test_results is None:
            raise ValueError("Must run classify_by_contours() before getting summary.")

        summary = { # need to grab all these items
            "accuracy": [self.classification_accuracy],
            "precision": [self.classification_precision],
            "recall": [self.classification_recall],
            "f1": [self.classification_f1],
            "confusion_matrix": [self.classification_confusion_matrix]
        }
        
        if save:
            pd.DataFrame(summary).to_csv(os.path.join(self.output_folder,
                "regressor_classification_summary.csv"), index=False)

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
            Dataframe containing the ranked SNPs and their importance scores.
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. " + 
            "Please run the train() method first.")

        X = self.data.knowns["alleles"].to_numpy()
        X = np.stack(X)
        Y = self.data.knowns[["x", "y"]].to_numpy()
        snp_names = np.arange(1, X.shape[1] + 1)
        errors = []

        for i in range(X.shape[1]):
            X_temp = X.copy()
            X_temp[:, i] = np.random.choice(X_temp[:, i], X_temp.shape[0])
            X_temp = torch.from_numpy(X_temp).float()
            preds = self.best_model(X_temp).detach().numpy()
            preds = self._unnormalize_locations(preds)
            errors.append(spatial.distance.cdist(preds, Y).mean())
       
        max_error = np.max(errors)
        importance = [1 - (e / max_error) for e in errors]
        importance_data = {"snp": snp_names, "error": errors,
                           "importance": importance}
        ranking = pd.DataFrame(importance_data).sort_values("importance",
                                                            ascending=False)

        if save:
            ranking.to_csv(os.path.join(self.output_folder,
                "regressor_site_importance.csv"), index=False)

        return ranking

    # Plotting functions below
    def plot_training_curve(self, save=True):
        """
        Plot the training curve of the model.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the plot to a png file. The default is True.
            
        Returns
        -------
        None.
        """

        _plot_training_curve(self.train_history, self.nn_type,
            self.output_folder, save)

    def plot_location(self, sampleID=None, save=True):
        """
        Plot the predicted location of a sample of unknown origin
        compared to known locations of populations.
        
        Parameters
        ----------
        sampleID : str, optional
            The sampleID to plot. If None, all samples will be plotted.
            The default is None.
        save : bool, optional
            Whether to save the plot to a png file. The default is True.
            
        Returns
        -------
        None.
        """
        if self.regression is None:
            raise Exception("No regression data available. Please run" + 
                " assign_unknown() first.")

        if sampleID is None:
            sample_list = self.data.unknowns.sampleID.unique()
        else:
            sample_list = [sampleID]

        unique_test_data = self.data.test.drop_duplicates(subset=["pop"])

        sample_data = pd.DataFrame()
        for sample in sample_list:
            pred_sample_df = pd.DataFrame()
            pred_sample_df["x"] = self.regression[self.regression["sampleID"] == sample
                ]["x_pred"].values
            pred_sample_df["y"] = self.regression[self.regression["sampleID"] == sample
                ]["y_pred"].values
            pred_sample_df["pop"] = "prediction"
            pred_sample_df["sampleID"] = sample

            pop_sample_df = pd.DataFrame()
            pop_sample_df["x"] = unique_test_data["x"].values
            pop_sample_df["y"] = unique_test_data["y"].values
            pop_sample_df["pop"] = unique_test_data["pop"].values
            pop_sample_df["sampleID"] = sample

            sample_data = pd.concat([sample_data, pred_sample_df, pop_sample_df], axis=0)

        # TODO: Create custom colour palette to highlight pred
        g = sns.FacetGrid(sample_data, hue="pop", col="sampleID",
            col_wrap=3, height=3) 
        g.map(plt.scatter, "x", "y", s=50)
        g.add_legend()

        if save:
            plt.savefig(os.path.join(self.output_folder, "location_plot.png"))

        plt.close()


    def plot_contour_map(self, sampleID=None):
        """
        Plots contour map for a given sampleID. If no sampleID is provided,
        all samples of unknown origin will be plotted.

        Parameters
        ----------
        sampleID : str, optional
            The sampleID to plot. If None, all samples will be plotted.
            The default is None.
        
        Returns
        -------
        None.
        """
        if self.contour_classification is None:
            raise ValueError("Classification results not available. Please run " +
                "classify_by_contours() before plotting contour maps.")

        if sampleID is None:
            sample_list = self.data.unknowns.sampleID.unique()
        else:
            sample_list = [sampleID]

        image_list = []

        for sample in sample_list:    
            image = mpimg.imread(os.path.join(self.output_folder,
                                "contour_" + sample + ".png"))
            image_list.append(image)

        cols = min(3, len(sample_list))
        rows = int(np.ceil(len(image_list) / cols))
        fig, axs = plt.subplots(rows, cols)
        plt.subplots_adjust(wspace=0.00, hspace=0.00)

        if (rows == 1 & cols == 1):
            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)
            axs.spines['bottom'].set_visible(False)
            axs.spines['left'].set_visible(False)
            axs.get_xaxis().set_ticks([])
            axs.get_yaxis().set_ticks([])
            axs.imshow(image_list[0])
        else:
            for i, ax in enumerate(axs.flat):
                if i < len(image_list):
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                    ax.imshow(image_list[i])
                else:
                    ax.axis("off")

        plt.show()
        plt.close()
        

    def plot_confusion_matrix(self, save=True):
        """
        Plots confusion matrix. This functions uses the true and predicted
        labels generated from the test data to give a visual representation
        of the accuracy of the model.

        Parameters
        ----------
        save : bool, optional
            Whether to save the plot to a png file. The default is True.
        
        Returns
        -------
        None.
        """
        _plot_confusion_matrix(self.classification_test_results,
            self.classification_confusion_matrix,
            self.nn_type, self.output_folder, save)


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
        if self.contour_classification is None:
            raise ValueError("No classification results to plot.")

        e_preds = self.contour_classification.copy()

        _plot_assignment(e_preds, col_scheme, self.output_folder,
            self.nn_type, save)


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
        classes = np.unique(self.classification_test_results["true_pop"])
        preds = pd.DataFrame(self.classification_confusion_matrix,
                             columns=classes,
                             index=classes)

        _plot_structure(preds, col_scheme, self.nn_type, 
            self.output_folder, save)

    def save(self, save_path=None, filename="regressor.pkl"):
        """
        Saves the current instance of the class to a pickle file.

        Parameters
        ----------
        save_path : str, optional
            The path to save the file to. If None, the file will be saved
            to the current working directory. The default is None.
        filename : str, optional
            The name of the file to save. The default is "regressor.pkl".

        Returns
        -------
        None.
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

    # Hidden functions below
    def _fit_regressor_model(self, epochs, train_loader, valid_loader, 
                             net, optimizer, loss_func):

        loss_dict = {"epoch": [], "train": [], "valid": []}

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

                if valid_loss < self.lowest_val_loss:
                    self.lowest_val_loss = valid_loss
                    torch.save(net, os.path.join(self.output_folder, "best_model.pt"))

            # Calculate average validation loss
            avg_valid_loss = valid_loss / len(valid_loader)

            loss_dict["epoch"].append(epoch)
            loss_dict["train"].append(avg_train_loss)
            loss_dict["valid"].append(avg_valid_loss)

        return pd.DataFrame(loss_dict)


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

    def _generate_bootstraps(self, nboots=50):

        test_locs_final = pd.DataFrame({"sampleID": [], "pop": [], "x": [],
                                        "y": [], "x_pred": [], "y_pred": []})
        pred_locs_final = pd.DataFrame({"sampleID": [], "pop": [], 
                                        "x_pred": [], "y_pred": []})    

        # Use bootstrap to randomly select sites from training/test/unknown data
        num_sites = self.data.train["alleles"].values[0].shape[0]

        for boot in range(nboots):
            site_indices = np.random.choice(range(num_sites), size=num_sites,
                                            replace=True)

            boot_data = GeneticData()
            boot_data.train = self.data.train.copy()
            boot_data.test = self.data.test.copy()
            boot_data.knowns = pd.concat([self.data.train, self.data.test])
            boot_data.unknowns = self.data.unknowns.copy()

            # Slice datasets by site_indices
            boot_data.train["alleles"] = [a[site_indices] for a in self.data.train["alleles"].values]
            boot_data.test["alleles"] = [a[site_indices] for a in self.data.test["alleles"].values]
            boot_data.unknowns["alleles"] = [a[site_indices] for a in self.data.unknowns["alleles"].values]

            # Train on new training set
            self.train(boot_data=boot_data)
            test_locs = self.test(boot_data=boot_data)
            test_locs["sampleID"] = test_locs.index
            pred_locs = self.assign_unknown(boot_data=boot_data)

            test_locs_final = pd.concat([test_locs_final,
                test_locs[["sampleID", "pop", "x", "y", "x_pred", "y_pred"]]])
            pred_locs_final = pd.concat([pred_locs_final,
                pred_locs[["sampleID", "pop", "x", "y", "x_pred", "y_pred"]]])

        self.test_locs_final = test_locs_final # option to save
        self.pred_locs_final = pred_locs_final # option to save

    def _test_classification(self, test_locs, num_contours, save_plots):

        test_report = {"sampleID": [], "true_pop": [],
                       "pred_pop": [], "kd_estimate": []}

        for sample in test_locs["sampleID"].unique():

            sample_df = test_locs[test_locs["sampleID"] == sample]

            X, Y, Z, xlim, ylim = self._generate_kde(sample_df)
            cset = self._find_cset_from_contours(X, Y, Z, xlim,
                ylim, test_locs, num_contours, sample, save_plots)

            # Find predicted pop
            pred_pop, kd = self._contour_finder(test_locs, cset)
            test_report["sampleID"].append(sample)
            test_report["true_pop"].append(np.unique(sample_df["pop"])[0])
            test_report["pred_pop"].append(pred_pop)
            test_report["kd_estimate"].append(kd)

        return pd.DataFrame(test_report)

    def _classify_unknowns(self, pred_locs, test_locs, num_contours, save_plots):

        classification_data = {"sampleID": [], "classification": [], "kd_estimate": []}

        for sample in pred_locs["sampleID"].unique():

            sample_df = pred_locs[pred_locs["sampleID"] == sample]

            X, Y, Z, xlim, ylim = self._generate_kde(sample_df)
            cset = self._find_cset_from_contours(X, Y, Z, xlim,
                ylim, test_locs, num_contours, sample, save_plots)

            # Find predicted pop
            pred_pop, kd = self._contour_finder(test_locs, cset)
            classification_data["sampleID"].append(sample)
            classification_data["classification"].append(pred_pop)
            classification_data["kd_estimate"].append(kd)

        return pd.DataFrame(classification_data)

    def _contour_finder(self, true_dat, cset):
        """
        Finds population in densest contour.

        Parameters
        ----------
        true_dat : pd.DataFrame
            Dataframe containing x and y coordinates of all populations in
            training set.
        cset : matplotlib.contour.QuadContourSet
            Contour values for each contour polygon.

        Returns
        pred_pop : string
            Name of population in densest contour.
        """

        cont_dict = {"pop": [], "cont": []}

        for pop in true_dat["pop"].values:
            cont_dict["pop"].append(pop)
            cont = 0
            point = np.array(
                [
                    [
                        true_dat[true_dat["pop"] == pop]["x"].values[0],
                        true_dat[true_dat["pop"] == pop]["y"].values[0],
                    ]
                ]
            )

            for i in range(1, len(cset.allsegs)):
                for j in range(len(cset.allsegs[i])):
                    path = matplotlib.path.Path(cset.allsegs[i][j].tolist())
                    inside = path.contains_points(point)
                    if inside[0]:
                        cont = i
                        break
                    else:
                        next
            cont_dict["cont"].append(np.round(cset.levels[cont], 2))

        pred_pop = cont_dict["pop"][np.argmin(cont_dict["cont"])]

        return pred_pop, min(cont_dict["cont"])

    def _generate_kde(self, sample_df):

        d_x = (max(sample_df["x_pred"]) - min(sample_df["x_pred"])) / 5
        d_y = (max(sample_df["y_pred"]) - min(sample_df["y_pred"])) / 5
        xlim = min(sample_df["x_pred"]) - d_x, max(sample_df["x_pred"]) + d_x
        ylim = min(sample_df["y_pred"]) - d_y, max(sample_df["y_pred"]) + d_y

        X, Y = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]

        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([sample_df["x_pred"], sample_df["y_pred"]])

        try:
            kernel = stats.gaussian_kde(values)
        except (ValueError) as e:
            raise Exception("Too few points to generate contours") from e

        Z = np.reshape(kernel(positions).T, X.shape)
        new_z = Z / np.max(Z)

        return X, Y, new_z, xlim, ylim

    def _find_cset_from_contours(self, X, Y, Z, xlim, ylim, test_locs, num_contours,
                       sample, save):

        # Plot
        plt.ioff()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        cset = ax.contour(X, Y, Z, levels=num_contours, colors="black")

        cset.levels = -np.sort(-cset.levels)

        for pop in np.unique(test_locs["pop"].values):
            x = test_locs[test_locs["pop"] == pop]["x"].values[0]
            y = test_locs[test_locs["pop"] == pop]["y"].values[0]
            plt.scatter(x, y, cmap=plt.cm.Spectral, label=pop)

        ax.clabel(cset, cset.levels, inline=1, fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.title(sample)
        plt.legend()

        if save:
            plt.savefig(self.output_folder + "/contour_" + \
                        sample + ".png", format="png")

        plt.close()

        return cset

    # Validation functions
    def _validate_init_inputs(self, data, nboots, random_state, output_folder):
        if not isinstance(data, GeneticData):
            raise TypeError("data must be an instance of GeneticData")

        if not isinstance(nboots, int):
            raise TypeError("nboots must be an integer")

        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer")

        if output_folder is not None:
            if not isinstance(output_folder, str):
                raise TypeError("output_folder must be a string")

            if not os.path.isdir(output_folder):
                raise ValueError("output_folder must be a valid directory")

    def _validate_train_inputs(self, epochs, valid_size, cv_splits, cv_reps,
                            learning_rate, batch_size, dropout_prop, boot_data):

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

        if boot_data is not None:
            if not isinstance(boot_data, GeneticData):
                raise TypeError("boot_data must be an instance of GeneticData")

    def _validate_contour_inputs(self, nboots, num_contours, save_plots, save):
            
        if not isinstance(nboots, int):
            raise TypeError("nboots must be an integer")

        if not isinstance(num_contours, int):
            raise TypeError("num_contours must be an integer")

        if not isinstance(save_plots, bool):
            raise TypeError("save_plots must be a boolean")

        if not isinstance(save, bool):
            raise TypeError("save must be a boolean")