import torch
from torch.autograd import Variable
from scipy import spatial
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import pickle
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
    """
    def __init__(self, data, nboots=20, random_state=123, output_folder=None):

        self.data = data # GeneticData object
        self.nboots = nboots
        self.random_state = random_state
        if output_folder is None:
            output_folder = os.getcwd()
        self.output_folder = output_folder
        self.train_history = None
        self.best_model = None
        self.regression = None
        self.median_distance = None
        self.mean_distance = None
        self.r2_lat = None
        self.r2_long = None
        self.summary = None
        self.contour_classification = None
        self.classification_test_results = None
        self.classification_accuracy = None
        self.classification_precision = None
        self.classification_recall = None
        self.classification_f1 = None
        self.classification_confusion_matrix = None

        self._nn_type = "regressor"

    def train(self, epochs=100, valid_size=0.2, cv_splits=1, cv_reps=1, boot_data=None):
        
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
            net = RegressorNet(X_train.shape[1], 16, len(y_train.unique()))
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
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

        self.train_history = loss_df_final
        self.best_model = torch.load(os.path.join(self.output_folder, "best_model.pt"))

    def test(self, boot_data=None, verbose=False):
        
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

        self.median_distance = np.median(dists)
        self.mean_distance = np.mean(dists)
        self.r2_long = np.corrcoef(y_pred[:, 0], y_test[:, 0])[0][1] ** 2
        self.r2_lat = np.corrcoef(y_pred[:, 1], y_test[:, 1])[0][1] ** 2

        self.summary = self.get_assignment_summary()

        if verbose:
            print(self.summary)

        return test_input

    def assign_unknown(self, boot_data=None):
        
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

        self.regression = unknown_data

        return unknown_data

    def classify_by_contours(self, nboots=5, num_contours=5,
        save_plots=True, save_data=True):

        self._generate_bootstraps(nboots=nboots) # also add option for multiple reps

        test_locs = self.test_locs_final
        pred_locs = self.pred_locs_final

        self.classification_test_results = self._test_classification(test_locs,
            num_contours, save_plots)
        self.contour_classification = self._classify_unknowns(pred_locs,
            test_locs, num_contours, save_plots)

        if save_data:
            self.classification_test_results.to_csv(os.path.join(self.output_folder,
                "contour_classification_test_report.csv"), index=False)
            self.contour_classification.to_csv(os.path.join(self.output_folder,
                "contour_classification_results.csv"), index=False)

        # Generate classification summary stats from test_report
        y_pred = self.classification_test_results["pred_pop"]
        y_true = self.classification_test_results["true_pop"]
        self.classification_confusion_matrix = np.round(
            confusion_matrix(y_true, y_pred, normalize="true"), 3)
        self.classification_accuracy = accuracy_score(y_true, y_pred)
        self.classification_precision = precision_score(y_true, y_pred, average="weighted")
        self.classification_recall = recall_score(y_true, y_pred, average="weighted")
        self.classification_f1 = f1_score(y_true, y_pred, average="weighted")

    # Reporting functions below
    def get_assignment_summary(self):

        summary = {
            "median_distance": [self.median_distance],
            "mean_distance": [self.mean_distance],
            "r2_long": [self.r2_long],
            "r2_lat": [self.r2_lat]
        }

        return summary

    def get_classification_summary(self):

        summary = { # need to grab all these items
            "accuracy": [self.classification_accuracy],
            "precision": [self.classification_precision],
            "recall": [self.classification_recall],
            "f1": [self.classification_f1],
            "confusion_matrix": [self.classification_confusion_matrix]
        }

        return summary

    # Plotting functions below
    def plot_training_curve(self, save=True):

        _plot_training_curve(self.train_history, self._nn_type,
            self.output_folder, save)

    def plot_location(self, sampleID=None, save=True):
        
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
        """
        _plot_confusion_matrix(self.classification_test_results,
            self.classification_confusion_matrix,
            self._nn_type, self.output_folder, save)

    def plot_roc_curve():
        pass # TODO: add later


    def plot_assignment(self, save=True, col_scheme="Spectral"):

        if self.contour_classification is None:
            raise ValueError("No classification results to plot.")

        e_preds = self.contour_classification.copy()

        _plot_assignment(e_preds, col_scheme, self.output_folder,
            self._nn_type, save)


    def plot_structure(self, save=True, col_scheme="Spectral"):
        """
        Plots the proportion of times individuals from the
        test data were assigned to the correct population. 
        Used for determining the accuracy of the classifier.
        """
        classes = np.unique(self.classification_test_results["true_pop"])
        preds = pd.DataFrame(self.classification_confusion_matrix,
                             columns=classes,
                             index=classes)

        _plot_structure(preds, col_scheme, self._nn_type, 
            self.output_folder, save)

    def save(self, save_path=None, filename="regressor.pkl"):
        """
        Saves the current instance of the class to a pickle file.
        """
        _save(self, save_path, filename)

    def load(self, load_path=None, filename="regressor.pkl"):
        """
        Loads a saved instance of the class from a pickle file.
        """
        _save(self, load_path, filename)

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