import torch
from torch.autograd import Variable
from scipy import spatial
from scipy import stats
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
import seaborn as sn
import numpy as np
import pandas as pd
import os
import itertools

from popfinder.preprocess import GeneticData
from popfinder._neural_networks import RegressorNet
from popfinder._helper import _generate_train_inputs
from popfinder._helper import _generate_data_loaders
from popfinder._helper import _data_converter
from popfinder._helper import _split_input_regressor

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
        self.median_distance = None
        self.mean_distance = None
        self.r2_lat = None
        self.r2_long = None
        self.summary = None
        self.contour_classification = None


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

    def test(self, boot_data=None):
        
        if boot_data is None:
            test_input = self.data.test
        else:
            test_input = boot_data.test

        X_test = test_input["alleles"]    
        y_test = test_input[["x", "y"]]
        X_test, y_test = _data_converter(X_test, y_test)

        y_pred = self.best_model(X_test).detach().numpy()
        y_pred = self._unnormalize_locations(y_pred)

        test_input = test_input.assign(x_pred=y_pred[:, 0], y_pred=y_pred[:, 1])

        dists = [
            spatial.distance.euclidean(
                y_pred[x, :], y_test[x, :]
            ) for x in range(len(y_pred))
        ]

        self.median_distance = np.median(dists) # won't be accurate, need to unnormalize
        self.mean_distance = np.mean(dists)
        self.r2_long = np.corrcoef(y_pred[:, 0], y_test[:, 0])[0][1] ** 2
        self.r2_lat = np.corrcoef(y_pred[:, 1], y_test[:, 1])[0][1] ** 2

        self.summary = self.get_assignment_summary()
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

        return unknown_data

    def generate_bootstraps(self, nboots=50):

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

    def generate_contours(self, num_contours=5, save_plots=True, 
                          save_data=True):

        test_locs = self.test_locs_final
        pred_locs = self.pred_locs_final

        classification_data = {"sampleID": [], "classification": [], "kd_estimate": []}

        for sample in pred_locs["sampleID"].unique():

            sample_df = pred_locs[pred_locs["sampleID"] == sample]
            classification_data["sampleID"].append(sample)
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

            # Plot
            fig = plt.figure(figsize=(8, 8))
            ax = fig.gca()
            plt.xlim(xlim[0], xlim[1])
            plt.ylim(ylim[0], ylim[1])
            cset = ax.contour(X, Y, new_z, levels=num_contours, colors="black")

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

            # Find predicted pop
            pred_pop, kd = self.contour_finder(test_locs, cset)
            classification_data["classification"].append(pred_pop)
            classification_data["kd_estimate"].append(kd)

            if save_plots is True:
                plt.savefig(self.output_folder + "/contour_" + \
                            sample + ".png", format="png")

            plt.close()

        self.contour_classification = pd.DataFrame(classification_data)

        if save_data:
            self.contour_classification.to_csv(self.output_folder + \
                "/classification.csv", index=False)

        return self.contour_classification

    def contour_finder(self, true_dat, cset):
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
            fig.savefig(self.output_folder + "/training_history.pdf",
                bbox_inches="tight")

        plt.close()

    def plot_location():
        pass

    def plot_contour_map():
        pass

    def plot_confusion_matrix(self, true_labels, pred_labels, save=True):

        cm = confusion_matrix(true_labels, pred_labels, normalize="true")
        cm = np.round(cm, 2)
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
            plt.savefig(self.output_folder + "/cm.png")

        plt.close()

    def plot_roc_curve():
        pass

    def plot_assignment(self, save=True, col_scheme="Spectral"):

        # Put code in shared function for both classifier and regressor?
        e_preds = self.contour_classification.copy()
        e_preds.set_index("sampleID", inplace=True)
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
        