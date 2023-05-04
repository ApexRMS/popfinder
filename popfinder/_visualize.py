import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import itertools
import os

def _plot_training_curve(train_history, nn_type, output_folder, save):

    plot_data = train_history.rename(columns={"valid": "validation"})
    plot_data = pd.melt(plot_data, id_vars=["epoch", "split", "rep"], 
            value_vars=["train", "validation"], 
            var_name="dataset", value_name="loss")

    d = {'color': ['C0', 'k'], "ls" : ["-","--"]}
    g = sns.FacetGrid(plot_data, col="split", row="rep", hue="dataset", hue_kws=d, height=2, aspect=1.5)
    g.map(sns.lineplot, "epoch", "loss", lw=0.5)
    g.add_legend(title="")

    if save:
        g.savefig(os.path.join(output_folder, nn_type + "_training_history.png"),
            bbox_inches="tight")

def _plot_confusion_matrix(test_results, confusion_matrix, nn_type,
    output_folder, save):

    true_labels = test_results["true_pop"] # test w/ classifier

    cm = np.round(confusion_matrix, 2)
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
        plt.savefig(os.path.join(output_folder, nn_type + "_confusion_matrix.png"))

def _plot_assignment(e_preds, col_scheme, output_folder,
    nn_type, save):

    e_preds.set_index("sampleID", inplace=True)

    if nn_type == "regressor":
        e_preds = pd.get_dummies(e_preds["classification"])
    elif nn_type == "classifier":
        e_preds = pd.get_dummies(e_preds["assigned_pop"])

    num_classes = len(e_preds.columns)

    sns.set()
    sns.set_style("ticks")
    e_preds.plot(kind="bar", stacked=True,
        colormap=ListedColormap(sns.color_palette(col_scheme, num_classes)),
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
        plt.savefig(os.path.join(
            output_folder, nn_type + "_assignment_plot.png"),
            bbox_inches="tight")

def _plot_structure(preds, col_scheme, nn_type, output_folder, save):

    num_classes = len(preds.index)

    sns.set()
    sns.set_style("ticks")
    preds.plot(kind="bar", stacked=True,
        colormap=ListedColormap(sns.color_palette(col_scheme, num_classes)),
        figsize=(12, 6), grid=None)
    legend = plt.legend(loc="center right", bbox_to_anchor=(1.2, 0.5),
        prop={"size": 15}, title="Predicted Pop")
    plt.setp(legend.get_title(), fontsize="x-large")
    plt.xlabel("Actual Pop", fontsize=20)
    plt.ylabel("Frequency of Assignment", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if save:
        plt.savefig(os.path.join(output_folder,
            nn_type + "_structure_plot.png"),
            bbox_inches="tight")