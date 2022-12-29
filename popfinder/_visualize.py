import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import itertools
import os

def _plot_training_curve(train_history, nn_type, output_folder, save):

    plt.switch_backend("agg")
    fig = plt.figure(figsize=(3, 1.5), dpi=200)
    plt.rcParams.update({"font.size": 7})
    ax1 = fig.add_axes([0, 0, 1, 1])
    ax1.plot(train_history["valid"][3:], "--", color="black",
        lw=0.5, label="Validation Loss")
    ax1.plot(train_history["train"][3:], "-", color="black",
        lw=0.5, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    if save:
        fig.savefig(os.path.join(output_folder, nn_type + "_training_history.png"),
            bbox_inches="tight")

    plt.close()

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

    plt.close()

def _plot_assignment(e_preds, col_scheme, output_folder,
    nn_type, save):

    e_preds.set_index("sampleID", inplace=True)

    if nn_type == "regressor":
        e_preds = pd.get_dummies(e_preds["classification"])
    elif nn_type == "classifier":
        e_preds = pd.get_dummies(e_preds["assigned_pop"])

    num_classes = len(e_preds.columns)

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
        plt.savefig(os.path.join(
            output_folder, nn_type + "_assignment_plot.png"),
            bbox_inches="tight")

    plt.close()

def _plot_structure(preds, col_scheme, nn_type, output_folder, save):

    num_classes = len(preds.index)

    sn.set()
    sn.set_style("ticks")
    preds.plot(kind="bar", stacked=True,
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
        plt.savefig(os.path.join(output_folder,
            nn_type + "_structure_plot.png"),
            bbox_inches="tight")