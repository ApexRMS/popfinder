import pandas as pd
import numpy as np
import multiprocessing as mp
import argparse
import os

from dataloader import GeneticData
from classifier import PopClassifier
# from popfinder.regressor import PopRegressor

def _train_on_bootstraps(clf_object, train_args):
    
    # Train on new training set
    clf_object.train(**train_args)
    # Save trained model to output folder
    # clf_object.save()

    # Return losses
    clf_object.train_history.to_csv(os.path.join(clf_object.output_folder, "loss.csv"), index=False)
    return clf_object.train_history

def create_classifier_objects(rep_start, nreps, nboots, popfinder_path):

    classifier_objects = []
    for rep in range(rep_start, nreps):
        for boot in range(nboots):

            popfinder = PopClassifier.load(os.path.join(popfinder_path, "classifier.pkl"))
            popfinder.output_folder = os.path.join(popfinder.output_folder, f"rep{rep+1}_boot{boot+1}")
            os.makedirs(popfinder.output_folder, exist_ok=True)

            # Use bootstrap to randomly select sites from training/test/unknown data
            num_sites = popfinder.data.train["alleles"].values[0].shape[0]

            site_indices = np.random.choice(range(num_sites), size=num_sites,
                                            replace=True)

            popfinder.__boot_data = GeneticData()
            popfinder.__boot_data.train = popfinder.data.train.copy()
            popfinder.__boot_data.test = popfinder.data.test.copy()
            popfinder.__boot_data.knowns = pd.concat([popfinder.data.train, popfinder.data.test])
            popfinder.__boot_data.unknowns = popfinder.data.unknowns.copy()

            # Slice datasets by site_indices
            popfinder.__boot_data.train["alleles"] = [a[site_indices] for a in popfinder.data.train["alleles"].values]
            popfinder.__boot_data.test["alleles"] = [a[site_indices] for a in popfinder.data.test["alleles"].values]
            popfinder.__boot_data.unknowns["alleles"] = [a[site_indices] for a in popfinder.data.unknowns["alleles"].values]

            classifier_objects.append(popfinder)

    return classifier_objects

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to PopClassifier object")
    parser.add_argument("--validsize", help="Validation size", type=float)
    parser.add_argument("--cvsplits", help="Number of cross-validation splits", type=int)
    parser.add_argument("--repstart", help="Starting repetition number", type=int)
    parser.add_argument("--nreps", help="Number of repetitions", type=int)
    parser.add_argument("--nboots", help="Number of bootstraps", type=int)
    parser.add_argument(
        "--patience", 
        help="Number of epochs to wait for improvement before stopping training")
    parser.add_argument(
        "--mindelta", 
        help="Minimum change in loss to be considered an improvement", 
        type=float)
    parser.add_argument("--learningrate", help="Learning rate", type=float)
    parser.add_argument("--batchsize", help="Batch size", type=int)
    parser.add_argument("--dropout", help="Dropout proportion", type=float)
    parser.add_argument("--hiddensize", help="Hidden layer size", type=int)
    parser.add_argument("--hiddenlayers", help="Number of hidden layers", type=int)
    parser.add_argument("--epochs", help="Number of epochs", type=int)   
    parser.add_argument("--jobs", help="Number of jobs", type=int)

    args = parser.parse_args()
    popfinder_path = args.path
    valid_size = args.validsize 
    cv_splits = args.cvsplits
    rep_start = args.repstart
    nreps = args.nreps
    nboots = args.nboots
    patience = args.patience
    min_delta = args.mindelta
    learning_rate = args.learningrate
    batch_size = args.batchsize
    dropout_prop = args.dropout
    hidden_size = args.hiddensize
    hidden_layers = args.hiddenlayers
    epochs = args.epochs
    num_jobs = args.jobs

    # Generate inputs
    classifier_objects = create_classifier_objects(rep_start, nreps, nboots, popfinder_path)
    # Create dictionary of train args
    train_args = {"valid_size": valid_size, "cv_splits": cv_splits, "nreps": nreps, 
                  "bootstraps": nboots, "patience": patience, "min_delta": min_delta,
                  "learning_rate": learning_rate, "batch_size": batch_size,
                  "dropout_prop": dropout_prop, "hidden_size": hidden_size,
                  "hidden_layers": hidden_layers, "epochs": epochs}

    if num_jobs == -1:
        num_jobs = mp.cpu_count()
    pool = mp.Pool(processes=num_jobs)
    results = pool.starmap(_train_on_bootstraps, [(c, train_args) for c in classifier_objects])
    pool.close()
    pool.join()

    for rep in range((nreps - rep_start)):
        for boot in range(nboots):
            ind = rep * nboots + boot
            results[ind]["rep"] = rep_start + rep + 1
            results[ind]["bootstrap"] = boot + 1
    
    final_results = pd.concat(results, ignore_index=True)
    final_results.to_csv(os.path.join(popfinder_path, "train_history.csv"), index=False)    