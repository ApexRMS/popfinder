from itertools import product
import pandas as pd
import random
import os
from popfinder.classifier import PopClassifier

def hyperparam_search(classifier, trials=None, valid_size=0.2, 
                      cv_splits=1, nreps=1, bootstraps=1, patience=10, 
                      min_delta=0.001, learning_rate=[0.01], dropout_prop=[0.025], 
                      batch_size=[16], epochs=[100], jobs=1, hyperparam_dict={}):
    """
    Perform grid search or random search hyperparameter optimization for a 
    PopClassifier object.

    Parameters
    ----------
    classifier : PopClassifier
        A PopClassifier object.
    trials : int, optional
        Number of hyperparameter combinations to test. If set to None,
        then will test all combinations. The default is None.
    valid_size : float, optional
        Proportion of data to use for validation. This is only used if 
        cv_splits is set to 1. The default is 0.2.
    cv_splits : int, optional
        Number of cross-validation splits. The default is 1 (i.e., no splits).
    nreps : int, optional
        Number of repetitions for cross-validation. The default is 1.
    bootstraps : int, optional
        Number of bootstraps for cross-validation. The default is 1.
    patience : int, optional
        Number of epochs to wait for improvement before stopping training. The default is 10.
    min_delta : float, optional
        Minimum change in loss to be considered an improvement. The default is 0.001.
    learning_rate : list, optional
        List of learning rates to test. The default is [0.01].
    dropout_prop : list, optional
        List of dropout proportions to test. The default is [0.025].
    batch_size : list, optional
        List of batch sizes to test. The default is [16].
    epochs : list, optional
        List of numbers of epochs to test. The default is [100].
    jobs : int, optional
        Number of jobs to run in parallel. The default is 1.
    hyperparam_dict : optional
        Dictionary of additional hyperparameters for the optimizer. For Adam, can include
        beta1, beta2, weight_decay, and epsilon. For SGD, can include 
        momentum, dampening, weight_decay, and nesterov. For LBFGS, can include
        max_iter, max_eval, tolerance_grad, tolerance_change, history_size, and
        line_search_fn. See the pytorch documentation for more details.

    
    Returns
    -------
    results_df : pandas.DataFrame
        A dataframe containing the results of the grid search. 
    """

    #TODO: test
    additional_params = dict({"lr": learning_rate,
                              "drop_prop": dropout_prop,
                              "batch_size": batch_size,
                              "epochs": epochs})
    hyperparam_dict = {**additional_params, **hyperparam_dict}

    # Create list of hyperparameter combinations
    hyperparams = list(product(*[hyperparam_dict[hp] for hp in hyperparam_dict]))

    # If trials is set, then randomly sample from hyperparams
    if trials is not None:
        hyperparams = random.sample(hyperparams, trials)

    # Create output folder to store results
    output_folder = os.path.join(classifier.output_folder, "hyperparam_search")
    output_file = "hyperparam_search_classifier.pkl"
    classifier.save(save_path = output_folder, filename = output_file)
    
    # Create list to store results
    results = []

    # Iterate through hyperparameter combinations
    for hp in hyperparams:

        print(f"Testing hyperparameters: {hp}")
        # Create a new classifier object
        new_classifier = PopClassifier.load(os.path.join(output_folder, output_file))
        fname = "lr" + str(hp[0]) + "_drop" + str(hp[1]) + "_batch" + str(hp[2]) + "_epochs" + str(hp[3])
        new_classifier.output_folder = os.path.join(output_folder, fname)

        # Set hyperparameters
        lr = hp[0]
        d = hp[1]
        bs = hp[2]
        e = hp[3]

        new_classifier.train(jobs=jobs, nreps=nreps, bootstraps=bootstraps, 
                             cv_splits=cv_splits, valid_size=valid_size,
                             learning_rate=lr, dropout_prop=d, 
                             batch_size=bs, epochs=e, patience=patience,
                             min_delta=min_delta)
        
        # Average results across splits, reps, and bootstraps
        averaged_history = new_classifier.train_history.groupby(['epoch']).mean()

        # Store results
        train_loss = min(averaged_history["train_loss"])
        valid_loss = min(averaged_history["valid_loss"])
        valid_acc = max(averaged_history["valid_accuracy"])
        valid_prec = max(averaged_history["valid_precision"])
        valid_recall = max(averaged_history["valid_recall"])
        valid_f1 = max(averaged_history["valid_f1"])
        valid_mcc = max(averaged_history["valid_mcc"])
        
        results.append({"lr": lr, "drop_prop": d, "batch_size": int(bs), 
                        "epochs": int(e), "train_loss": train_loss,
                        "valid_loss": valid_loss, "valid_acc": valid_acc,
                        "valid_prec": valid_prec, "valid_recall": valid_recall,
                        "valid_f1": valid_f1, "valid_mcc": valid_mcc})
    
    # Create dataframe from results
    results_df = pd.DataFrame(results)
    return results_df 