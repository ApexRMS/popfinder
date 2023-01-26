import pandas as pd
import numpy as np
import multiprocessing as mp
import argparse
import os

from popfinder.dataloader import GeneticData
from popfinder.regressor import PopRegressor

nreps=5
nboots=50
epochs=100
valid_size=0.2 
cv_splits=1
cv_reps=1
learning_rate=0.001 
batch_size=16
dropout_prop=0.01

def _train_on_bootstraps(arg_list):

    popfinder_path, nboots, epochs, valid_size, cv_splits, cv_reps, learning_rate, batch_size, dropout_prop, rep = arg_list

    popfinder = PopRegressor.load(os.path.join(popfinder_path, "regressor.pkl"))
    popfinder.output_folder = os.path.join(popfinder_path, "rep{}".format(rep))
    os.makedirs(popfinder.output_folder, exist_ok=True)

    test_locs_final = pd.DataFrame({"sampleID": [], "pop": [], "x": [],
                                    "y": [], "x_pred": [], "y_pred": []})
    pred_locs_final = pd.DataFrame({"sampleID": [], "pop": [], 
                                    "x_pred": [], "y_pred": []})    

    # Use bootstrap to randomly select sites from training/test/unknown data
    num_sites = popfinder.data.train["alleles"].values[0].shape[0]

    for _ in range(nboots):

        site_indices = np.random.choice(range(num_sites), size=num_sites,
                                        replace=True)

        boot_data = GeneticData()
        boot_data.train = popfinder.data.train.copy()
        boot_data.test = popfinder.data.test.copy()
        boot_data.knowns = pd.concat([popfinder.data.train, popfinder.data.test])
        boot_data.unknowns = popfinder.data.unknowns.copy()

        # Slice datasets by site_indices
        boot_data.train["alleles"] = [a[site_indices] for a in popfinder.data.train["alleles"].values]
        boot_data.test["alleles"] = [a[site_indices] for a in popfinder.data.test["alleles"].values]
        boot_data.unknowns["alleles"] = [a[site_indices] for a in popfinder.data.unknowns["alleles"].values]

        # Train on new training set
        popfinder.data
        popfinder.train(epochs=epochs, valid_size=valid_size,
                cv_splits=cv_splits, cv_reps=cv_reps,
                learning_rate=learning_rate, batch_size=batch_size,
                dropout_prop=dropout_prop, boot_data=boot_data)
        popfinder.test(boot_data=boot_data)
        test_locs = popfinder.test_results.copy()
        test_locs["sampleID"] = test_locs.index
        pred_locs = popfinder.assign_unknown(boot_data=boot_data, save=False)

        test_locs_final = pd.concat([test_locs_final,
            test_locs[["sampleID", "pop", "x", "y", "x_pred", "y_pred"]]])
        pred_locs_final = pd.concat([pred_locs_final,
            pred_locs[["sampleID", "pop", "x", "y", "x_pred", "y_pred"]]])

    return test_locs_final, pred_locs_final


# def _parallelize_runs(self, func, nboots, nreps, epochs, valid_size,
#                       cv_splits, cv_reps, learning_rate, batch_size,
#                       dropout_prop):
#     num_processes = mp.cpu_count() - 1
#     with mp.Pool(num_processes) as pool:
#         results = pool.map(func, [(nboots, epochs, valid_size, cv_splits,
#                                     cv_reps, learning_rate, batch_size,
#                                     dropout_prop) for _ in range(nreps)])

    # return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="Path to PopRegressor object")
    args = parser.parse_args()
    popfinder_path = args.p

    pool = mp.Pool(processes=5) # test with 1 process
    results = pool.map(_train_on_bootstraps, [[popfinder_path, nboots, epochs, valid_size, cv_splits,
                                    cv_reps, learning_rate, batch_size,
                                    dropout_prop, rep] for rep in range(nreps)])

    # results = [_train_on_bootstraps([popfinder_path, nboots, epochs, valid_size, cv_splits,
    #                             cv_reps, learning_rate, batch_size,
    #                             dropout_prop, rep]) for rep in range(nreps)]
    results.to_csv("results.csv")

    final_results = pd.concat([results])