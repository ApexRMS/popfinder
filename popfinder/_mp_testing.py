import pandas as pd
import multiprocessing as mp
import argparse
import os

from classifier import PopClassifier

def _test_on_bootstraps(rep, boot, popfinder_path):
    
    classifier_object = PopClassifier.load(os.path.join(
        popfinder_path, f"rep{rep}_boot{boot}", "classifier.pkl"))
    
    # Test for all CV splits
    classifier_object.test(use_best_model=False)

    # Return losses
    return classifier_object.test_results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="Path to PopClassifier object")
    parser.add_argument("-n", help="Number of bootstraps", type=int)
    parser.add_argument("-r", help="Number of repetitions", type=int)
    parser.add_argument("-j", help="Number of jobs", type=int)
    args = parser.parse_args()
    popfinder_path = args.p
    nboots = args.n
    nreps = args.r
    num_jobs = args.j

    if num_jobs == -1:
        num_jobs = mp.cpu_count()
    pool = mp.Pool(processes=num_jobs)
    results = pool.starmap(
        _test_on_bootstraps, 
        [(rep, boot, popfinder_path) for rep, boot in zip(range(nreps), range(nboots))])
    pool.close()
    pool.join()

    for rep in range(nreps):
        for boot in range(nboots):
            ind = rep * nboots + boot
            results[ind]["rep"] = rep + 1
            results[ind]["bootstrap"] = boot + 1
    
    final_results = pd.concat(results, ignore_index=True)
    final_results.to_csv(os.path.join(popfinder_path, "test_results.csv"), index=False)