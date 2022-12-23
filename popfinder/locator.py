import allel
import os
import sys
import zarr
import time
import subprocess
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import spatial
from tqdm import tqdm
from matplotlib import pyplot as plt
import json
from tensorflow.keras import backend as K


def locator(
    sample_data,
    gen_dat,
    train_split=0.9,
    windows=False,
    window_start=0,
    window_stop=None,
    window_size=5e5,
    bootstrap=False,
    jacknife=False,
    jacknife_prop=0.05,
    nboots=50,
    batch_size=32,
    max_epochs=5000,
    patience=10,
    min_mac=2,
    max_SNPs=None,
    impute_missing=False,
    dropout_prop=0.25,
    nlayers=10,
    width=256,
    out="out",
    seed=None,
    gpu_number=None,
    plot_history=True,
    keep_weights=False,
    load_params=None,
    keras_verbose=1,
):
    """
    Estimates sample locations from genotype matrices. Code
    modified from Battey et al. 2020:
    https://github.com/kr-colab/locator

    Parameters:
    -----------
    sample_data : string
        Tab-delimited text file with columns 'sampleID', 'x',
        and 'y'. SampleIDs must exactly match those of the
        VCF. Samples without known locations should be NA.
    train_split : float
        Proportion of samples to be used for training, ranges
        from 0-1 (Default=0.9).
    windows : boolean
        Run windowed analysis over a single chromosome (requires
        zarr input; Default=False).
    window_start : int
        Starting point of windowed analysis (Default=0).
    window_stop : int
        Stopping point of windowed analysis (Default=None).
    window_size : int
        Size of window for windowed analysis (Default=5e5).
    bootstrap : boolean
        Run bootstrap replicates by retraining on bootstrapped
        data (Default=False).
    jacknife : boolean
        Run jacknife uncertainty estimate on a trained network.
        NOTE: we recommend this onyl as a fast heuristic --
        use bootstrap option or run windowed analysis for final
        results (Default=False).
    nboots : int
        Number of bootstrap replicates to run (Default=50).
    batch_size : int
        Number of samples per batch (Default=32).
    max_epochs : int
        Number of epochs (Default=5000).
    patience : int
        Number of epochs to run the optimizer after last
        improvement in validation loss (Default=100).
    min_mac : int
        Minimum minor allele count (Default=2).
    max_SNPs : int
        Randomly select max_SNPs variants to use in the
        analysis (Default=None).
    impute_missing : boolean
        If False, all alleles at missing sites are ancestral
        Default=False).
    dropout_prop : float
        Proportion of weights to zero at the dropout layer
        (Default=0.25).
    nlayers : int
        Number of layers in the network (Default=10).
    width : int
        Number of units per layer in the network (Default=256).
    out : string
        File name stem for output (Default='out').
    seed : int
        Random seed for train/test splits and SNP subsetting
        (Default=None).
    gpu_number : string
        GPU (Default=None).
    plot_history : boolean
        Plot training history (Default=True).
    keep_weights : boolean
        Keep model weights after training (Default=False).
    load_params : string
        Path to a _params.json file to load parameters from a
        previous run. Parameters from the json file will
        supersede all parameters provided (Default=None).
    keras_verbose : int
        Verbose argument passed to keras in model training.
        0 = silent; 1 = progress bars for minibatches;
        2 = show epochs. Yes, 1 is more verbose than 2. Blame
        keras (Default=1).
    """

    # set seed and gpu
    if seed is not None:
        np.random.seed(seed)
    if gpu_number is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

    # store run params
    with open(out + "_params.json", "w") as f:
        json.dump(locals(), f, indent=2, default=repr)
    f.close()

    # load old run parameters
    if load_params is not None:
        with open(load_params, "r") as f:
            obj = json.load(f)  # fix dict thing
        f.close()
        for key, val in obj.items():
            exec(key + "=val")

    # Get data format from gen_dat suffix
    if gen_dat.endswith('vcf'):
        data_format = 'vcf'
    elif gen_dat.endswith('txt'):
        data_format = 'matrix'
    elif gen_dat.endswith('zarr'):
        data_format = 'zarr'

    # Set output file prefix
    out = out + "/loc"

    if not bootstrap and not jacknife:
        boot = None
        genotypes, samples = load_genotypes(gen_dat, data_format)
        sample_data, locs = sort_samples(samples, sample_data, genotypes)
        meanlong, sdlong, meanlat, sdlat, locs = normalize_locs(locs)
        ac = filter_snps(genotypes, min_mac, max_SNPs, impute_missing)
        checkpointer, earlystop, reducelr = load_callbacks(
            "FULL", bootstrap, jacknife, out, keras_verbose, patience
        )
        (
            train,
            test,
            traingen,
            testgen,
            trainlocs,
            testlocs,
            pred,
            predgen,
        ) = split_train_test(ac, locs, train_split)
        model = load_network(traingen, dropout_prop, nlayers, width)
        start = time.time()
        history, model = train_network(
            model,
            traingen,
            testgen,
            trainlocs,
            testlocs,
            bootstrap,
            jacknife,
            checkpointer,
            earlystop,
            reducelr,
            keras_verbose,
            max_epochs,
            batch_size,
            out,
            boot,
        )
        dists = predict_locs(
            model,
            predgen,
            sdlong,
            meanlong,
            sdlat,
            meanlat,
            testlocs,
            pred,
            samples,
            testgen,
            bootstrap,
            jacknife,
            windows,
            out,
            history,
            boot,
        )
        if plot_history:
            plot_history_fn(history, dists, out)
        if not keep_weights:
            subprocess.run("rm " + out + "_weights.hdf5", shell=True)
        end = time.time()
        elapsed = end - start
        print("run time " + str(elapsed / 60) + " minutes")
    elif bootstrap:
        boot = "FULL"
        genotypes, samples = load_genotypes(gen_dat, data_format)
        sample_data, locs = sort_samples(samples, sample_data, genotypes)
        meanlong, sdlong, meanlat, sdlat, locs = normalize_locs(locs)
        ac = filter_snps(genotypes, min_mac, max_SNPs, impute_missing)
        checkpointer, earlystop, reducelr = load_callbacks(
            "FULL", bootstrap, jacknife, out, keras_verbose, patience
        )
        (
            train,
            test,
            traingen,
            testgen,
            trainlocs,
            testlocs,
            pred,
            predgen,
        ) = split_train_test(ac, locs, train_split)
        model = load_network(traingen, dropout_prop, nlayers, width)
        start = time.time()
        history, model = train_network(
            model,
            traingen,
            testgen,
            trainlocs,
            testlocs,
            bootstrap,
            jacknife,
            checkpointer,
            earlystop,
            reducelr,
            keras_verbose,
            max_epochs,
            batch_size,
            out,
            boot,
        )
        dists = predict_locs(
            model,
            predgen,
            sdlong,
            meanlong,
            sdlat,
            meanlat,
            testlocs,
            pred,
            samples,
            testgen,
            bootstrap,
            jacknife,
            windows,
            out,
            history,
            boot,
        )
        if plot_history:
            plot_history_fn(history, dists, out)
        if not keep_weights:
            subprocess.run("rm " + out + "_bootFULL_weights.hdf5",
                           shell=True)
        end = time.time()
        elapsed = end - start
        print("run time " + str(elapsed / 60) + " minutes")
        for boot in range(nboots):
            np.random.seed(np.random.choice(range(int(1e6)), 1))
            checkpointer, earlystop, reducelr = load_callbacks(
                boot, bootstrap, jacknife, out, keras_verbose, patience
            )
            print("starting bootstrap " + str(boot))
            traingen2 = copy.deepcopy(traingen)
            testgen2 = copy.deepcopy(testgen)
            predgen2 = copy.deepcopy(predgen)
            site_order = np.random.choice(
                traingen2.shape[1], traingen2.shape[1], replace=True
            )
            traingen2 = traingen2[:, site_order]
            testgen2 = testgen2[:, site_order]
            predgen2 = predgen2[:, site_order]
            model = load_network(traingen2, dropout_prop, nlayers, width)
            start = time.time()
            history, model = train_network(
                model,
                traingen2,
                testgen2,
                trainlocs,
                testlocs,
                bootstrap,
                jacknife,
                checkpointer,
                earlystop,
                reducelr,
                keras_verbose,
                max_epochs,
                batch_size,
                out,
                boot,
            )
            dists = predict_locs(
                model,
                predgen2,
                sdlong,
                meanlong,
                sdlat,
                meanlat,
                testlocs,
                pred,
                samples,
                testgen2,
                bootstrap,
                jacknife,
                windows,
                out,
                history,
                boot,
            )
            if plot_history:
                plot_history_fn(history, dists, out)
            if not keep_weights:
                subprocess.run(
                    "rm " + out + "_boot" + str(boot) + "_weights.hdf5",
                    shell=True
                )
            end = time.time()
            elapsed = end - start
            K.clear_session()
            print("run time " + str(elapsed / 60) + " minutes\n\n")
    elif jacknife:
        boot = "FULL"
        genotypes, samples = load_genotypes(gen_dat, data_format)
        sample_data, locs = sort_samples(samples, sample_data, genotypes)
        meanlong, sdlong, meanlat, sdlat, locs = normalize_locs(locs)
        ac = filter_snps(genotypes, min_mac, max_SNPs, impute_missing)
        checkpointer, earlystop, reducelr = load_callbacks(
            boot, bootstrap, jacknife, out, keras_verbose, patience
        )
        (
            train,
            test,
            traingen,
            testgen,
            trainlocs,
            testlocs,
            pred,
            predgen,
        ) = split_train_test(ac, locs, train_split)
        model = load_network(traingen, dropout_prop, nlayers, width)
        start = time.time()
        history, model = train_network(
            model,
            traingen,
            testgen,
            trainlocs,
            testlocs,
            bootstrap,
            jacknife,
            checkpointer,
            earlystop,
            reducelr,
            keras_verbose,
            max_epochs,
            batch_size,
            out,
            boot,
        )
        dists = predict_locs(
            model,
            predgen,
            sdlong,
            meanlong,
            sdlat,
            meanlat,
            testlocs,
            pred,
            samples,
            testgen,
            bootstrap,
            jacknife,
            windows,
            out,
            history,
            boot,
        )
        if plot_history:
            plot_history_fn(history, dists, out)
        end = time.time()
        elapsed = end - start
        print("run time " + str(elapsed / 60) + " minutes")
        print("starting jacknife resampling")
        af = []
        for i in tqdm(range(ac.shape[0])):
            af.append(sum(ac[i, :]) / (ac.shape[1] * 2))
        af = np.array(af)
        for boot in tqdm(range(nboots)):
            checkpointer, earlystop, reducelr = load_callbacks(
                boot, bootstrap, jacknife, out, keras_verbose, patience
            )
            pg = copy.deepcopy(predgen)  # this asshole
            sites_to_remove = np.random.choice(
                pg.shape[1], int(pg.shape[1] * jacknife_prop),
                replace=False
            )  # treat X% of sites as missing data
            for i in sites_to_remove:
                pg[:, i] = np.random.binomial(2, af[i], pg.shape[0])
                # pg[:,i]=af[i]
            dists = predict_locs(
                model,
                pg,
                sdlong,
                meanlong,
                sdlat,
                meanlat,
                testlocs,
                pred,
                samples,
                testgen,
                bootstrap,
                jacknife,
                windows,
                out,
                history,
                boot,
                verbose=False,
            )
        if not keep_weights:
            subprocess.run("rm " + out + "_bootFULL_weights.hdf5",
                           shell=True)


def load_genotypes(gen_dat, data_format):
    """
    Loads genetic data and parses into relevant genotype
    and sample data.

    Parameters
    ----------
    gen_dat : string
        Path to genetic data.
    data_format : string
        File format of SNPs for all samples. Options include
        'vcf', 'zarr', and 'matrix'.

    Returns
    -------
    genotypes : GenotypeArray
    samples : np.array
    """
    if data_format == "zarr":
        print("reading zarr")
        callset = zarr.open_group(gen_dat, mode="r")
        gt = callset["calldata/GT"]
        genotypes = allel.GenotypeArray(gt[:])
        samples = callset["samples"][:]
        # positions = callset["variants/POS"]
    elif data_format == "vcf":
        print("reading VCF")
        vcf = allel.read_vcf(gen_dat, log=sys.stderr)
        genotypes = allel.GenotypeArray(vcf["calldata/GT"])
        samples = vcf["samples"]
    elif data_format == "matrix":
        gmat = pd.read_csv(gen_dat, sep="\t")
        samples = np.array(gmat["sampleID"])
        gmat = gmat.drop(labels="sampleID", axis=1)
        gmat = np.array(gmat, dtype="int8")
        for i in range(
            gmat.shape[0]
        ):  # kludge to get haplotypes for reading in to allel.
            h1 = []
            h2 = []
            for j in range(gmat.shape[1]):
                count = gmat[i, j]
                if count == 0:
                    h1.append(0)
                    h2.append(0)
                elif count == 1:
                    h1.append(1)
                    h2.append(0)
                elif count == 2:
                    h1.append(1)
                    h2.append(1)
            if i == 0:
                hmat = h1
                hmat = np.vstack((hmat, h2))
            else:
                hmat = np.vstack((hmat, h1))
                hmat = np.vstack((hmat, h2))
        genotypes = allel.HaplotypeArray(
            np.transpose(hmat)
        ).to_genotypes(ploidy=2)
    return genotypes, samples


def sort_samples(samples, sample_data, genotypes):
    """
    Sorts samples so they are in the same order as the vcf.

    Parameters
    ----------
    samples : list
        List of sampleIDs from the vcf file
    sample_data

    Returns
    -------
    sample_data : Dataframe
    locs : np.array
    """
    sample_data = pd.read_csv(sample_data, sep="\t")
    sample_data["sampleID2"] = sample_data["sampleID"]
    sample_data.set_index("sampleID", inplace=True)
    samples = samples.astype("str")
    sample_data = sample_data.reindex(
        np.array(samples)
    )  # sort loc table so samples are in same order as vcf samples
    if not all(
        [
            sample_data[
                "sampleID2"
            ][x] == samples[x] for x in range(len(samples))
        ]
    ):  # check that all sample names are present
        print("sample ordering failed! Check that sample IDs match the VCF.")
        sys.exit()
    locs = np.array(sample_data[["x", "y"]])
    print("loaded " + str(np.shape(genotypes)) + " genotypes\n\n")
    return sample_data, locs


# replace missing sites with binomial(2,mean_allele_frequency)
def replace_md(genotypes):
    """
    Replace missing sites with values taken from binomial
    distribution with parameter mean_allele_frequency.

    Parameters
    ----------
    genotypes : GenotypeArray

    Returns
    -------
    ac : allele count array
    """
    print("imputing missing data")
    dc = genotypes.count_alleles()[:, 1]
    ac = genotypes.to_allele_counts()[:, :, 1]
    missingness = genotypes.is_missing()
    ninds = np.array([np.sum(x) for x in ~missingness])
    af = np.array([dc[x] / (2 * ninds[x]) for x in range(len(ninds))])
    for i in tqdm(range(np.shape(ac)[0])):
        for j in range(np.shape(ac)[1]):
            if missingness[i, j]:
                ac[i, j] = np.random.binomial(2, af[i])
    return ac


def filter_snps(genotypes, min_mac, max_SNPs, impute_missing):
    """
    Filters SNPs based on parameters specified.

    Parameters
    ----------
    genotypes
    min_mac
    max_SNPs

    Returns
    -------
    ac : allele counts
    """
    print("filtering SNPs")
    tmp = genotypes.count_alleles()
    biallel = tmp.is_biallelic()
    genotypes = genotypes[biallel, :, :]
    if not min_mac == 1:
        derived_counts = genotypes.count_alleles()[:, 1]
        ac_filter = [x >= min_mac for x in derived_counts]
        genotypes = genotypes[ac_filter, :, :]
    if impute_missing:
        ac = replace_md(genotypes)
    else:
        ac = genotypes.to_allele_counts()[:, :, 1]
    if max_SNPs is not None:
        ac = ac[np.random.choice(
            range(ac.shape[0]), max_SNPs, replace=False
        ), :]
    print("running on " + str(len(ac)) + " genotypes after filtering\n\n\n")
    return ac


def normalize_locs(locs):
    """
    Normalize location corrdinates.

    Parameters
    ----------
    locs

    Returns
    -------
    meanlong
    sdlong
    meanlat
    sdlat
    locs
    """
    meanlong = np.nanmean(locs[:, 0])
    sdlong = np.nanstd(locs[:, 0])
    meanlat = np.nanmean(locs[:, 1])
    sdlat = np.nanstd(locs[:, 1])
    locs = np.array(
        [[(x[0] - meanlong) / sdlong, (x[1] - meanlat) / sdlat] for x in locs]
    )
    return meanlong, sdlong, meanlat, sdlat, locs


def split_train_test(ac, locs, train_split):
    """
    Splits data into test, train, and prediction sets.

    Parameters
    ----------
    ac
    locs

    Returns
    -------
    train
    test
    traingen
    testgen
    trainlocs
    testlocs
    pred
    predgen
    """
    train = np.argwhere(~np.isnan(locs[:, 0]))
    train = np.array([x[0] for x in train])
    pred = np.array([x for x in range(len(locs)) if x not in train])
    test = np.random.choice(train, round((1 - train_split) * len(train)),
                            replace=False)
    train = np.array([x for x in train if x not in test])
    traingen = np.transpose(ac[:, train])
    trainlocs = locs[train]
    testgen = np.transpose(ac[:, test])
    testlocs = locs[test]
    predgen = np.transpose(ac[:, pred])
    return train, test, traingen, testgen, trainlocs, testlocs, pred, predgen


def load_network(traingen, dropout_prop, nlayers, width):
    """
    Creates neural network based on specifications.

    Parameters
    ----------
    traingen
    dropout_prop
    nlayers
    width
    """

    def euclidean_distance_loss(y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.BatchNormalization(
            input_shape=(traingen.shape[1],)
        )
    )
    for i in range(int(np.floor(nlayers / 2))):
        model.add(tf.keras.layers.Dense(width, activation="elu"))
    model.add(tf.keras.layers.Dropout(dropout_prop))
    for i in range(int(np.ceil(nlayers / 2))):
        model.add(tf.keras.layers.Dense(width, activation="elu"))
    model.add(tf.keras.layers.Dense(2))
    model.add(tf.keras.layers.Dense(2))
    model.compile(optimizer="Adam", loss=euclidean_distance_loss)
    return model


def load_callbacks(boot, bootstrap, jacknife,
                   out, keras_verbose, patience):
    """
    Specifies Keras callbacks, including checkpoints, early stopping,
    and reducing learning rate.

    Parameters
    ----------
    boot
    bootstrap
    jacknife
    out
    keras_verbose
    patience
    batch_size

    Returns
    -------
    checkpointer
    earlystop
    reducelr
    """
    if bootstrap or jacknife:
        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath=out + "_boot" + str(boot) + "_weights.hdf5",
            verbose=keras_verbose,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            save_freq="epoch",
        )
    else:
        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath=out + "_weights.hdf5",
            verbose=keras_verbose,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            save_freq="epoch",
        )
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=patience
    )
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=int(patience / 6),
        verbose=keras_verbose,
        mode="auto",
        min_delta=0,
        cooldown=0,
        min_lr=0,
    )
    return checkpointer, earlystop, reducelr


def train_network(
    model,
    traingen,
    testgen,
    trainlocs,
    testlocs,
    bootstrap,
    jacknife,
    checkpointer,
    earlystop,
    reducelr,
    keras_verbose,
    max_epochs,
    batch_size,
    out,
    boot,
):
    """
    Trains neural network using given parameters.

    Parameters
    ----------
    model
    traingen
    testgen
    trainlocs
    testlocs
    bootstrap
    jacknife
    checkpointer
    earlystop
    reducelr
    keras_verbose
    max_epochs
    batch_size

    Returns
    -------
    history
    model
    """
    history = model.fit(
        traingen,
        trainlocs,
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=keras_verbose,
        validation_data=(testgen, testlocs),
        callbacks=[checkpointer, earlystop, reducelr],
    )
    if bootstrap or jacknife:
        model.load_weights(out + "_boot" + str(boot) + "_weights.hdf5")
    else:
        model.load_weights(out + "_weights.hdf5")
    return history, model


def predict_locs(
    model,
    predgen,
    sdlong,
    meanlong,
    sdlat,
    meanlat,
    testlocs,
    pred,
    samples,
    testgen,
    bootstrap,
    jacknife,
    windows,
    out,
    history,
    boot,
    size=None,
    i=None,
    verbose=True,
):
    """
    Predict locations from trained neural network.

    Parameters
    ----------
    model
    predgen
    sdlong
    meanlong
    sdlat
    meanlat
    testlocs
    pred
    samples
    testgen
    verbose

    Returns
    -------
    dists
    """
    if verbose:
        print("predicting locations...")
    prediction = model.predict(predgen)
    prediction = np.array(
        [
            [
                x[0] * sdlong + meanlong, x[1] * sdlat + meanlat
            ] for x in prediction
        ]
    )
    predout = pd.DataFrame(prediction)
    predout.columns = ["x", "y"]
    predout["sampleID"] = samples[pred]
    if bootstrap or jacknife:
        predout.to_csv(out + "_boot" + str(boot) + "_predlocs.txt",
                       index=False)
        testlocs2 = np.array(
            [
                [
                    x[0] * sdlong + meanlong, x[1] * sdlat + meanlat
                ] for x in testlocs
            ]
        )
    elif windows:
        predout.to_csv(
            out + "_" + str(i) + "-" + str(i + size - 1) + "_predlocs.txt",
            index=False
        )  # this is dumb
        testlocs2 = np.array(
            [
                [
                    x[0] * sdlong + meanlong, x[1] * sdlat + meanlat
                ] for x in testlocs
            ]
        )
    else:
        predout.to_csv(out + "_predlocs.txt", index=False)
        testlocs2 = np.array(
            [
                [
                    x[0] * sdlong + meanlong, x[1] * sdlat + meanlat
                ] for x in testlocs
            ]
        )
    p2 = model.predict(testgen)  # print validation loss to screen
    p2 = np.array(
        [
            [
                x[0] * sdlong + meanlong, x[1] * sdlat + meanlat
            ] for x in p2
        ]
    )
    r2_long = np.corrcoef(p2[:, 0], testlocs2[:, 0])[0][1] ** 2
    r2_lat = np.corrcoef(p2[:, 1], testlocs2[:, 1])[0][1] ** 2
    mean_dist = np.mean(
        [
            spatial.distance.euclidean(
                p2[x, :], testlocs2[x, :]
            ) for x in range(len(p2))
        ]
    )
    median_dist = np.median(
        [
            spatial.distance.euclidean(
                p2[x, :], testlocs2[x, :]
            ) for x in range(len(p2))
        ]
    )
    dists = [
        spatial.distance.euclidean(
            p2[x, :], testlocs2[x, :]
        ) for x in range(len(p2))
    ]

    if verbose:

        print(
            "R2(x)="
            + str(r2_long)
            + "\nR2(y)="
            + str(r2_lat)
            + "\n"
            + "mean validation error "
            + str(mean_dist)
            + "\n"
            + "median validation error "
            + str(median_dist)
            + "\n"
        )

    hist = pd.DataFrame(history.history)
    hist.to_csv(out + "_history.txt", sep="\t", index=False)

    return dists


def plot_history_fn(history, dists, out):
    """
    Plots training vs validation history over time (epochs).

    Parameters
    ----------
    history
    dists
    out

    Returns
    -------
    PDF
    """
    plt.switch_backend("agg")
    fig = plt.figure(figsize=(4, 1.5), dpi=200)
    plt.rcParams.update({"font.size": 7})
    ax1 = fig.add_axes([0, 0, 0.4, 1])
    ax1.plot(history.history["val_loss"][3:], "-", color="black", lw=0.5)
    ax1.set_xlabel("Validation Loss")
    ax2 = fig.add_axes([0.55, 0, 0.4, 1])
    ax2.plot(history.history["loss"][3:], "-", color="black", lw=0.5)
    ax2.set_xlabel("Training Loss")
    fig.savefig(out + "_fitplot.pdf", bbox_inches="tight")