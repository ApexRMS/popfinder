import numpy as np
import pandas as pd
import allel # change to sgkit eventually
import zarr
import h5py
import sys
import os

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

class GeneticData():
    def __init__(self, genetic_data=None, sample_data=None, test_size=0.2, seed=123):

        self.genetic_data = genetic_data
        self.sample_data = sample_data
        self.seed = seed
        self.meanlong = None
        self.meanlat = None
        self.stdlon = None
        self.stdlat = None

        if genetic_data is not None and sample_data is not None:
            self._initialize(test_size=test_size, seed=seed)

    def read_data(self):
        """
        Reads a .zarr, .vcf, or h5py file containing genetic data and
        compiles information into a pandas DataFrame for either a 
        classifier or regressor neural network.

        Returns
        -------
        data : Pandas DataFrame
            Contains information on corresponding sampleID and
            genetic information.
        """

        # Check formats of datatypes
        if os.path.exists(self.genetic_data) is False:
            raise ValueError("Path to genetic_data does not exist")

        if os.path.exists(self.sample_data) is False:
            raise ValueError("Path to sample_data does not exist")

        # Load genotypes
        print("loading genotypes")
        samples, dc = self._load_genotypes(self.genetic_data)

        # Load data and organize for output
        print("loading sample data")
        locs = pd.read_csv(self.sample_data, sep="\t")
        locs = self._sort_samples(locs, samples)
        locs["alleles"] = list(dc)

        # Normalize location data
        self._retrieve_summary_stats(locs)
        locs = self._normalize_locations(locs)

        return locs

    def split_unknowns(self, data):
        """
        Splits data into known and unknown samples.
        
        Parameters
        ----------
        data : Pandas DataFrame
            Contains information on corresponding sampleID and
            genetic information. This is the output of `read_data()`.
        
        Returns
        -------
        known : Pandas DataFrame
            Contains information on corresponding sampleID and
            genetic information for known samples.
        unknown : Pandas DataFrame
            Contains information on corresponding sampleID and
            genetic information for unknown samples.
        """
        # Split data into known and unknown
        unknown = data[data["pop"].isnull()]
        known = data[data["pop"].notnull()]

        return known, unknown

    def split_train_test(self, data=None, stratify_by_pop=True, test_size=0.2, seed=123):
        """
        Splits data into training and testing sets.
        
        Parameters
        ----------  
        stratify_by_pop : bool
            Whether to stratify the data by population. Default is True.
        test_size : float
            Proportion of data to be used for testing. Default is 0.2.
        seed : int
            Random seed for reproducibility. Default is 123.
            
        Returns
        -------
        train : Pandas DataFrame
            Contains information on corresponding sampleID and
            genetic information for training samples.
        test : Pandas DataFrame
            Contains information on corresponding sampleID and
            genetic information for testing samples.
        """
        # Split data into training and testing
        if stratify_by_pop is True:
            train, test = self._stratified_split(data, test_size=test_size, seed=seed)
        else:
            train, test = self._random_split(data, test_size=test_size, seed=seed)

        return train, test

    def split_kfcv(self, data=None, n_splits=5, n_reps=1, stratify_by_pop=True):
        """
        Splits data into training and testing sets.

        Parameters
        ----------
        n_splits : int
            Number of splits for cross-validation. Default is 5.
        n_reps : int
            Number of repetitions for cross-validation. Default is 1.
        stratify_by_pop : bool
            Whether to stratify the data by population. Default is True.
        seed : int
            Seed for random number generator. Default is 123.

        Returns
        -------
        list of tuples
            Each tuple contains a training and testing set. The length
            of the list is n_splits * n_reps.
        """
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_reps,
                                       random_state=self.seed)
        dataset_list = []

        if data is None:
            known_data = self.knowns.copy()
        else:
            known_data = data

        if stratify_by_pop:
            for _, (train_ind, test_ind) in enumerate(rskf.split(known_data["alleles"],
                                                                 known_data["pop"], 
                                                                 groups=known_data["pop"])):
                train = known_data.iloc[train_ind]
                test = known_data.iloc[test_ind]
                dataset_tuple = (train, test)
                dataset_list.append(dataset_tuple)
        else:
            for _, (train_ind, test_ind) in enumerate(rskf.split(known_data["alleles"], 
                                                                 known_data["pop"])):
                train = known_data.iloc[train_ind]
                test = known_data.iloc[test_ind]
                dataset_tuple = (train, test)
                dataset_list.append(dataset_tuple)

        return dataset_list

    def _initialize(self, test_size=0.2, seed=123):

        self.data = self.read_data()
        self.knowns, self.unknowns = self.split_unknowns(self.data)
        self.train, self.test = self.split_train_test(test_size=test_size, seed=seed)

    def _load_genotypes(self, genetic_data):

        if genetic_data.endswith(".zarr"):
            callset = zarr.open_group(genetic_data, mode="r")
            gt = callset["calldata/GT"]
            gen = allel.GenotypeArray(gt[:])
            samples = callset["samples"][:]

        elif genetic_data.endswith(".vcf") or genetic_data.endswith(".vcf.gz"):
            vcf = allel.read_vcf(genetic_data, log=sys.stderr)
            gen = allel.GenotypeArray(vcf["calldata/GT"])
            samples = vcf["samples"]

        elif genetic_data.endswith(".locator.hdf5"):
            h5 = h5py.File(genetic_data, "r")
            dc = np.array(h5["derived_counts"])
            samples = np.array(h5["samples"])
            h5.close()

        else:
            raise ValueError("genetic_data must have extension 'zarr', 'vcf', or 'hdf5'")

        # count derived alleles for biallelic sites
        if genetic_data.endswith(".locator.hdf5") is False:

            print("counting alleles")
            ac = gen.to_allele_counts()
            biallel = gen.count_alleles().is_biallelic()
            dc = np.array(ac[biallel, :, 1], dtype="int_")
            dc = np.transpose(dc)

        return samples, dc

    def _sort_samples(self, locs, samples):

        if not pd.Series([
            "x", "pop", "y", "sampleID"
            ]).isin(locs.columns).all():
            raise ValueError("sample_data does not have correct columns")

        locs["id"] = locs["sampleID"]
        locs.set_index("id", inplace=True)

        # sort loc table so samples are in same order as genotype samples
        locs = locs.reindex(np.array(samples))

        # Create order column for indexing
        locs["order"] = np.arange(0, len(locs))

        # check that all sample names are present
        if not all(
            [locs["sampleID"][x] == samples[x] for x in range(len(samples))]
        ):
            raise ValueError(
                "sample ordering failed! Check that sample IDs match VCF.")

        return locs

    def _stratified_split(self, data, test_size=0.2, seed=123):

        if data is None:
            data = self.knowns

        X_train, X_test, y_train, y_test = train_test_split(
            data["alleles"], data[["x", "y", "pop"]],
            stratify=data["pop"],
            random_state=seed, test_size=test_size)

        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        return train, test

    def _random_split(self, test_size=0.2, seed=123):

        if data is None:
            data = self.knowns

        X_train, X_test, y_train, y_test = train_test_split(
        data["alleles"], data[["x", "y", "pop"]],
        random_state=seed, test_size=test_size)

        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        return train, test

    def _retrieve_summary_stats(self, raw_data):

        self.meanlong = np.nanmean(raw_data['x'])
        self.sdlong = np.nanstd(raw_data['x'])
        self.meanlat = np.nanmean(raw_data['y'])
        self.sdlat = np.nanstd(raw_data['y'])

    def _normalize_locations(self, raw_data):
        """
        Normalize location corrdinates.

        Parameters
        ----------
        raw_data : pd.DataFrame
            A pandas DataFrame corresponding to the results from `read_data()`.

        Returns
        -------
        data : pd.DataFrame
            A pandas DataFrame with normalized location coordinates.
        """
        raw_data["x_norm"] = (raw_data['x'].tolist() - self.meanlong) / self.sdlong
        raw_data["y_norm"] = (raw_data['y'].tolist() - self.meanlat) / self.sdlat

        return raw_data

    def _unnormalize_locations(self, data):
        """
        Unnormalize location corrdinates.

        Parameters
        ----------
        data : pd.DataFrame
            A pandas DataFrame corresponding to the results from `read_data()`.

        Returns
        -------
        data : pd.DataFrame
            A pandas DataFrame with unnormalized location coordinates.
        """
        data["x"] = data['x_norm'].tolist() * self.sdlong + self.meanlong
        data["y"] = data['y_norm'].tolist() * self.sdlat + self.meanlat

        return data