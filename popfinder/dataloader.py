import numpy as np
import pandas as pd
import allel # change to sgkit eventually
import sys
import os

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

class GeneticData():
    """
    Class for reading in genetic data and sample data and compiling
    information into a pandas DataFrame for either a classifier or
    regressor neural network.

    Parameters
    ----------
    genetic_data : str
        Path to genetic data file. Can be .zarr, .vcf, or .h5py.
    sample_data : str
        Path to sample data file. Must be .tsv or .txt.
    test_size : float
        Proportion of data to be used for testing.
    seed : int
        Seed for random number generator.
    
    Attributes
    ----------
    genetic_data : str
        Path to genetic data file. Can be .zarr, .vcf, or .h5py.
    sample_data : str
        Path to sample data file. Must be .tsv or .txt and contain the 
        columns "pop" (population name), and "sampleID". The "sampleID" 
        must match the sample IDs in the genetic data file. 
    seed : int
        Seed for random number generator.
    
    Methods
    -------
    read_data()
        Reads a .zarr, .vcf, or h5py file containing genetic data and
        compiles information into a pandas DataFrame for either a
        classifier or regressor neural network.
    split_train_test(data, test_size, seed)
        Splits data into training and testing sets.
    split_kfcv(data, n_splits, n_reps, seed)
        Splits data into k-fold cross validation sets.
    split_unknowns(data)
        Splits data into known and unknown samples.
    """
    def __init__(self, genetic_data=None, sample_data=None, test_size=0.2, 
                 test_samples=None, exclude_pops=None, seed=123):

        self._validate_init_inputs(genetic_data, sample_data, test_size, 
                                   test_samples, seed)

        self.genetic_data = genetic_data
        self.sample_data = sample_data
        self.seed = seed

        if genetic_data is not None and sample_data is not None:
            self._initialize(test_size=test_size, test_samples=test_samples, 
                             exclude_pops=exclude_pops, seed=seed)

    def read_data(self, exclude_pops):
        """
        Reads a .vcf file containing genetic data and
        compiles information into a pandas DataFrame for either a 
        classifier or regressor neural network.

        Parameters
        ----------
        exclude_pops : list
            List of populations that are in the genetic_data and sample_data 
            files but you want to exclude from the analysis.
        
        Returns
        -------
        data : Pandas DataFrame
            Contains information on corresponding sampleID and
            genetic information.
        """

        self._validate_read_data_inputs()

        # Load genotypes
        print("Loading genotypes...")
        samples, dc = self._load_genotypes(self.genetic_data)

        # Load data and organize for output
        print("Loading sample data...")
        locs = pd.read_csv(self.sample_data, sep="\t")
        locs = self._sort_samples(locs, samples)
        locs["alleles"] = list(dc)
        
        if exclude_pops is not None:
            locs = locs[~locs["pop"].isin(exclude_pops)]

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

    def split_train_test(self, data=None, stratify_by_pop=True, test_size=0.2, 
                         test_samples=None, seed=123, bootstrap=False):
        """
        Splits data into training and testing sets.
        
        Parameters
        ----------
        data : Pandas DataFrame
            Contains information on corresponding sampleID and
            genetic information. This is the output of `read_data()`.  
        stratify_by_pop : bool
            Whether to stratify the data by population. Default is True.
        test_size : float
            Proportion of data to be used for testing. Default is 0.2.
        seed : int
            Random seed for reproducibility. Default is 123.
        bootstrap : bool
            Whether to bootstrap the training data. Default is False.
            
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
        # TODO: test the test_samples method
        if test_samples is not None:
            test_samples = pd.read_tsv(test_samples).tolist()
            test = data[data["sampleID"].isin(test_samples)]
            train = data[~data["sampleID"].isin(test_samples)]

        elif stratify_by_pop is True:
            train, test = self._stratified_split(data, test_size=test_size, seed=seed)
            
        else:
            train, test = self._random_split(data, test_size=test_size, seed=seed)

        if bootstrap:
            train = train.sample(frac=1, replace=True, random_state=seed)

        return train, test

    def split_kfcv(self, data=None, n_splits=5, n_reps=1, seed=123, stratify_by_pop=True, bootstrap=False):
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
        bootstrap : bool
            Whether to bootstrap the training data. Default is False.

        Returns
        -------
        list of tuples
            Each tuple contains a training and testing set. The length
            of the list is n_splits * n_reps.
        """
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_reps,
                                       random_state=seed)
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

                if bootstrap:
                    train = train.sample(frac=1, replace=True, random_state=seed)

                dataset_tuple = (train, test)
                dataset_list.append(dataset_tuple)
        else:
            for _, (train_ind, test_ind) in enumerate(rskf.split(known_data["alleles"], 
                                                                 known_data["pop"])):
                train = known_data.iloc[train_ind]
                test = known_data.iloc[test_ind]

                if bootstrap:
                    train = train.sample(frac=1, replace=True, random_state=seed)
                    
                dataset_tuple = (train, test)
                dataset_list.append(dataset_tuple)

        return dataset_list

    def update_unknowns(self, new_genetic_data, new_sample_data):
        """
        Reads a .vcf file containing genetic data 
        of new samples from unknown origins, replacing the old
        data for samples from unknown origins stored in the GeneticData
        object.

        Parameters
        ----------
        new_genetic_data : str
            Path to genetic .vcf file.
        new_sample_data : str
            Path to sample data file. Must be .tsv or .txt and contain the 
            columns "x" (longitude), "y" (latitude), "pop" (population name),
            and "sampleID". The "sampleID" must match the sample IDs in 
            the genetic data file. The "pop" column should be filled with 
            NAs since the new samples are of unknown origin.

        Returns
        -------
        None
        """

        self._validate_update_unknowns_inputs(new_genetic_data, new_sample_data)

        # Load genotypes
        print("loading genotypes")
        samples, dc = self._load_genotypes(new_genetic_data)

        # Load data and organize for output
        print("loading sample data")
        locs = pd.read_csv(new_sample_data, sep="\t")
        locs = self._sort_samples(locs, samples)
        locs["alleles"] = list(dc)

        # Reset unknowns and full data
        _, self.unknowns = self.split_unknowns(locs)
        self.data = pd.concat([self.knowns, self.unknowns], ignore_index=True)

    def _initialize(self, test_size=0.2, test_samples=None, exclude_pops=None, seed=123):

        self.data = self.read_data(exclude_pops)
        self.knowns, self.unknowns = self.split_unknowns(self.data)
        self.train, self.test = self.split_train_test(
            test_size=test_size, test_samples=test_samples, seed=seed)

        # Create label encoder from train target
        self.label_enc = LabelEncoder()
        self.label_enc.fit_transform(self.train["pop"])

    def _load_genotypes(self, genetic_data):

        if genetic_data.endswith(".vcf") or genetic_data.endswith(".vcf.gz"):
            vcf = allel.read_vcf(genetic_data, log=sys.stderr)
            gen = allel.GenotypeArray(vcf["calldata/GT"])
            samples = vcf["samples"]

        else:
            raise ValueError("genetic_data must have extension 'vcf'")

        # Count derived alleles for biallelic sites
        ac = gen.to_allele_counts()
        biallel = gen.count_alleles().is_biallelic()
        dc = np.array(ac[biallel, :, 1], dtype="int_")
        dc = np.transpose(dc)

        return samples, dc

    def _sort_samples(self, locs, samples):

        if not pd.Series([
            "pop", "sampleID"
            ]).isin(locs.columns).all():
            raise ValueError("sample_data does not have correct columns")

        # Sort sample data so in the same order as the VCF file
        locs["id"] = locs["sampleID"]
        locs.set_index("id", inplace=True)
        locs = locs.reindex(np.array(samples))
        locs["order"] = np.arange(0, len(locs))

        # Check sample names in input txt file match those in the VCF
        if not all(np.array(locs["sampleID"]) == samples):
            raise ValueError(
                "Sample IDs in sample_data file do not match VCF.")

        return locs

    def _stratified_split(self, data, test_size=0.2, seed=123):

        if data is None:
            data = self.knowns

        X_train, X_test, y_train, y_test = train_test_split(
            data["alleles"], data[["pop"]],
            stratify=data["pop"],
            random_state=seed, test_size=test_size)

        successful_split = False

        while not successful_split:
            train = pd.concat([X_train, y_train], axis=1)
            test = pd.concat([X_test, y_test], axis=1)
            training_pops = np.unique(train["pop"])
            testing_pops = np.unique(test["pop"])
            successful_split = np.array_equal(training_pops, testing_pops)

        return train, test

    def _random_split(self, test_size=0.2, seed=123):

        if data is None:
            data = self.knowns

        X_train, X_test, y_train, y_test = train_test_split(
        data["alleles"], data[["pop"]],
        random_state=seed, test_size=test_size)

        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        return train, test

    def _validate_init_inputs(self, genetic_data, sample_data, test_size, 
                              test_samples, seed):

        if genetic_data is not None and not isinstance(genetic_data, str):
            raise ValueError("genetic_data must be a string")

        if sample_data is not None and not isinstance(sample_data, str):
            raise ValueError("sample_data must be a string")

        if not isinstance(test_size, float):
            raise ValueError("test_size must be a float")

        if test_size > 1 or test_size < 0:
            raise ValueError("test_size must be between 0 and 1")
        
        if test_samples is not None and not isinstance(test_samples, str):
            raise ValueError("test_samples must be a string")

        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")

    def _validate_read_data_inputs(self):

        if self.genetic_data is None:
            raise ValueError("genetic_data is None")

        if self.sample_data is None:
            raise ValueError("sample_data is None")

        if os.path.exists(self.genetic_data) is False:
            raise ValueError("Path to genetic_data does not exist")

        if os.path.exists(self.sample_data) is False:
            raise ValueError("Path to sample_data does not exist")

        if self.genetic_data.endswith((".vcf")) is False:
            raise ValueError("genetic_data must have extension 'vcf'")

        if self.sample_data.endswith((".txt", ".tsv")) is False:
            raise ValueError("sample_data must have extension 'txt' or 'tsv'")

        locs = pd.read_csv(self.sample_data, sep="\t")
        locs_list = locs.columns.tolist()
        if set(locs_list) != set(["pop", "sampleID"]):
            raise ValueError("sample_data file does not have correct columns")

    def _validate_update_unknowns_inputs(self, new_genetic_data, new_sample_data):

        if not isinstance(new_genetic_data, str):
            raise ValueError("new_unknowns must be a path to a genetic data file")

        if not isinstance(new_sample_data, str):
            raise ValueError("new_unknowns must be a path to a sample data file")

        if os.path.exists(new_genetic_data) is False:
            raise ValueError("Path to new_genetic_data does not exist")

        if os.path.exists(new_sample_data) is False:
            raise ValueError("Path to new_sample_data does not exist")

        if new_genetic_data.endswith((".zarr", ".vcf", ".hdf5")) is False:
            raise ValueError("new_genetic_data must have extension 'zarr', 'vcf', or 'hdf5'")

        if new_sample_data.endswith((".txt", ".tsv")) is False:
            raise ValueError("new_sample_data must have extension 'txt' or 'tsv'")

        locs = pd.read_csv(new_sample_data, sep="\t")
        locs_list = locs.columns.tolist()
        if set(locs_list) != set(["pop", "sampleID"]):
            raise ValueError("sample_data file does not have correct columns")

        
