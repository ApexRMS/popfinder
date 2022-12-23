import numpy as np
import pandas as pd
import allel
import zarr
import h5py
import sys
import os

def read_data(infile, sample_data, save_allele_counts=False):
    """
    Reads a .zarr, .vcf, or h5py file containing genetic data and
    compiles information into a pandas DataFrame for either a 
    classifier or regressor neural network.

    Parameters
    ----------
    infile : string
        Path to the .zarr, .vcf, or h5py file.
    sample_data : string
        Path to .txt file containing sample information
        (columns are x, y, sampleID, and pop). See documentation
        on help formatting this file.
    save_allele_counts : boolean
        Saves derived allele count information (Default=False).

    Returns
    -------
    sample_df : Pandas DataFrame
        Contains information on corresponding sampleID and
        population classifications.
    dc : np.array
        Array of derived allele counts.
    unknowns : dataframe
        If kfcv is set to False, returns a dataframe with
        information about sampleID and indices for samples
        of unknown origin.
    """

    # Check formats of datatypes
    if os.path.exists(infile) is False:
        raise ValueError("Path to infile does not exist")

    if os.path.exists(sample_data) is False:
        raise ValueError("Path to sample_data does not exist")

    # Load genotypes
    print("loading genotypes")
    if infile.endswith(".zarr"):

        callset = zarr.open_group(infile, mode="r")
        gt = callset["calldata/GT"]
        gen = allel.GenotypeArray(gt[:])
        samples = callset["samples"][:]

    elif infile.endswith(".vcf") or infile.endswith(".vcf.gz"):

        vcf = allel.read_vcf(infile, log=sys.stderr)
        gen = allel.GenotypeArray(vcf["calldata/GT"])
        samples = vcf["samples"]

    elif infile.endswith(".locator.hdf5"):

        h5 = h5py.File(infile, "r")
        dc = np.array(h5["derived_counts"])
        samples = np.array(h5["samples"])
        h5.close()

    else:
        raise ValueError("Infile must have extension 'zarr', 'vcf', or 'hdf5'")

    # count derived alleles for biallelic sites
    if infile.endswith(".locator.hdf5") is False:

        print("counting alleles")
        ac = gen.to_allele_counts()
        biallel = gen.count_alleles().is_biallelic()
        dc = np.array(ac[biallel, :, 1], dtype="int_")
        dc = np.transpose(dc)

        if (save_allele_counts and
                not infile.endswith(".locator.hdf5")):

            print("saving derived counts for reanalysis")
            outfile = h5py.File(infile + ".locator.hdf5", "w")
            outfile.create_dataset("derived_counts", data=dc)
            outfile.create_dataset("samples", data=samples,
                                   dtype=h5py.string_dtype())
            outfile.close()

    # Load data and organize for output
    print("loading sample data")

    locs = pd.read_csv(sample_data, sep="\t")

    if not pd.Series(["x",
                      "pop",
                      "y",
                      "sampleID"]).isin(locs.columns).all():
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

    else:

        # Find unknown locations as NAs in the dataset
        unknowns = locs.iloc[np.where(pd.isnull(locs["pop"]))]

        # Extract known location information for training
        samples = samples[np.where(pd.notnull(locs["pop"]))]
        locs = locs.iloc[np.where(pd.notnull(locs["pop"]))]
        order = np.array(locs["order"])
        locs = np.array(locs["pop"])
        samp_list = pd.DataFrame({"samples": samples,
                                  "pops": locs,
                                  "order": order})

    return samp_list, dc, unknowns