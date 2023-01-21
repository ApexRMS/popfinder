from popfinder.dataloader import GeneticData
from popfinder.classifier import PopClassifier
from popfinder.regressor import PopRegressor
import pytest
import numpy as np
import pandas as pd
import os

# Test dataloader class
def test_genetic_data_inputs():

    gen_dat = GeneticData()
    assert isinstance(gen_dat, GeneticData)
    with pytest.raises(ValueError, match="genetic_data is None"):
        gen_dat.read_data()

    gen_dat = GeneticData(genetic_data="tests/test_data/test.vcf")  
    with pytest.raises(ValueError, match="sample_data is None"):
        gen_dat.read_data()

    with pytest.raises(ValueError, match="Path to genetic_data does not exist"):
        gen_dat = GeneticData(genetic_data="bad/path.vcf",
                            sample_data="tests/test_data/testNA.txt")

    with pytest.raises(ValueError, match="genetic_data must have extension 'zarr', 'vcf', or 'hdf5'"):
        gen_dat = GeneticData(genetic_data="tests/test_data/testNA.txt",
                            sample_data="tests/test_data/testNA.txt")

    with pytest.raises(ValueError, match="Path to sample_data does not exist"):
        gen_dat = GeneticData(genetic_data="tests/test_data/test.vcf",
                            sample_data="bad/path.txt")

    with pytest.raises(ValueError, match="sample_data must have extension 'txt' or 'tsv'"):
        gen_dat = GeneticData(genetic_data="tests/test_data/test.vcf",
                            sample_data="tests/test_data/test.vcf")

    with pytest.raises(ValueError, match="sample_data does not have correct columns"):
        gen_dat = GeneticData(genetic_data="tests/test_data/test.vcf",
                            sample_data="tests/test_data/test_bad.txt")

    with pytest.raises(ValueError, match="genetic_data must be a string"):
        gen_dat = GeneticData(genetic_data=123,
                            sample_data="tests/test_data/testNA.txt")

    with pytest.raises(ValueError, match="sample_data must be a string"):
        gen_dat = GeneticData(genetic_data="tests/test_data/test.vcf",
                            sample_data=123)

    with pytest.raises(ValueError, match="test_size must be a float"):
        gen_dat = GeneticData(genetic_data="tests/test_data/test.vcf",
                            sample_data="tests/test_data/testNA.txt",
                            test_size="0.2")

    with pytest.raises(ValueError, match="seed must be an integer"):
        gen_dat = GeneticData(genetic_data="tests/test_data/test.vcf",
                            sample_data="tests/test_data/testNA.txt",
                            test_size=0.2,
                            seed=0.5)             

def test_genetic_data():

    gen_dat = GeneticData(genetic_data="tests/test_data/test.vcf",
                          sample_data="tests/test_data/testNA.txt")
    assert isinstance(gen_dat, GeneticData)

    dat = gen_dat.read_data()
    assert dat.equals(gen_dat.data)

    assert gen_dat.data.empty == False
    assert gen_dat.knowns.empty == False
    assert gen_dat.unknowns.empty == False
    assert gen_dat.train.empty == False
    assert gen_dat.test.empty == False

# Test classifier class

# Test regressor class