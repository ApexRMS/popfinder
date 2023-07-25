from popfinder.dataloader import GeneticData
from popfinder.classifier import PopClassifier
from popfinder.regressor import PopRegressor
from popfinder._neural_networks import ClassifierNet
from popfinder._neural_networks import RegressorNet
import pytest
import numpy as np
import pandas as pd
import torch
import os
import shutil
import re

TEST_OUTPUT_FOLDER = "tests/test_outputs"

# Empty test output folder if it exists
if os.path.exists(TEST_OUTPUT_FOLDER):
    shutil.rmtree(TEST_OUTPUT_FOLDER)

# Recreate test output folder
os.mkdir(TEST_OUTPUT_FOLDER)

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
        GeneticData(genetic_data="bad/path.vcf",
                            sample_data="tests/test_data/testNA.txt")

    with pytest.raises(ValueError, match="genetic_data must have extension 'zarr', 'vcf', or 'hdf5'"):
        GeneticData(genetic_data="tests/test_data/testNA.txt",
                            sample_data="tests/test_data/testNA.txt")

    with pytest.raises(ValueError, match="Path to sample_data does not exist"):
        GeneticData(genetic_data="tests/test_data/test.vcf",
                            sample_data="bad/path.txt")

    with pytest.raises(ValueError, match="sample_data must have extension 'txt' or 'tsv'"):
        GeneticData(genetic_data="tests/test_data/test.vcf",
                            sample_data="tests/test_data/test.vcf")

    with pytest.raises(ValueError, match="sample_data file does not have correct columns"):
        GeneticData(genetic_data="tests/test_data/test.vcf",
                            sample_data="tests/test_data/test_bad.txt")

    with pytest.raises(ValueError, match="genetic_data must be a string"):
        GeneticData(genetic_data=123,
                            sample_data="tests/test_data/testNA.txt")

    with pytest.raises(ValueError, match="sample_data must be a string"):
        GeneticData(genetic_data="tests/test_data/test.vcf",
                            sample_data=123)

    with pytest.raises(ValueError, match="test_size must be a float"):
        GeneticData(genetic_data="tests/test_data/test.vcf",
                            sample_data="tests/test_data/testNA.txt",
                            test_size="0.2")

    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        GeneticData(genetic_data="tests/test_data/test.vcf",
                            sample_data="tests/test_data/testNA.txt",
                            test_size=2.5)

    with pytest.raises(ValueError, match="seed must be an integer"):
        GeneticData(genetic_data="tests/test_data/test.vcf",
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
def test_classifier_inputs():
    
    data_obj = GeneticData(genetic_data="tests/test_data/test.vcf", 
                        sample_data="tests/test_data/testNA.txt")

    with pytest.raises(TypeError, match="data must be an instance of GeneticData"):
        PopClassifier(data=None)

    with pytest.raises(TypeError, match="random_state must be an integer"):
        PopClassifier(data_obj, random_state=0.5)

    with pytest.raises(TypeError, match="output_folder must be a string"):
        PopClassifier(data_obj, output_folder=123)

    with pytest.raises(ValueError, match="output_folder must be a valid directory"):
        PopClassifier(data_obj, output_folder="bad/path")

def test_classifier_train():

    data_obj = GeneticData(genetic_data="tests/test_data/test.vcf", 
                    sample_data="tests/test_data/testNA.txt")
    classifier = PopClassifier(data_obj, output_folder=TEST_OUTPUT_FOLDER)
    
    assert isinstance(classifier, PopClassifier)
    assert classifier.data.data.equals(data_obj.data)
    assert classifier.data.knowns.equals(data_obj.knowns)
    assert classifier.data.unknowns.equals(data_obj.unknowns)
    assert classifier.data.train.equals(data_obj.train)
    assert classifier.data.test.equals(data_obj.test)

    with pytest.raises(TypeError, match="epochs must be an integer"):
        classifier.train(epochs=0.5)

    with pytest.raises(TypeError, match="valid_size must be a float"):
        classifier.train(valid_size="0.2")

    with pytest.raises(ValueError, match="valid_size must be between 0 and 1"):
        classifier.train(valid_size=2.5)   

    with pytest.raises(TypeError, match="cv_splits must be an integer"):
        classifier.train(cv_splits="0.2")

    with pytest.raises(TypeError, match="cv_reps must be an integer"):
        classifier.train(cv_reps="0.2")

    with pytest.raises(TypeError, match="learning_rate must be a float"):
        classifier.train(learning_rate="0.2")

    with pytest.raises(ValueError, match="learning_rate must be between 0 and 1"):
        classifier.train(learning_rate=2.7)     

    with pytest.raises(TypeError, match="batch_size must be an integer"):
        classifier.train(batch_size=0.5)

    with pytest.raises(TypeError, match="dropout_prop must be a float"):
        classifier.train(dropout_prop="0.2")   

    with pytest.raises(ValueError, match="dropout_prop must be between 0 and 1"):
        classifier.train(dropout_prop=2) 

    classifier.train()
    assert isinstance(classifier.train_history, pd.DataFrame)
    assert classifier.train_history.empty == False
    assert isinstance(classifier.best_model, torch.nn.Module)

def test_classifier_test():

    data_obj = GeneticData(genetic_data="tests/test_data/test.vcf", 
                    sample_data="tests/test_data/testNA.txt")
    classifier = PopClassifier(data_obj, output_folder=TEST_OUTPUT_FOLDER)
    classifier.train()
    classifier.test()

    assert isinstance(classifier.test_results, pd.DataFrame)
    assert classifier.test_results.empty == False
    assert isinstance(classifier.confusion_matrix, np.ndarray)
    assert classifier.confusion_matrix.shape == (5,5)
    assert np.round(classifier.confusion_matrix.sum(), 1) == np.round(5.0, 1)
    assert isinstance(classifier.accuracy, float)
    assert classifier.accuracy > 0.0
    assert classifier.accuracy < 1.0
    assert isinstance(classifier.precision, float)
    assert classifier.precision > 0.0
    assert classifier.precision < 1.0
    assert isinstance(classifier.recall, float)
    assert classifier.recall > 0.0
    assert classifier.recall < 1.0
    assert isinstance(classifier.f1, float)
    assert classifier.f1 > 0.0
    assert classifier.f1 < 1.0

def test_classifier_assign_unknown_and_get_results():

    data_obj = GeneticData(genetic_data="tests/test_data/test.vcf", 
                    sample_data="tests/test_data/testNA.txt")
    classifier = PopClassifier(data_obj, output_folder=TEST_OUTPUT_FOLDER)
    classifier.train()
    classifier.test()
    unknown_data = classifier.assign_unknown()

    assert unknown_data.equals(classifier.classification)
    assert os.path.exists(os.path.join(classifier.output_folder,
                          "classifier_assignment_results.csv"))
    os.remove(os.path.join(classifier.output_folder,
                            "classifier_assignment_results.csv"))

    class_sum = classifier.get_classification_summary(save=False)
    assert isinstance(class_sum, pd.DataFrame)
    assert not os.path.exists(os.path.join(classifier.output_folder,
                              "classifier_classification_summary.csv"))

    classifier.get_classification_summary(save=True)
    assert os.path.exists(os.path.join(classifier.output_folder,
                          "classifier_classification_summary.csv"))
    os.remove(os.path.join(classifier.output_folder,
                            "classifier_classification_summary.csv"))

    site_rank = classifier.rank_site_importance()
    assert isinstance(site_rank, pd.DataFrame)
    assert site_rank.empty == False
    assert len(classifier.data.data.alleles[0]) == len(site_rank)

def test_classifier_save_and_load():

    data_obj = GeneticData(genetic_data="tests/test_data/test.vcf", 
                    sample_data="tests/test_data/testNA.txt")
    classifier = PopClassifier(data_obj, output_folder=TEST_OUTPUT_FOLDER)
    classifier.train()
    classifier.test()
    classifier.assign_unknown()
    classifier.save()

    assert os.path.exists(os.path.join(classifier.output_folder,
                          "classifier.pkl"))

    classifier2 = PopClassifier.load(load_path=os.path.join(classifier.output_folder,
                                "classifier.pkl"))

    assert classifier2.train_history.equals(classifier.train_history)
    assert classifier2.test_results.equals(classifier.test_results)
    assert np.array_equal(classifier2.confusion_matrix, classifier.confusion_matrix)
    assert classifier2.accuracy == classifier.accuracy
    assert classifier2.precision == classifier.precision
    assert classifier2.recall == classifier.recall
    assert classifier2.f1 == classifier.f1
    assert classifier2.classification.equals(classifier.classification)
    assert isinstance(classifier2.best_model, ClassifierNet)
    assert isinstance(classifier.best_model, ClassifierNet)

    os.remove(os.path.join(classifier.output_folder,
                            "classifier.pkl"))


# Test regressor class
# def test_regressor_inputs():

#     data_obj = GeneticData(genetic_data="tests/test_data/test.vcf", 
#                         sample_data="tests/test_data/testNA.txt")

#     with pytest.raises(TypeError, match="data must be an instance of GeneticData"):
#         PopRegressor(data=None)

#     with pytest.raises(TypeError, match="nboots must be an integer"):
#         PopRegressor(data=data_obj, nboots=0.5)

#     with pytest.raises(TypeError, match="random_state must be an integer"):
#         PopRegressor(data_obj, random_state=0.5)

#     with pytest.raises(TypeError, match="output_folder must be a string"):
#         PopRegressor(data_obj, output_folder=123)

#     with pytest.raises(ValueError, match="output_folder must be a valid directory"):
#         PopRegressor(data_obj, output_folder="bad/path")

# def test_regressor_train():

#     data_obj = GeneticData(genetic_data="tests/test_data/test.vcf", 
#                     sample_data="tests/test_data/testNA.txt")
#     regressor = PopRegressor(data_obj, output_folder=TEST_OUTPUT_FOLDER)
    
#     assert isinstance(regressor, PopRegressor)
#     assert regressor.data.data.equals(data_obj.data)
#     assert regressor.data.knowns.equals(data_obj.knowns)
#     assert regressor.data.unknowns.equals(data_obj.unknowns)
#     assert regressor.data.train.equals(data_obj.train)
#     assert regressor.data.test.equals(data_obj.test)

#     with pytest.raises(TypeError, match="epochs must be an integer"):
#         regressor.train(epochs=0.5)

#     with pytest.raises(TypeError, match="valid_size must be a float"):
#         regressor.train(valid_size="0.2")

#     with pytest.raises(ValueError, match="valid_size must be between 0 and 1"):
#         regressor.train(valid_size=2.5)   

#     with pytest.raises(TypeError, match="cv_splits must be an integer"):
#         regressor.train(cv_splits="0.2")

#     with pytest.raises(TypeError, match="cv_reps must be an integer"):
#         regressor.train(cv_reps="0.2")

#     with pytest.raises(TypeError, match="learning_rate must be a float"):
#         regressor.train(learning_rate="0.2")

#     with pytest.raises(ValueError, match="learning_rate must be between 0 and 1"):
#         regressor.train(learning_rate=2.7)     

#     with pytest.raises(TypeError, match="batch_size must be an integer"):
#         regressor.train(batch_size=0.5)

#     with pytest.raises(TypeError, match="dropout_prop must be a float"):
#         regressor.train(dropout_prop="0.2")   

#     with pytest.raises(ValueError, match="dropout_prop must be between 0 and 1"):
#         regressor.train(dropout_prop=2) 

#     with pytest.raises(TypeError, match="boot_data must be an instance of GeneticData"):
#         regressor.train(boot_data="0.2")

#     regressor.train()
#     assert isinstance(regressor.train_history, pd.DataFrame)
#     assert regressor.train_history.empty == False
#     assert isinstance(regressor.best_model, torch.nn.Module)

# def test_regressor_test():

#     data_obj = GeneticData(genetic_data="tests/test_data/test.vcf", 
#                     sample_data="tests/test_data/testNA.txt")
#     regressor = PopRegressor(data_obj, output_folder=TEST_OUTPUT_FOLDER)
#     regressor.train()
#     regressor.test()

#     assert isinstance(regressor.test_results, pd.DataFrame)
#     assert regressor.test_results.empty == False
#     assert isinstance(regressor.median_distance, float)
#     assert isinstance(regressor.mean_distance, float)
#     assert isinstance(regressor.r2_long, float)
#     assert isinstance(regressor.r2_lat, float)
#     assert isinstance(regressor.summary, dict)

# def test_regressor_assign_unknown_and_get_results():

#     data_obj = GeneticData(genetic_data="tests/test_data/test.vcf", 
#                     sample_data="tests/test_data/testNA.txt")
#     regressor = PopRegressor(data_obj, output_folder=TEST_OUTPUT_FOLDER)
#     regressor.train()
#     regressor.test()
#     unknown_data = regressor.assign_unknown()

#     assert unknown_data.equals(regressor.regression)
#     assert os.path.exists(os.path.join(regressor.output_folder,
#                           "regressor_assignment_results.csv"))
#     os.remove(os.path.join(regressor.output_folder,
#                             "regressor_assignment_results.csv"))

#     assign_sum = regressor.get_assignment_summary(save=False)
#     assert isinstance(assign_sum, dict)
#     assert not os.path.exists(os.path.join(regressor.output_folder,
#                               "regressor_classification_summary.csv"))

#     with pytest.raises(ValueError, match=re.escape("Must run classify_by_contours() before getting summary.")):
#         regressor.get_classification_summary(save=False)

#     site_rank = regressor.rank_site_importance()
#     assert isinstance(site_rank, pd.DataFrame)
#     assert site_rank.empty == False
#     assert len(regressor.data.data.alleles[0]) == len(site_rank)

# def test_regressor_classify_by_contours():

#     data_obj = GeneticData(genetic_data="tests/test_data/test.vcf", 
#                     sample_data="tests/test_data/testNA.txt")
#     regressor = PopRegressor(data_obj, output_folder=TEST_OUTPUT_FOLDER)

#     with pytest.raises(TypeError, match="nboots must be an integer"):
#         regressor.classify_by_contours(nboots=0.5)

#     with pytest.raises(TypeError, match="num_contours must be an integer"):
#         regressor.classify_by_contours(num_contours=0.5)

#     with pytest.raises(TypeError, match="save_plots must be a boolean"):
#         regressor.classify_by_contours(save_plots="True")

#     with pytest.raises(TypeError, match="save must be a boolean"):
#         regressor.classify_by_contours(save="True")

#     regressor.classify_by_contours()

#     assert isinstance(regressor.classification_test_results, pd.DataFrame)
#     assert regressor.classification_test_results.empty == False
#     assert isinstance(regressor.contour_classification, pd.DataFrame)
#     assert regressor.contour_classification.empty == False
#     assert isinstance(regressor.classification_confusion_matrix, np.ndarray)
#     assert isinstance(regressor.classification_accuracy, float)
#     assert isinstance(regressor.classification_precision, float)
#     assert isinstance(regressor.classification_recall, float)
#     assert isinstance(regressor.classification_f1, float)

#     class_sum = regressor.get_classification_summary(save=False)
#     assert isinstance(class_sum, dict)
#     assert class_sum["accuracy"] == regressor.classification_accuracy
#     assert class_sum["precision"] == regressor.classification_precision
#     assert class_sum["recall"] == regressor.classification_recall
#     assert class_sum["f1"] == regressor.classification_f1
#     assert (class_sum["confusion_matrix"] == regressor.classification_confusion_matrix).all()

# def test_regressor_save_and_load():

#     data_obj = GeneticData(genetic_data="tests/test_data/test.vcf", 
#                     sample_data="tests/test_data/testNA.txt")
#     regressor = PopRegressor(data_obj, output_folder=TEST_OUTPUT_FOLDER)
#     regressor.train()
#     regressor.test()
#     regressor.assign_unknown()
#     regressor.save()

#     assert os.path.exists(os.path.join(regressor.output_folder,
#                           "regressor.pkl"))

#     regressor2 = PopRegressor.load(load_path=os.path.join(regressor.output_folder,
#                                 "regressor.pkl"))

#     assert regressor2.train_history.equals(regressor.train_history)
#     assert regressor2.test_results.equals(regressor.test_results)
#     assert regressor2.median_distance == regressor.median_distance
#     assert regressor2.mean_distance == regressor.mean_distance
#     assert regressor2.r2_lat == regressor.r2_lat
#     assert regressor2.r2_long == regressor.r2_long
#     assert regressor2.regression.equals(regressor.regression)
#     assert isinstance(regressor2.best_model, RegressorNet)
#     assert isinstance(regressor.best_model, RegressorNet)

#     os.remove(os.path.join(regressor.output_folder,
#                             "regressor.pkl"))

#     regressor.classify_by_contours()
#     regressor.save()

#     regressor2 = PopRegressor.load(load_path=os.path.join(regressor.output_folder,
#                                 "regressor.pkl"))

#     assert regressor2.train_history.equals(regressor.train_history)
#     assert regressor2.test_results.equals(regressor.test_results)
#     assert regressor2.contour_classification.equals(regressor.contour_classification)
#     assert regressor2.classification_test_results.equals(regressor.classification_test_results)
#     assert regressor2.classification_accuracy == regressor.classification_accuracy
#     assert regressor2.classification_precision == regressor.classification_precision
#     assert regressor2.classification_recall == regressor.classification_recall
#     assert regressor2.classification_f1 == regressor.classification_f1
#     assert regressor2.classification_confusion_matrix.tolist() == regressor.classification_confusion_matrix.tolist()
#     assert isinstance(regressor2.best_model, RegressorNet)
#     assert isinstance(regressor.best_model, RegressorNet)

#     os.remove(os.path.join(regressor.output_folder,
#                             "regressor.pkl"))