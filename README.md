# popfinder

The `popfinder` Python package performs genetic population assignment using neural networks. Using `popfinder`, you can load genetic information and sample information to train either a classifier neural network or a regressor neural network. A classifier neural network will try to identify the population of samples of unknown origin. The regressor neural network will try to identify latitudinal and longitudinal coordinates of samples of unknown origin. The regressor module comes with additional functionality that will perform classification of samples of unknown origin using kernel density estimates of predicted locations.

## Table of Contents

[Installation](#installation)

- [Dependencies](#dependencies)

- [Using conda](#using-conda)

- [Using pip](#using-pip)

[Usage](#usage)

- [Python IDE](#python-ide)

- [Command Line](#command-line)

[Reference](#reference)

## Installation

`popfinder` can be installed using either the `conda` or `pip` package managers. `conda` is a general package manager capable of installing packages from many sources, but `pip` is strictly a Python package manager. While the installation instructions below are based on a Windows 10 operating system, similar steps can be used to install `pysyncrosim` for Linux.

### Dependencies

`popfinder` was developed using **Python 3.10** and the following python packages:

```
numpy=1.24.0
pandas=1.5.2
pytorch=1.13.1
scikit-learn
dill=0.3.6
seaborn=0.12.1
matplotlib=3.6.2
scikit-allel
zarr=2.13.3
h5py=1.12.2
scipy=1.9.3
```

### Using conda

Follow these steps to get started with `conda` and use `conda` to install `popfinder`.

1. Install `conda` using the Miniconda or Anaconda installer (in this tutorial we use Miniconda). To install Miniconda, follow [this link](https://docs.conda.io/en/latest/miniconda.html) and under the **Latest Miniconda Installer Links**, download Miniconda for your operating system. Open the Miniconda installer and follow the default steps to install `conda`. For more information, see the [conda documentation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. To use `conda`, open the command prompt that was installed with the Miniconda installer. To find this prompt, type "anaconda prompt" in the **Windows Search Bar**. You should see an option appear called **Anaconda Prompt (miniconda3)**. Select this option to open a command line window. All code in the next steps will be typed in this window.

3. You can either install `popfinder` and its dependencies into your base environment, or set up a new `conda` environment (recommended). Run the code below to set up and activate a new `conda` environment called "popfinder_env" that uses Python 3.10.

```bash
# Create new conda environment
conda create -n popfinder_env python=3.10

# Activate environment
conda activate popfinder_env
```

You should now see that "(base)" has been replaced with "(popfinder_env)" at the beginning of each prompt.

4. Set the package channel for `conda`. To be able to install the dependencies for `popfinder`, you need to access the `conda-forge` package channel. To configure this channel, run the following code in the Anaconda Prompt.

```bash
# Set conda-forge package channel
conda config --add channels conda-forge
```

5. Install `pytorch` using `conda install`. The `pytorch` package is required by `popfinder`, but can only be installed using the `pytorch` conda channel.

```bash
# Install pytorch
conda install -c pytorch pytorch
```

6. Install `popfinder` using `conda install`. Installing `popfinder` will also install its dependencies.

```bash
# Install popfinder
conda install popfinder
```

`popfinder` should now be installed and ready to use!

### Using pip

Use `pip` to install `popfinder` to your default python installation. You can install Python from https://www.python.org/downloads/. You can also find information on how to install `pip` from the [pip documentation](https://pip.pypa.io/en/stable/installation/).

Install `popfinder` using `pip install`. Installing `popfinder` will also install its dependencies.

```bash
# Make sure you are using the latest version of pip
pip install --upgrade pip

# Install popfinder
pip install popfinder
```

## Usage

The following usage examples use the genetic data and sample data found in this [folder](tests\test_data). The data used in the following example is actual genomic data obtained from RAD-seq analysis of Leach's storm-petrels from 5 unique populations.

### Python IDE

#### Set Up

First, install `popfinder` using either `conda install` or `pip install`. See the installation instructions above for more information.

Then, in a new Python script, import the 2 classes of `popfinder`.

```python
from popfinder.dataloader import GeneticData
from popfinder.classifier import PopClassifier
```

#### Load Data

The `dataloader` module contains the `GeneticData` class. This class is used for loading all genetic data and sample data, as well as preprocessing the data in preparation for running the neural networks.

When creating a new instance of the `GeneticData` class, it must be initialized with a path to the `genetic_data` and a path to the `sample_data`. The genetic data can come in the form of a .vcf file, and contains allelic information for each sample. The sample data is a tab-delimited .txt file with the following columns: `pop`, and `sampleID`. The sample IDs in the .txt file must match the sample IDs in the genetic data file. If the sample is from an unknown location, then the `pop` column should have an `NA` value.

Run the below code to create an instance of the `GeneticData` class.

```python
data_object = GeneticData(genetic_data="tests/test_data/test.vcf",
                          sample_data="tests/test_data/testNA.txt")
```

Upon creating the `GeneticData` instance with the given data, the class will split the data into samples of known versus unknown origin, and of the samples of known origin, it will further split the data into a training and testing dataset. If there are no samples of unknown origin in the dataset, then the `data.unknowns` will return an empty dataframe. You can access these datasets using the following class attributes.

```python
# View all loaded data
data_object.data

# View data corresponding to samples of unknown origin
data_object.unknowns

# View data corresponding to samples of known origin
data_object.knowns

# View training dataset
data_object.train

# View testing dataset
data_object.test
```

#### Use the classifier module

The `classifier` module contains the `PopClassifier` class. This class is used for training a classifier neural network, using this neural network to perform population assignment, and visualizing the end results.

The only required argument for initializing an instance of this class is an instance of the `GeneticData` class. In our case, this instance is the `data_object` we created in the previous step.

Run the below code to create an instance of the `PopClassifier` class.

```python
classifier = PopClassifier(data_object)
```

Next, we will train our `classifier`. This will allow the neural network to learn our data so it can make more accurate predictions.

```python
classifier.train()
```

We can view the training history of our `classifier` using the below method. This will generate a plot that shows the loss of the neural network on the training data versus the loss on the validation data. A well-trained model should show converging loss values for the training and validation by the last epoch.

```python
classifier.plot_training_curve()
```

![image of training plot](https://github.com/ApexRMS/popfinder/blob/main/figures/classifier_training_history.png)

Once we are satisfied with the training of our model, we can use the `test()` method to evaluate our trained model.

```python
classifier.test()
```

We can visualize the accuracy, precision, and recall of the model by plotting a confusion matrix from the test results. The confusion matrix has the true population of origin along the Y-axis and the predicted population of origin along the X-axis. The scores along the diagonal represent the proportion of times samples from a given population were correctly assigned to that population.

```python
classifier.plot_confusion_matrix()
```

![image of confusion matrix](https://github.com/ApexRMS/popfinder/blob/main/figures/classifier_confusion_matrix.png)

Finally, we can use our trained and tested model to assign individuals of unknown origin to populations.

```python
classifier.assign_unknowns()
```

After running the above code, we can either display a dataframe or view a plot of assignment probabilities for each sample.

```python
classifier.plot_assignment()
```

![image of assignment plot](https://github.com/ApexRMS/popfinder/blob/main/figures/classifier_assignment_plot.png)

You can also retrieve information about which SNPs were most influential in training the model using the `rank_site_importance()` method. This method will return a dataframe containing information about each SNP and the corresponding error when the SNP is randomized during model training and validation. In the dataframe, SNPs that have a higher error value also have greater influence on the model, and by extension play a greater role in population assignment.

```python
classifier.rank_site_importance()
```

[Insert dataframe of SNP importance]

### Hyperparameter tuning

The `tuning` module contains the `hyperparam_search` function. This function is used to perform a grid search or a random search over a range of hyperparameters for the neural network. The function will train and test a model for each combination of hyperparameters and return a dataframe of performance metrics to help you select which hyperparameters to use for the final model.

First, load the `hyperparam_search` function from the `tuning` module.

```python
from popfinder.tuning import hyperparam_search
```

Next, decide which hyperparameters you would like to search over. You can choose values for the number of epochs, the learning rate, the dropout proportion, and the batch size.

```python
epoch_options = [10, 20, 30]
learning_options = [0.001, 0.01, 0.1]
dropout_options = [0.1, 0.2, 0.3]
batch_options = [32, 64, 128]
```

Then, run the `hyperparam_search` function. This function requires an instance of the `PopClassifier` class, and the hyperparameters you would like to search over. The function will return a dataframe of performance metrics for each combination of hyperparameters. If you would like to perform a random search instead of a grid search, you can set the `trials` argument to the number of random combinations of hyperparameters you would like to try.

```python
classifier = PopClassifier(data_object)
results = hyperparam_search(classifier, cv_splits=5, learning_rate=learning_options, 
                            dropout_prop=dropout_options, batch_size=batch_options, 
                            epochs=epoch_options)
```

Depending on the metric you would like to use to choose your model hyperparameters, you can search the dataframe for the best hyperparameters. For example, if you would like to choose the hyperparameters that maximize the Matthews Correlation Coefficient (MCC), you can use the following code.

```python
best_hyperparams = results.loc[results['valid_mcc'].idxmax()]
```

Finally, you are ready to train your model!
```python
classifier.train(epochs=best_hyperparams['epochs'], 
                 learning_rate=best_hyperparams['learning_rate'], 
                 dropout_prop=best_hyperparams['dropout_prop'], 
                 batch_size=best_hyperparams['batch_size'])

```


### Command Line

You can also run `popfinder` from the command line. To run the classifier from the command line, run the `pop_classifier` function. For a full list of methods and arguments for each function, run the `--help` command.

```bash
pop_classifier --help
```

The general workflow for using the command line version of `popfinder` is similar to using it in the Python IDE. At each step below, the updated model is loaded from and saved to the `output_folder`. If no `output_folder` is given, then a folder called `` will be created in the current working directory and all outputs will be saved to this folder.

1. Load the data.

```bash
pop_classifier --load_data --genetic_data="tests/test_data/test.vcf" --sample_data="tests/test_data/testNA.txt"
```

2. Train the model.

```bash
pop_classifier --train
```

3. Evaluate the model on the test dataset.

```bash
pop_classifier --test
```

4. Perform population assignment with the trained/tested model.

```bash
pop_classifier --assign
```

The output folder will contain results files, such as model evaluation statistics and a dataframe of sample population assignments. You can also generate plots based on the model results that will be saved to the output folder, such as the following:

```bash
pop_classifier --plot_assignment
```

## Reference

A reference containing all the available classes and methods for `popfinder` can be found on the [documentation website](https://popfinder.readthedocs.io/en/latest/).
