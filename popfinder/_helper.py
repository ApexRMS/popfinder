import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn import preprocessing
import numpy as np
import dill
import os

def _generate_train_inputs(data_obj, valid_size, cv_splits, cv_reps, seed=123):

    if cv_splits == 1:
        train_input, valid_input = data_obj.split_train_test(data_obj.train, test_size=valid_size, seed=seed)
        inputs = [(train_input, valid_input)]

    elif cv_splits > 1:
        inputs = data_obj.split_kfcv(data_obj.train, n_splits=cv_splits, n_reps=cv_reps, seed=seed)

    return inputs

def _split_input_classifier(clf, input):
        
    train_input, valid_input = input

    X_train = train_input["alleles"]
    X_valid = valid_input["alleles"]
    y_train = train_input["pop"] # one hot encode
    y_valid = valid_input["pop"] # one hot encode

    # Label encode y values
    clf.label_enc = preprocessing.LabelEncoder()
    y_train = clf.label_enc.fit_transform(y_train)
    y_valid = clf.label_enc.transform(y_valid)

    X_train, y_train = _data_converter(X_train, y_train)
    X_valid, y_valid = _data_converter(X_valid, y_valid)

    return X_train, y_train, X_valid, y_valid

def _split_input_regressor(input):
        
    train_input, valid_input = input

    X_train = train_input["alleles"]
    X_valid = valid_input["alleles"]
    y_train = train_input[["x", "y"]]
    y_valid = valid_input[["x", "y"]]

    X_train, y_train = _data_converter(X_train, y_train)
    X_valid, y_valid = _data_converter(X_valid, y_valid)

    return X_train, y_train, X_valid, y_valid

def _generate_data_loaders(X_train, y_train, X_valid, y_valid, batch_size=16):

    train = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader

def _data_converter(x, y, variable=False):

    features = torch.from_numpy(np.vstack(np.array(x)).astype(np.float32))
    if torch.isnan(features).sum() != 0:
        print("Remove NaNs from features")        
    if variable:
        features = Variable(features)

    if y is not None:
        targets = torch.from_numpy(np.vstack(np.array(y)))
        if torch.isnan(targets).sum() != 0:
            print("remove NaNs from target")
        if variable:
            targets = Variable(targets)
            
        return features, targets

    else:
        return features

def _save(obj, save_path=None, file="model.pkl"):
    """
    Saves the current instance of the class to a pickle file.
    """
    if save_path is None:
        save_path = obj.output_folder
    with open(os.path.join(save_path, file), "wb") as f:
        dill.dump(obj, f)

def _load(load_path=None):
    """
    Loads a saved instance of the class from a pickle file.
    """
    with open(load_path, "rb") as f:
        return dill.load(f)