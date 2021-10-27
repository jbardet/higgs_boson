# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def predict_labels_logistic(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    w0 = np.ones((data.shape[0], 1))
    data = np.hstack((data, w0))
    y_pred = sigmoid(np.dot(data, weights))
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def normalize(x_train, x_test):
    centred_x = x_train - np.mean(x_train, axis=0)
    normalized_x = centred_x / np.std(x_train, axis=0)
    return normalized_x, (x_test - np.mean(x_train, axis=0)) / np.std(x_train,axis=0)

def split_data(x, y, ratio, seed=101):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def modify_missing_data(X,missing_data,threshold,train_X):
    nb_features = (np.shape(X))[1]
    nb_samples = X.shape[0]
    indices_missingdata = []
    percentage_missingdata = np.zeros(nb_features)
    indices_features = list(range(1, nb_features))
    indices_badfeatures = []
    median = np.zeros(nb_features)
    #threshold_removing_feature = 0.70
    for i in range(nb_features):
        indices_missingdata.append(np.where(X[:,i] == missing_data))
        percentage_missingdata[i] = np.shape(indices_missingdata[i])[1]/nb_samples
        #print(percentage_missingdata[i])# we can see that a lot of features have a percentage of 0.709
        # we will remove the features that have a percentage of missing data more than 70%
        if percentage_missingdata[i]> threshold:
            indices_badfeatures.append(i)
            indices_features.remove(i+1)
             # we keep these features but we replace missing data with the median(better for outliers) we can also use mean
        else:
            median[i] = np.median(train_X[:,i][train_X[:,i] != missing_data])
            # X can be either train or test data, in both cases we repalce missing data with median of the train data
            X[indices_missingdata[i],i] = median[i]
            #median[i] = np.median(X[:,i][X[:,i] != missing_data])
            #X[indices_missingdata[i],i] = median[i]

    # delete col of bad features
    new_X = np.delete(X,indices_badfeatures, 1)
    return new_X,indices_badfeatures,indices_features

def build_poly(tx, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    new_tx = []
    for i in range(tx.shape[1]):
        poly = np.ones((len(tx[:, i]), 1))

        for deg in range(1, degree+1):
            poly = np.c_[poly, np.power(tx[:, i], deg)]

        if i == 0:
            new_tx = poly[:,1:]
        else:
            new_tx = np.c_[new_tx, poly]

    return new_tx