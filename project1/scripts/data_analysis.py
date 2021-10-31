import numpy as np
from proj1_helpers import *
import matplotlib.pyplot as plt
from visualisation import *

Train_path = '../data/train.csv'
Test_path = '../data/test.csv'
y, tX, ids = load_csv_data(Train_path)
name_features = load_namefeatures(Train_path)[2:]
nb_samples, nb_features = np.shape(tX)


# -999 is too far(outlier) compared to the rest of the data, we can consider it a missing data
# we can replace these missing data by either median(better with outliers) or mean of the rest of the data

# Data cleaning, removing features that have too much missing_data(more than 70%)
# replacing missing data (-999) with the median of the rest of the data
def modify_missing_data(X,missing_data,threshold,train_X):
    nb_features = (np.shape(X))[1]
    indices_missingdata = []
    percentage_missingdata = np.zeros(nb_features)
    indices_features = list(range(1, nb_features))
    indices_badfeatures = []
    median = np.zeros(nb_features)

    for i in range(nb_features):
        indices_missingdata.append(np.where(X[:,i] == missing_data))
        percentage_missingdata[i] = np.shape(indices_missingdata[i])[1]/nb_samples

         # we will remove the features that have a percentage of missing data more than threshold
        if percentage_missingdata[i]> threshold:
            indices_badfeatures.append(i)
            indices_features.remove(i+1)
             # we keep these features but we replace missing data with the median(better for outliers)
        else:
            median[i] = np.median(train_X[:,i][train_X[:,i] != missing_data])
            # X can be either train or test data, in both cases we repalce missing data with median of the train data
            X[indices_missingdata[i],i] = median[i]
    # delete col of bad features
    new_X = np.delete(X,indices_badfeatures, 1)
    return new_X,indices_badfeatures,indices_features


#standardize data of each column

def normalize(X):
    centred_x = X - np.mean(X)
    normalized_x = centred_x/np.std(X,axis=0)
    return normalized_x


def covaraince_matrix(X,indice_features,threshold_correlation):
    nb_features = X.shape[1]
    corelated_features=[]

    C = np.corrcoef(X,rowvar=False)#each column represents a variable, while the rows contain observations.
    for i in range(nb_features):
        for j in range(i):
            if (np.absolute(C[i][j]) > threshold_correlation and i!=j):
             #if  abs(C[i:j])>threshold_correlation and i!=j:
                corelated_features.append((i+1,j+1))
    return C,corelated_features


