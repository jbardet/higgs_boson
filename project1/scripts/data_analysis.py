import numpy as np
from proj1_helpers import *
import matplotlib.pyplot as plt
from visualisation import *

Train_path = '../data/train.csv'
Test_path = '../data/test.csv'
y, tX, ids = load_csv_data(Train_path)
name_features = load_namefeatures(Train_path)[2:]
nb_samples, nb_features = np.shape(tX)

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
