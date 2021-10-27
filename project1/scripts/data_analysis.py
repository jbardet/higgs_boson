import numpy as np
from proj1_helpers import *
import matplotlib.pyplot as plt
from visualisation import *

Train_path = '../data/train.csv'
Test_path = '../data/test.csv'
y, tX, ids = load_csv_data(Train_path)
name_features = load_namefeatures(Train_path)[2:]
#yt, tXt, idst = load_csv_data(Test_path)
nb_samples, nb_features = np.shape(tX)
#histogramme(tX,nb_features)

# -999 is too far(outlier) compared to the rest of the data, we can consider it a missing data
# we can replace these missing data by either median(better with outliers) or mean of the rest of the data

# Data cleaning, removing features that have too much missing_data(more than 75%)
# replacing missing data (-999) with the mean of the rest of the data
def modify_missing_data(X,missing_data,threshold,train_X):
    nb_features = (np.shape(X))[1]
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
    #print(mean,indices_badfeatures)
#new_features = f(nb_f) -indices_badfeatures

#standardize data of each column
# mean or median?
def normalize(X):
    centred_x = X - np.mean(X)
    normalized_x = centred_x/np.std(X,axis=0)
    return normalized_x

# testing the functions
new_X,indices_badfeatures,indice_features = modify_missing_data(tX,-999,0.7)
#print(new_X,indices_badfeatures)
#print(indice_features,indices_badfeatures)

#new_indice_features= indices_features-indices_badfeatures
#new_indice_features=list(np.array(indices_features) - np.array(indices_badfeatures))
#print(new_indice_features)


# we shouldn't normalize good features like f12,14,15 17?, are they categorical features
normalized_x = normalize(new_X)
#histogramme(tX,nb_features)
#histogramme(normalized_x,new_nb_features)
#two_histogramme(new_X,normalized_x,new_nb_features)

# other ideas: remove f6 16/f10 13 (all/most data in same range?)
# manuellement plot 7 13 20 condensens entre 0 et 250
#23 prend que val 0 1 2 3 categorique??

# covariance matrix but idk if we can do it with numpy: we can remove corelated features

def covaraince_matrix(X,indice_features,threshold_correlation):
    nb_features = X.shape[1]
    corelated_features=[]
    #for i in nb_features:

    C = np.corrcoef(X,rowvar=False)#each column represents a variable, while the rows contain observations.
    for i in range(nb_features):
        for j in range(i):
            if (np.absolute(C[i][j]) > threshold_correlation and i!=j):
             #if  abs(C[i:j])>threshold_correlation and i!=j:
                corelated_features.append((i+1,j+1))
    return C,corelated_features
#print(covaraince_matrix(new_X,indice_features),covaraince_matrix(new_X,indice_features).shape)
C,corelated_features = covaraince_matrix(new_X,indice_features,0.9)
print(0.9,corelated_features)
C,corelated_features = covaraince_matrix(new_X,indice_features,0.85)
print(0.85,corelated_features)
C,corelated_features = covaraince_matrix(new_X,indice_features,0.8)
print(0.8,corelated_features)
#print(C[2,:],corelated_features)
#print(np.corrcoef(new_X[:2,:2]).shape,np.corrcoef(new_X[:2,:2]))
# heatmap to visualise?
print(name_features)

plot_point(tX,name_features,ids)

# to add: features to normalise: all cleaned features - categorcal features like JETPRI_num
# do we have Nan values? if yes we have to clean them
# JETPRI_num is a categorical features (maybe the only one) seperate data,test,train data corresponding to JETPRI_num={0 1 2 3}
