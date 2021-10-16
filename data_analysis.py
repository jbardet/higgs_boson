import numpy as np
from proj1_helpers import *
import matplotlib.pyplot as plt
from visualisation import *

Train_path = '../data/train.csv'
Test_path = '../data/test.csv'
y, tX, ids = load_csv_data(Train_path)
#yt, tXt, idst = load_csv_data(Test_path)
nb_samples, nb_features = np.shape(tX)
#histogramme(tX,nb_features)

# -999 is too far(outlier) compared to the rest of the data, we can consider it a missing data
# we can replace these missing data by either median(better with outliers) or mean of the rest of the data

# Data cleaning, removing features that have too much missing_data(more than 75%)
# replacing missing data (-999) with the mean of the rest of the data
def modify_missing_data(X,missing_data,threshold):
    nb_features = (np.shape(X))[1]
    indices_missingdata = []
    percentage_missingdata = np.zeros(nb_features)
    indices_badfeatures = []
    mean = np.zeros(nb_features)
    #threshold_removing_feature = 0.70
    for i in range(nb_features):
        indices_missingdata.append(np.where(X[:,i] == missing_data))
        percentage_missingdata[i] = np.shape(indices_missingdata[i])[1]/nb_samples
        #print(percentage_missingdata[i])# we can see that a lot of features have a percentage of 0.709
        # we will remove the features that have a percentage of missing data more than 70%
        if percentage_missingdata[i]> threshold:
            indices_badfeatures.append(i)
             # we keep these features but we replace missing data with the mean
        else:
            mean[i] = np.mean(tX[:,i][tX[:,i] != missing_data])
            tX[indices_missingdata[i],i] = mean[i]
    # delete col of bad features
    new_X = np.delete(X,indices_badfeatures, 1)
    return new_X,indices_badfeatures
    #print(mean,indices_badfeatures)
#new_features = f(nb_f) -indices_badfeatures

#standardize data of each column
# mean or median?
def normalize(X):
    centred_x = X - np.mean(X)
    normalized_x = centred_x/np.std(X,axis=0)
    return normalized_x
'''
indices_missingdata = []
percentage_missingdata = np.zeros(nb_features)
for i in range(nb_features):
    indices_missingdata.append(np.where(tX[:,i] == -999))
    percentage_missingdata[i] = np.shape(indices_missingdata[i])[1]/nb_samples
    #print(indices_missingdata[i], np.shape(indices_missingdata[i])[1],percentage_missingdata[i])
    print(percentage_missingdata[i])
    # we can see that a lot of features have a percentage of 0.709 which is a lot

# we will remove the features that have a percentage of missing data more than 70%
indices_badfeatures = []
mean= np.zeros(nb_features)
threshold_removing_feature = 0.70
for i in range(nb_features):
    if percentage_missingdata[i]> threshold_removing_feature:
        indices_badfeatures.append(i)
         # we keep these features but we replace missing data with the mean
    else:
        mean[i] = np.mean(tX[:,i][tX[:,i] != -999])
        tX[indices_missingdata[i],i] = mean[i]

print(mean,indices_badfeatures)
#new_features = f(nb_f) -indices_badfeatures
standarized_xi=[[]]
#standardize data of each column
for i in range (nb_features):
    centred_xi = tX[:,i]-mean[i]
    stdi = np.std(tX[:,i])
    standarized_xi[:,i] = centred_xi/stdi
   '''
# testing the functions
new_X,indices_badfeatures = modify_missing_data(tX,-999,0.7)
print(new_X,indices_badfeatures)

new_nb_features = nb_features-len(indices_badfeatures)
# we shouldn't normalize good features like f12,14,15 17?, are they categorical features
normalized_x = normalize(new_X)
histogramme(tX,nb_features)
histogramme(normalized_x,new_nb_features)
#two_histogramme(new_X,normalized_x,new_nb_features)

# other ideas: remove f6 16/f10 13 (all/most data in same range?)

# covariance matrix but idk if we can do it with numpy: we can remove corelated features


