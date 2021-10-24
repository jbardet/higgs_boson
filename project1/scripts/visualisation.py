import matplotlib.pyplot as plt
import numpy as np
from proj1_helpers import *


# function qui retourne le header: nom des feautures??
# a supprimer vu on peux pas use matlplotlib (c'est que pour la visualisation)
def histogramme(tX, nb_features):
    for i in range(nb_features):
        plt.hist(tX[:,i])
        plt.title('feature %d' %(i+1))
        plt.show()
def two_histogramme(tX,new_X,nb_features):
    for i in range(nb_features):
        plt.hist(tX[:,i])
        plt.hist(new_X[:,i])
        plt.title('feature %d' %(i+1))
        plt.show()
def plot_point(tX,name_features,ids):
    nb_feature=tX.shape[1]
    for i in range(nb_feature):
        #plt.plot(tX[:,i])
        # x c est l'ID, y c est value of the feature i
        plt.scatter(ids-ids[0],tX[:,i],10)
        #plt.title('name_feature', name_features[i] )
        plt.xlabel('numero point')
        plt.ylabel(name_features[i])
        plt.title(name_features[i])
        plt.show()


# f2,3,4,10,11,14,17,22,30 plein de 0
# f5,7,13,25,26,28,29 que -999 et 0
# f6,27 plein de -999
# f24 plein de 0/-999
# f9,20 que des 0
# f8,12,15,16,18,19,21,23 good (la plupart entre -3 et 3)
