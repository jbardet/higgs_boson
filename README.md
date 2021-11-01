# EPFL CS-433: Machine Learning Project 1 

## Description

In this project we tested different Machine Learning methods to classify a dataset from CERN where Higgs Boson are created. Futher informations about techniques used and data analysis can be found in the ML_project1 paper. 

For data analysis we implemented the following functions in the data_analysis.py :
- normalize
- covaraince_matrix

In the param_selection.py we implemented the functions to select best parameters : 
- gamma_grid_search : learning rate
- lambda_grid_search : penalty parameter
- compare_models : k-fold cross validation 
- pol_degree_grid_search : grid search of polynomial degrees using k-fold validation

In roc_auc.py there is the computation of the Receiver Operating Characteristic (ROC) curve and its area under the curve (AUC)

In visualization.py there is some functions helping to visualize data and understand it. 

All other helpers functions are in proj1_helpers.py :
- modify_missing_data : take care of -999 values in dataset
- normalization and standardization 
- splitting data : split data in train / test sets
- build_poly : build polynomial basis
- load_csv_data : load train and test CSV files
- one_hot : creates one_hot encoding of categorical feature
- predict_labels : predict labels for logistic or linear regression

Finally, we implemented the following methods in implementations.py : 
- Least Squares 
- Least Squares with Gradient Descent
- Least Squares with Stochastic Gradient Descent
- Ridge Regression
- Logistic Regression
- Regularized Logistic Regression

We found that the best model is least square using normal equations when paired with features augmented with a polynomial of degree 13. It achieved 82.4% accuracy and a F1 of 0.728 on the given test set. The submission platfrom to test accuracy of our methods is in AIcrowd platfrom (https://www.aicrowd.com/challenges/epfl-machine-learning-higgs).

## Install and use project 

To install and use our methods implemented in this project, you can clone the repository locally and functions are coded in the different files. The main code used for submission is run.py and it imports functions from other scripts to test functions and analyze data. You can also run the Jupyter notebook plots.ipynb to look at visualization of the data and results from the data analysis and classification. 
