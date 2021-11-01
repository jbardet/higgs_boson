# EPFL CS-433: Machine Learning Project 1 

## Description

In this project we tested different Machine Learning methods to classify a dataset from CERN where Higgs Boson are created. Futher informations about techniques used and data analysis can be found in the ML_project1 paper. We implemented the following methods : 
- Least Squares 
- Least Squares with Gradient Descent
- Least Squares with Stochastic Gradient Descent
- Ridge Regression
- Logistic Regression
- Regularized Logistic Regression

We found that the best model is least square using normal equations when paired with features augmented with a polynomial of degree 13. It achieved 82.4% accuracy and a F1 of 0.728 on the given test set. The submission platfrom to test accuracy of our methods is in AIcrowd platfrom (https://www.aicrowd.com/challenges/epfl-machine-learning-higgs).

## Install and use project 

To install and use our methods implemented in this project, you can clone the repository locally and functions are coded in the different files. The main code is run.py and it imports functions from other scripts to test functions and analyze data. You can also run the Jupyter notebook plots.ipynb to look at visualization of the data and results from the data analysis and classification. 

