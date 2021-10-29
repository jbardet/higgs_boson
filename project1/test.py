import numpy as np
from scripts.proj1_helpers import *
from scripts.implementations import ridge_regression, logistic_regression, least_squares, least_squares_SGD,least_squares_GD
from roc_auc import roc

DATA_TRAIN_PATH = 'data/train.csv'
y, tx, ids = load_csv_data(DATA_TRAIN_PATH)

x_train, x_test, y_train, y_test = split_data(tx, y, 0.7)

max_iters = 100
gamma = 1e-3
lambda_ = 1e-4

x_train_cleaned, _, _ = modify_missing_data(x_train, -999, 0.9, x_train)
x_test_cleaned, _, _ = modify_missing_data(x_test, -999, 0.9, x_train)

x_train_cleaned_normalized, x_test_cleaned_normalized = standardize_cat(x_train_cleaned, x_test_cleaned)

x_train_onehot = one_hot(x_train_cleaned_normalized)
x_test_onehot = one_hot(x_test_cleaned_normalized)

degree = 11

x_train_poly = build_poly(x_train_onehot, degree)
x_test_poly = build_poly(x_test_onehot, degree)

initial_w = np.zeros(x_train_poly.shape[1])

## Test on some small data
weights, loss = ridge_regression(y_train, x_train_poly, lambda_)
y_pred = predict_labels(weights, x_test_poly)

roc(x_test_poly, y_test, weights, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

accuracy = np.sum(y_pred==y_test) / y_pred.shape[0]
print("Normalized + cleaned", loss, accuracy)
