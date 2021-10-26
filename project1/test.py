from operator import truediv
import numpy as np
from scripts.proj1_helpers import *
from scripts.implementations import logistic_regression, least_squares_GD, least_squares

DATA_TRAIN_PATH = 'data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = 'data/test.csv'
#_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = 'outputs/test.txt'

x_train, x_test, y_train, y_test = split_data(tX, y, 0.7)

def normalize(x_train, x_test):
    centred_x = x_train - np.mean(x_train, axis=0)
    normalized_x = centred_x / np.std(x_train, axis=0)
    return normalized_x, (x_test - np.mean(x_train, axis=0)) / np.std(x_train,axis=0)

x_train, x_test = normalize(x_train, x_test)

initial_w = np.zeros(tX.shape[1])
max_iters = 100
gamma = 1e-2

## Test on some small data
weights, loss = logistic_regression(y_train, x_train, initial_w, 100, gamma)
#weights, loss = least_squares(y_train, x_train)

print(loss)
y_pred = predict_labels_logistic(weights, x_test)
print(y_pred)

accuracy = np.sum(y_pred==y_test) / y_pred.shape[0]
print(accuracy)

#create_csv_submission(ids_test, y_pred, OUTPUT_PATH)