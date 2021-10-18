from operator import truediv
import numpy as np
from scripts.proj1_helpers import *
from scripts.implementations import logistic_regression

DATA_TRAIN_PATH = 'data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = 'data/test.csv'
#_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = 'outputs/test.txt'

initial_w = np.zeros(tX.shape[1])
max_iters = 100
gamma = 1e-3

## Test on some small data
weights, loss = logistic_regression(y[:30000], tX[:30000], initial_w, max_iters, gamma)

print(loss)
y_pred = predict_labels(weights, tX[30000:])
print(np.dot(tX[30000:], weights))

accuracy = np.sum(y_pred==y[30000:]) / y_pred.shape[0]

print(accuracy)

#create_csv_submission(ids_test, y_pred, OUTPUT_PATH)