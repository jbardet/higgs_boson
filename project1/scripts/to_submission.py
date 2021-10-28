from proj1_helpers import *
from implementations import ridge_regression


DATA_TRAIN_PATH = '../data/train.csv'
y, tx, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = '../data/test.csv'
_, tx_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = '../outputs/test.txt'

lambda_ = 1e-4

x_train_cleaned, _, _ = modify_missing_data(tx, -999, 0.9, tx)
x_test_cleaned, _, _ = modify_missing_data(tx_test, -999, 0.9, tx)

x_train_cleaned_normalized, x_test_cleaned_normalized = standardize_cat(x_train_cleaned, x_test_cleaned)

x_train_onehot = one_hot(x_train_cleaned_normalized)
x_test_onehot = one_hot(x_test_cleaned_normalized)

degree = 11

x_train_poly = build_poly(x_train_onehot, degree)
x_test_poly = build_poly(x_test_onehot, degree)

weights, loss = ridge_regression(y, x_train_poly, lambda_)
y_pred = predict_labels(weights, x_test_poly)


create_csv_submission(ids_test, y_pred, OUTPUT_PATH)