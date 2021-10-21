import numpy as np
from proj1_helpers import *
from implementations import loss_logistic, logistic_regression, reg_logistic_regression, least_squares_GD, least_squares_SGD, ridge_regression
from plots import cross_validation_visualization

DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

def normalize(x_train, x_test):
    centred_x = x_train - np.mean(x_train, axis=0)
    normalized_x = centred_x / np.std(x_train, axis=0)
    return normalized_x, (x_test - np.mean(x_train, axis=0)) / np.std(x_train,axis=0)

seed = 101
gammas = np.logspace(-4, 0, 10)
k_fold = 4
k_indices = build_k_indices(y, k_fold, seed)

# Losses on test set
loss_logreg = []
loss_ls_GD = []
loss_ls_SGD = []

iter = 1

for gamma in gammas:
    loss_logreg_tmp = []
    loss_ls_GD_tmp = []
    loss_ls_SGD_tmp = []

    print(iter)

    for k in range(k_fold):
        test_indice = k_indices[k]
        train_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)
        y_test = y[test_indice]
        y_train = y[train_indice]
        x_test = tX[test_indice]
        x_train = tX[train_indice]
        x_train, x_test = normalize(x_train, x_test)
        initial_w = np.zeros(x_train.shape[1])

        # for log reg
        w0 = np.ones((x_test.shape[0], 1))
        x_test_reg = np.hstack((x_test, w0))

        w, _ = logistic_regression(y_train, x_train, initial_w, 50, gamma)
        loss = loss_logistic(x_test_reg, y_test, w)
        loss_logreg_tmp.append(loss)

        w, _ = least_squares_GD(y_train, x_train, initial_w, 50, gamma)
        loss = np.sqrt(2 * compute_mse(y_test, x_test, w)) #rmse loss
        loss_ls_GD_tmp.append(loss)

        w, _ = least_squares_SGD(y_train, x_train, initial_w, 50, gamma)
        loss = np.sqrt(2 * compute_mse(y_test, x_test, w)) #rmse loss
        loss_ls_SGD_tmp.append(loss)

    loss_logreg.append(np.mean(loss_logreg_tmp))
    loss_ls_GD.append(np.mean(loss_ls_GD_tmp))
    loss_ls_SGD.append(np.mean(loss_ls_SGD_tmp))

    iter += 1

folder_path = "../figs/"

cross_validation_visualization(gammas, loss_logreg, "logistic_regression", folder_path)
cross_validation_visualization(gammas, loss_ls_GD, "least_squares_GD", folder_path)
cross_validation_visualization(gammas, loss_ls_SGD, "least_squares_SGD", folder_path)


### PENALTY PARAMETER ###

gamma_logreg = 1e-2 #Change to best gamma found before
lambdas = np.linspace(0, 2, 20)

loss_logreg_reg = []
loss_ridge = []

iter = 1

for lambda_ in lambdas:
    loss_logreg_reg_tmp = []
    loss_ridge_tmp = []

    print(iter)

    for k in range(k_fold):

        test_indice = k_indices[k]
        train_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)
        y_test = y[test_indice]
        y_train = y[train_indice]
        x_test = tX[test_indice]
        x_train = tX[train_indice]
        x_train, x_test = normalize(x_train, x_test)
        initial_w = np.zeros(x_train.shape[1])

        # for log reg
        w0 = np.ones((x_test.shape[0], 1))
        x_test_reg = np.hstack((x_test, w0))

        w, _ = reg_logistic_regression(y_train, x_train, lambda_, initial_w, 50, gamma)
        loss = loss_logistic(x_test, y_test, w)
        loss_logreg_reg_tmp.append(loss)

        w, _ = ridge_regression(y_train, x_train, lambda_)
        loss = np.sqrt(2 * compute_mse(y_test, x_test, w)) #rmse loss
        loss_ridge_tmp.append(loss)

    loss_logreg_reg.append(np.mean(loss_logreg_reg_tmp))
    loss_ridge.append(np.mean(loss_ridge_tmp))

    iter += 1

cross_validation_visualization(lambdas, loss_logreg_reg, "reg_logistic_regression", folder_path, "lambda")
cross_validation_visualization(lambdas, loss_ridge, "ridge_regression", folder_path, "lambda")