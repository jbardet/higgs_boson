import numpy as np
from proj1_helpers import *
from implementations import loss_logistic, logistic_regression, reg_logistic_regression, least_squares_GD, least_squares_SGD, ridge_regression


def gamma_grid_search(y, tx, gammas, k_fold, seed=101):

    k_indices = build_k_indices(y, k_fold, seed)

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
            x_test = tx[test_indice]
            x_train = tx[train_indice]
            x_train, x_test = normalize(x_train, x_test)
            initial_w = np.zeros(x_train.shape[1])

            # for log reg
            w0 = np.ones((x_test.shape[0], 1))
            x_test_reg = np.hstack((x_test, w0))

            w, loss_train = logistic_regression(y_train, x_train, initial_w, 50, gamma)
            loss_test = loss_logistic(x_test_reg, y_test, w)
            loss_logreg_tmp.append([loss_train, loss_test])

            w, loss_train = least_squares_GD(y_train, x_train, initial_w, 50, gamma)
            loss_test = compute_mse(y_test, x_test, w) #mse loss
            loss_ls_GD_tmp.append([loss_train, loss_test])

            w, loss_train = least_squares_SGD(y_train, x_train, initial_w, 50, gamma)
            loss_test = compute_mse(y_test, x_test, w) #mse loss
            loss_ls_SGD_tmp.append([loss_train, loss_test])

        loss_logreg.append(np.mean(loss_logreg_tmp, axis=0))
        loss_ls_GD.append(np.mean(loss_ls_GD_tmp, axis=0))
        loss_ls_SGD.append(np.mean(loss_ls_SGD_tmp, axis=0))

        iter += 1

    return loss_logreg, loss_ls_GD, loss_ls_SGD

### PENALTY PARAMETER ###
def lambda_grid_search(y, tx, lambdas, k_fold, seed=101):

    k_indices = build_k_indices(y, k_fold, seed)
    gamma = 1e-5 #Change to best gamma found before

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
            x_test = tx[test_indice]
            x_train = tx[train_indice]
            x_train, x_test = normalize(x_train, x_test)
            initial_w = np.zeros(x_train.shape[1])

            # for log reg
            w0 = np.ones((x_test.shape[0], 1))
            x_test_reg = np.hstack((x_test, w0))

            w, loss_train = reg_logistic_regression(y_train, x_train, lambda_, initial_w, 50, gamma)
            loss_test = loss_logistic(x_test_reg, y_test, w)
            loss_logreg_reg_tmp.append([loss_train, loss_test])

            w, loss_train = ridge_regression(y_train, x_train, lambda_)
            loss_test = compute_mse(y_test, x_test, w) #mse loss
            loss_ridge_tmp.append([loss_train, loss_test])

        loss_logreg_reg.append(np.mean(loss_logreg_reg_tmp, axis=0))
        loss_ridge.append(np.mean(loss_ridge_tmp, axis=0))

        iter += 1
    
    return loss_logreg_reg, loss_ridge