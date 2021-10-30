import numpy as np
from proj1_helpers import standardize_cat, build_k_indices, compute_mse, modify_missing_data, predict_labels, predict_labels_logistic, one_hot, build_poly, load_csv_data
from implementations import loss_logistic, logistic_regression, reg_logistic_regression, least_squares_GD, least_squares_SGD, least_squares, ridge_regression

def gamma_grid_search(y, tx, gammas, k_fold, degree=1, seed=101):
    """Performs grid search of gamma parameter using k-fold validation
       Only for logistic_regression, least_squares_GD, least_squares_SGD
       OUTPUTS : mean the losses of each method for each gamma"""

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
            train_indice = k_indices[np.arange(k_indices.shape[0]) != k].reshape(-1)

            y_test = y[test_indice]
            y_train = y[train_indice]
            x_test = tx[test_indice]
            x_train = tx[train_indice]

            x_train_cleaned, _, _ = modify_missing_data(x_train, -999, 0.9, x_train)
            x_test_cleaned, _, _ = modify_missing_data(x_test, -999, 0.9, x_train)

            x_train_cleaned_normalized, x_test_cleaned_normalized = standardize_cat(x_train_cleaned, x_test_cleaned)

            x_train_onehot = one_hot(x_train_cleaned_normalized)
            x_test_onehot = one_hot(x_test_cleaned_normalized)

            x_train_poly = build_poly(x_train_onehot, degree)
            x_test_poly = build_poly(x_test_onehot, degree)

            initial_w = np.zeros(x_train_poly.shape[1])

            # for log reg add offset term w0
            w0 = np.ones((x_test_poly.shape[0], 1))
            x_test_reg = np.hstack((x_test_poly, w0))

            w, loss_train = logistic_regression(y_train, x_train_poly, initial_w, 100, gamma)
            loss_test = loss_logistic(x_test_reg, y_test, w)
            loss_logreg_tmp.append([loss_train, loss_test])

            w, loss_train = least_squares_GD(y_train, x_train_poly, initial_w, 100, gamma)
            loss_test = compute_mse(y_test, x_test_poly, w) #mse loss
            loss_ls_GD_tmp.append([loss_train, loss_test])

            w, loss_train = least_squares_SGD(y_train, x_train_poly, initial_w, 100, gamma)
            loss_test = compute_mse(y_test, x_test_poly, w) #mse loss
            loss_ls_SGD_tmp.append([loss_train, loss_test])

        loss_logreg.append(np.mean(loss_logreg_tmp, axis=0))
        loss_ls_GD.append(np.mean(loss_ls_GD_tmp, axis=0))
        loss_ls_SGD.append(np.mean(loss_ls_SGD_tmp, axis=0))

        iter += 1

    return loss_logreg, loss_ls_GD, loss_ls_SGD

### PENALTY PARAMETER ###
def lambda_grid_search(y, tx, lambdas, k_fold, degree=1, seed=101):
    """Performs grid search of lambda parameter using k-fold validation
       Only for ridge_regression and reg_logistic_regression
       OUTPUTS : the mean losses of each method for each lambda"""

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
            train_indice = k_indices[np.arange(k_indices.shape[0]) != k].reshape(-1)

            y_test = y[test_indice]
            y_train = y[train_indice]
            x_test = tx[test_indice]
            x_train = tx[train_indice]

            x_train_cleaned, _, _ = modify_missing_data(x_train, -999, 0.9, x_train)
            x_test_cleaned, _, _ = modify_missing_data(x_test, -999, 0.9, x_train)

            x_train_cleaned_normalized, x_test_cleaned_normalized = standardize_cat(x_train_cleaned, x_test_cleaned)

            x_train_onehot = one_hot(x_train_cleaned_normalized)
            x_test_onehot = one_hot(x_test_cleaned_normalized)

            x_train_poly = build_poly(x_train_onehot, degree)
            x_test_poly = build_poly(x_test_onehot, degree)

            initial_w = np.zeros(x_train_poly.shape[1])

            # for log reg add offset term w0
            w0 = np.ones((x_test_poly.shape[0], 1))
            x_test_reg = np.hstack((x_test_poly, w0))

            w, loss_train = reg_logistic_regression(y_train, x_train_poly, lambda_, initial_w, 100, gamma)
            loss_test = loss_logistic(x_test_reg, y_test, w)
            loss_logreg_reg_tmp.append([loss_train, loss_test])

            w, loss_train = ridge_regression(y_train, x_train_poly, lambda_)
            loss_test = compute_mse(y_test, x_test_poly, w) #mse loss
            loss_ridge_tmp.append([loss_train, loss_test])

        loss_logreg_reg.append(np.mean(loss_logreg_reg_tmp, axis=0))
        loss_ridge.append(np.mean(loss_ridge_tmp, axis=0))

        iter += 1
    
    return loss_logreg_reg, loss_ridge


def compare_models(y, tx, gammas, lambdas, k_fold, degree=1, seed=101):
    """Performs k-fold cross validation of the methods using the accuracy metric
       OUTPUTS : the mean accuracy of each model"""

    k_indices = build_k_indices(y, k_fold, seed)

    acc_ls_GD = []
    acc_ls_SGD = []
    acc_ls = []
    acc_ridge = []
    acc_logreg = []
    acc_logreg_reg = []

    for k in range(k_fold):

        test_indice = k_indices[k]
        train_indice = k_indices[np.arange(k_indices.shape[0]) != k].reshape(-1)

        y_test = y[test_indice]
        y_train = y[train_indice]
        x_test = tx[test_indice]
        x_train = tx[train_indice]

        x_train_cleaned, _, _ = modify_missing_data(x_train, -999, 0.9, x_train)
        x_test_cleaned, _, _ = modify_missing_data(x_test, -999, 0.9, x_train)

        x_train_cleaned_normalized, x_test_cleaned_normalized = standardize_cat(x_train_cleaned, x_test_cleaned)

        x_train_onehot = one_hot(x_train_cleaned_normalized)
        x_test_onehot = one_hot(x_test_cleaned_normalized)

        x_train_poly = build_poly(x_train_onehot, degree)
        x_test_poly = build_poly(x_test_onehot, degree)

        initial_w = np.zeros(x_train_poly.shape[1])

        weights, loss = least_squares_GD(y_train, x_train_poly, initial_w, 100, gammas[2])
        y_pred = predict_labels(weights, x_test_poly)
        accuracy = np.sum(y_pred==y_test) / y_pred.shape[0]
        acc_ls_GD.append(accuracy)

        weights, loss = least_squares_SGD(y_train, x_train_poly, initial_w, 100, gammas[3])
        y_pred = predict_labels(weights, x_test_poly)
        accuracy = np.sum(y_pred==y_test) / y_pred.shape[0]
        acc_ls_SGD.append(accuracy)

        weights, loss = least_squares(y_train, x_train_poly)
        y_pred = predict_labels(weights, x_test_poly)
        accuracy = np.sum(y_pred==y_test) / y_pred.shape[0]
        acc_ls.append(accuracy)

        weights, loss = ridge_regression(y_train, x_train_poly, lambdas[1])
        y_pred = predict_labels(weights, x_test_poly)
        accuracy = np.sum(y_pred==y_test) / y_pred.shape[0]
        acc_ridge.append(accuracy)

        weights, loss = logistic_regression(y_train, x_train_poly, initial_w, 100, gammas[1])
        y_pred = predict_labels_logistic(weights, x_test_poly)
        accuracy = np.sum(y_pred==y_test) / y_pred.shape[0]
        acc_logreg.append(accuracy)

        weights, loss = reg_logistic_regression(y_train, x_train_poly, lambdas[0], initial_w, 100, gammas[1])
        y_pred = predict_labels_logistic(weights, x_test_poly)
        accuracy = np.sum(y_pred==y_test) / y_pred.shape[0]
        acc_logreg_reg.append(accuracy)


    #return np.mean(acc_logreg, axis=0), np.mean(acc_logreg_reg, axis=0), np.mean(acc_ls_GD, axis=0), np.mean(acc_ls_SGD, axis=0), np.mean(acc_ls, axis=0), 
    return np.mean(acc_ls_GD, axis=0), np.mean(acc_ls_SGD, axis=0), np.mean(acc_ls, axis=0), np.mean(acc_ridge, axis=0), np.mean(acc_logreg, axis=0), np.mean(acc_logreg_reg, axis=0)

# Grid search for polynomial degree (ridge only)
def pol_degree_grid_search(y, tx, k_fold, degrees, seed=101):
    """Performs grid search of polynomial degrees using k-fold validation
       Only for least_squares and ridge_regression
       OUTPUTS : the mean accuracy for each polynomial basis"""

    k_indices = build_k_indices(y, k_fold, seed)

    acc_ls = []
    acc_ridge = []

    lambda_ = 1e-4

    for k in range(k_fold):

        test_indice = k_indices[k]
        train_indice = k_indices[np.arange(k_indices.shape[0]) != k].reshape(-1)

        y_test = y[test_indice]
        y_train = y[train_indice]
        x_test = tx[test_indice]
        x_train = tx[train_indice]

        x_train_cleaned, _, _ = modify_missing_data(x_train, -999, 0.9, x_train)
        x_test_cleaned, _, _ = modify_missing_data(x_test, -999, 0.9, x_train)

        x_train_cleaned_normalized, x_test_cleaned_normalized = standardize_cat(x_train_cleaned, x_test_cleaned)

        x_train_onehot = one_hot(x_train_cleaned_normalized)
        x_test_onehot = one_hot(x_test_cleaned_normalized)

        acc_ls_tmp = []
        acc_ridge_tmp = []

        for degree in degrees:

            x_train_poly = build_poly(x_train_onehot, degree)
            x_test_poly = build_poly(x_test_onehot, degree)

            weights, loss = least_squares(y_train, x_train_poly)
            y_pred = predict_labels(weights, x_test_poly)
            accuracy = np.sum(y_pred==y_test) / y_pred.shape[0]
            acc_ls_tmp.append(accuracy)
            
            weights, loss = ridge_regression(y_train, x_train_poly, lambda_)
            y_pred = predict_labels(weights, x_test_poly)
            accuracy = np.sum(y_pred==y_test) / y_pred.shape[0]
            acc_ridge_tmp.append(accuracy)

        acc_ls.append(acc_ls_tmp)
        acc_ridge.append(acc_ridge_tmp)
    
    return np.mean(acc_ls, axis=0), np.mean(acc_ridge, axis=0)