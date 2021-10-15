import numpy as np

### LOGISTIC REGRESSION ###

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient_logistic(tx, y, w):

    h = sigmoid(np.matmul(tx.T, w))

    return 1 / tx.shape[0] * np.matmul(tx.T, h - y)


def cost_logistic(tx, y, w):
    
    h = sigmoid(np.matmul(tx.T, w))

    return 1 / tx.shape[0] * np.sum(np.matmul(-y, h) - np.matmul(1-y, h))

# Uses GD
def logistic_regression(y, tx, initial_w, max_iters, gamma):

    w = initial_w
    loss = 0

    for n_iter in range(max_iters):

        grad = gradient_logistic(tx, y, w)
        loss = cost_logistic(tx, y, w)

        w = w - gamma * grad

    return (w, loss)


### REGULARIZED LOGISTIC REGRESSION ###

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
    w = initial_w
    loss = 0

    for n_iter in range(max_iters):

        grad = gradient_logistic(tx, y, w)
        loss = cost_logistic(tx, y, w) + lambda_ * np.sum(np.squared(w)) # L2 regularization

        w = w - gamma * (grad + 2 * lambda_ * w)

    return (w, loss)