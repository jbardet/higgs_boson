import numpy as np

### LOGISTIC REGRESSION ###

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


""" USE -1 and 1 LABELS
# compute gradient
def stoch_gradient_logistic(tx, y, w, i):

    h = sigmoid(-y[i] * np.dot(tx[i], w))

    return -y[i] * tx[i] * h

def gradient_logistic(tx, y, w):

    n = tx.shape[0]
    gradfx = np.zeros(w.shape)

    for i in range(n):
        gradfx += stoch_gradient_logistic(tx, y, w, i)

    return gradfx/n

## USE CROSS ENTROPY https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression
def cost_logistic(tx, y, w):

    y[y < 0] = 0

    h = sigmoid(np.matmul(tx, w))
    epsilon = 1e-5
    cost = (1 / y.shape[0]) * np.matmul(((-y).T, np.log(h + epsilon)) - np.matmul((1-y).T, np.log(1-h + epsilon)))
    return cost
""" 

# Uses 0 and 1 labels
def gradient_logistic(tx, y, w):
   return 1 / y.shape[0] *  np.matmul(tx.T, (sigmoid(np.matmul(tx, w)) - y)) 

def cost_logistic(tx, y, w):

    h = sigmoid(np.matmul(tx, w))
    epsilon = 1e-5

    return 1 / y.shape[0] * (np.matmul((-y).T, np.log(h + epsilon)) - np.matmul((1-y).T, np.log(1-h + epsilon)))

# Uses GD
def logistic_regression(y, tx, initial_w, max_iters, gamma):

    y[y < 0] = 0

    w = initial_w
    loss = 0

    for n_iter in range(max_iters):

        #i = np.random.random_integers(low=0, high=y.shape[0])

        grad = gradient_logistic(tx, y, w)
        loss = cost_logistic(tx, y, w)

        w = w - gamma * grad

    return w, loss


### REGULARIZED LOGISTIC REGRESSION ###

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
    w = initial_w
    loss = 0

    for n_iter in range(max_iters):

        grad = gradient_logistic(tx, y, w)
        loss = cost_logistic(tx, y, w) + lambda_ * np.sum(np.squared(w)) # L2 regularization

        w = w - gamma * (grad + 2 * lambda_ * w)

    return w, loss