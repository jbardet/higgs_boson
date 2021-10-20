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

def least_squares_GD(y, tx, w_initial, max_iters, gamma):
    """Compute Least Squares with Gradient Descent"""
    """INPUTS : vector with data (tx, y), the initial w, the maximum number of iterations and the learning rate gamma"""
    """OUTPUTS : the weights w of the model and the loss"""
    N = len(y)
    w = w_initial
    for i in range(max_iters) :
        mse_grad_loss = (-1/N)*np.matmul(tx.T, (y-np.dot(tx,w)))
        w = w - gamma*mse_grad_loss
    loss = (1/(2*N))*np.square(y-np.dot(tx,w))
    return (w, loss)

def least_squares_SGD(y, tx, w_initial, max_iters, gamma):
    """Compute Least Squares with Stochastic Gradient Descent"""
    """INPUTS : vector with data (tx, y), the initial w, the maximum number of iterations and the learning rate gamma"""
    """OUTPUTS : the weights w of the model """
    N = len(y)
    w = w_initial
    i=0
    while i<max_iters :
        for j in range(N) : 
            mse_grad_loss = np.dot(tx[j].T, (y[j]-np.dot(tx[j], w)))
            w = w - gamma*mse_grad_loss
        i+=1
    loss = (1/(2*N))*np.square(y-np.dot(tx,w))
    return (w, loss)

def least_squares(y, tx):
    """Compute Least Squares with normal equations"""
    """INPUTS : vector with data (tx, y)"""
    """OUTPUTS : the weights w of the model """
    w =  np.matmul(np.linalg.solve(np.matmul((tx.T), tx),(tx.T)), y)
    N = len(y)
    loss = (1/(2*N))*np.square(y-np.dot(tx,w))
    return (w, loss)

def ridge_regression(y, tx, lambda_): 
    """Compute Ridge Regresssion with normal equations"""
    """INPUTS : vector with data (tx, y) and penalty parameter lambda_"""
    """OUTPUTS : the weights w of the model """
    N = len(y)
    w =  np.matmul(np.linalg.solve((np.matmul((tx.T), tx)+lambda_*2*N*np.identity(tx.shape[1])),(tx.T)), y)
    loss = (1/(2*N))*np.square(y-np.dot(tx,w)) + lambda_*np.sum(np.square(w))
    return (w, loss)