import numpy as np
import matplotlib.pyplot as plt
from scripts.proj1_helpers import sigmoid

def roc(x_test, y_test, weights, thresholds) :
    # threshold as parameter or do we set it inside the definition of the function ?
    # threshold must be between 0 and 1
    tprs = []
    fprs = []
    for thresh in thresholds :
        print(f'Threshold : {thresh} \n')
        y_pred = predict_labels_thresh(weights, x_test, thresh)
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fn = np.sum((y_pred == -1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == -1))
        tn = np.sum((y_pred == -1) & (y_test == -1))
        print(f'True positive : {tp}, True negative : {tn}, False negative : {fn}, False positive : {fp} \n')
        tpr = tp/(tp+fn) # = sensitivity
        fpr = fp/(fp+tn) # = 1 - specificity
        print(f'Sensitivity : {tpr}, Specificity : {1-fpr} \n')
        tprs.append(tpr)
        fprs.append(fpr)
    auc = 0
    for i in range(len(tprs)-1) :
        # AUC computation using linear trapezoidal method
        auc = auc+ 0.5*(tprs[i+1]-fprs[i])*(fprs[i+1]+fprs[i])
    return auc, fprs, tprs
   
def predict_labels_log_thresh(weights, data, thresh):
    """Generates class predictions given weights, and a test data matrix"""
    w0 = np.ones((data.shape[0], 1))
    data = np.hstack((data, w0))
    y_pred = sigmoid(np.dot(data, weights))
    y_pred[np.where(y_pred <= thresh)] = -1
    y_pred[np.where(y_pred > thresh)] = 1

    return y_pred

def predict_labels_thresh(weights, data, thresh):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    #converting threshold from 0 to 1 into -1 to 1
    thresh = -1 + ((1-(-1)) / (1-0)) * (thresh - 0)
    print(f'Threshold adjusted: {thresh}')
    y_pred[np.where(y_pred <= thresh)] = -1
    y_pred[np.where(y_pred > thresh)] = 1

    return y_pred
