import matplotlib.pyplot as plt


def cross_validation_visualization(lambds, loss, method, folder_path, xlabel='gamma'):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, loss, marker=".", color='r', label='test error')
    plt.xlabel(xlabel)
    plt.ylabel("loss")
    plt.xlim(1e-4, 1)
    plt.title(method)
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig((folder_path+method))