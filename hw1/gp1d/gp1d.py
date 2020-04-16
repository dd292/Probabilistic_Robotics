import numpy as np
from scipy.special import kv, gamma
import math
import matplotlib.pyplot as plt


def predict_squared_exponential_kernel(train_x, train_y, test_x, l, sigma_f, noise_sigma):
    """
    :param train_x: a numpy array of size [N]
    :param train_y: a numpy array of size [N]
    :param test_x:  a numpy array of size [M]
    :param l: length parameter of kernel. float
    :param sigma_f: scale parameter of kernel. float
    :param noise_sigma: noise standard deviation. float

    :return: mean: a numpy array of size [M]
             variance: a numpy array of size [M]

        Note: only return the variances, not the covariances
              i.e. the diagonal of the covariance matrix
    """


    K11 = get_pce_kernel(l, sigma_f, train_x, train_x)
    K12 = get_pce_kernel(l, sigma_f, train_x, test_x)
    K21 = get_pce_kernel(l, sigma_f, test_x, train_x)
    K22 = get_pce_kernel(l, sigma_f, test_x, test_x)

    inverted_val= np.linalg.inv(K11 + np.eye(K11.shape[0])*noise_sigma)
    mean= np.matmul(K21, np.matmul(inverted_val, train_y))
    variance = K22 - np.matmul(np.matmul(K21, inverted_val), K12)
    return mean, np.diagonal(variance)

def get_pce_kernel(l, sigma_f, X, Y):
    kernel = np.zeros((X.shape[0], Y.shape[0]))
    for iter1, i in enumerate(X):
        for iter2, j in enumerate(Y):
            kernel[iter1, iter2] = sigma_f * np.exp((np.linalg.norm(i - j) ** 2) / (-2 * l ** 2))
    return kernel

def get_matern_kernel(nu, l, X, Y):
    kernel = np.zeros((X.shape[0], Y.shape[0]))
    v = nu
    gam = gamma(v)


    for iter1, i in enumerate(X):
        for iter2, j in enumerate(Y):
            r = np.abs(i-j)
            interm = (math.sqrt(2*v)*r)/l
            bess = kv(v,interm)

            if interm != 0:
                kernel[iter1, iter2] = ((2**(1-v))*(interm**v) *bess )/gam
            else:
                kernel[iter1, iter2] = 1

    return kernel

def predict_matern_kernel(train_x, train_y, test_x, nu, l, noise_sigma):
    """
    :param train_x: a numpy array of size [N]
    :param train_y: a numpy array of size [N]
    :param test_x:  a numpy array of size [M]
    :param nu: parameter of kernel. float
    :param l:  parameter of kernel. float
    :param noise_sigma: noise standard deviation. float

    :return: mean: a numpy array of size [M]
             variance: a numpy array of size [M]

        Note: only return the variances, not the covariances
              i.e. the diagonal of the covariance matrix
    """
    K11 = get_matern_kernel(nu, l, train_x, train_x)
    K12 = get_matern_kernel(nu, l, train_x, test_x)
    K21 = get_matern_kernel(nu, l, test_x, train_x)
    K22 = get_matern_kernel(nu, l, test_x, test_x)

    inverted_val = np.linalg.inv(K11 + np.eye(K11.shape[0]) * noise_sigma)
    mean = np.matmul(K21, np.matmul(inverted_val, train_y))
    variance = K22 - np.matmul(np.matmul(K21, inverted_val), K12)
    return mean, np.diagonal(variance)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    def generate_training_data(n_points, x_min, x_max, func, noise_sigma, seed=1234):
        rng = np.random.RandomState(seed)
        xs = rng.uniform(x_min, x_max, n_points)
        ys = func(xs) + rng.randn(n_points) * noise_sigma
        return xs, ys

    f = np.sin
    noise_sigma = 0.35

    train_x, train_y = generate_training_data(n_points=20, x_min=0.0, x_max=10.0, func=f, noise_sigma=0.2)

    # Ground truth
    gt_x = np.linspace(0.0, 10.0, 100)
    gt_y = f(gt_x)

    test_x = np.linspace(0.0, 10.0, 100)

    def plot(predict_y, predict_y_variance):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), tight_layout=True)

        axs[0].scatter(train_x, train_y, facecolors='none', edgecolors='k', label='Noisy training data')
        axs[0].plot(gt_x, gt_y, color='k', label='True function')
        axs[0].set_title('Training data')
        axs[0].legend(bbox_to_anchor=(0.7, -0.05))

        axs[1].scatter(train_x, train_y, facecolors='none', edgecolors='k', label='Noisy training data')
        axs[1].plot(gt_x, gt_y, color='k', label='True function')
        axs[1].plot(test_x, predict_y, color='b', label='Test mean')
        axs[1].plot(test_x, predict_y + np.sqrt(predict_y_variance) * 2.0, color='r', label='Test variance')
        axs[1].plot(test_x, predict_y - np.sqrt(predict_y_variance) * 2.0, color='r')
        axs[1].plot(test_x, predict_y + np.sqrt(predict_y_variance) * 2.0 + noise_sigma, color='g', label='Test variance + noise')
        axs[1].plot(test_x, predict_y - np.sqrt(predict_y_variance) * 2.0 - noise_sigma, color='g')
        axs[1].set_title('Test predictions - means and variances')
        axs[1].legend(bbox_to_anchor=(0.75, -0.05))

        plt.show()

    predict_y, predict_y_variance = predict_squared_exponential_kernel(
        train_x, train_y, test_x, l=1.0, sigma_f=1.0, noise_sigma=noise_sigma)

    plot(predict_y, predict_y_variance)

    predict_y, predict_y_variance = predict_matern_kernel(
        train_x, train_y, test_x, nu=1.5, l=1.0, noise_sigma=noise_sigma)

    plot(predict_y, predict_y_variance)