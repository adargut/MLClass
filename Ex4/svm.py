#################################
# Your name: Adar Gutman
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""


# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    linearSVM = svm.SVC(kernel='linear', C=1000)
    linearSVM.fit(X=X_train, y=y_train)
    num_support_vectors_lin = len(linearSVM.support_vectors_)
    create_plot(X_train, y_train, linearSVM)
    plt.show()

    quadraticSVM = svm.SVC(kernel='poly', C=1000, degree=2)
    quadraticSVM.fit(X=X_train, y=y_train)
    num_support_vectors_quad = len(quadraticSVM.support_vectors_)
    create_plot(X_train, y_train, quadraticSVM)
    # plt.show()

    rbfSVM = svm.SVC(kernel='rbf', C=1000)
    rbfSVM.fit(X=X_train, y=y_train)
    num_support_vectors_rbf = len(rbfSVM.support_vectors_)
    create_plot(X_train, y_train, rbfSVM)
    # plt.show()

    return np.asarray([num_support_vectors_lin, num_support_vectors_quad, num_support_vectors_rbf])


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    lowest_C_exp = -5
    highest_C_exp = 5
    best_C = None
    best_score = 0
    scores = []

    for i in range(lowest_C_exp, highest_C_exp + 1):
        C = 10 ** i
        linearSVM = svm.SVC(C=C, kernel='linear')
        linearSVM.fit(X_train, y_train)
        accuracy = linearSVM.score(X_val, y_val)

        if accuracy > best_score:
            best_score = accuracy
            best_C = C

        scores.append((i, accuracy,))
        create_plot(X_train, y_train, linearSVM)
        plt.show()

    plt.plot(*zip(*scores))
    plt.title('Accuray of linear SVM as a function of C')
    plt.xlabel('C value (log10 scale)')
    plt.ylabel('Accuracy')
    # plt.show()

    return best_C


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    lowest_gamma_exp = -5
    highest_gamma_exp = 5
    C = 10
    best_gamma = None
    best_score = 0
    scores = []

    for i in range(lowest_gamma_exp, highest_gamma_exp + 1):
        gamma = 10 ** i
        rbfSVM = svm.SVC(gamma=gamma, C=C)
        rbfSVM.fit(X=X_train, y=y_train)
        accuracy = rbfSVM.score(X_val, y_val)

        if accuracy > best_score:
            best_gamma = gamma
            best_score = accuracy

        scores.append((i, accuracy,))
        create_plot(X_train, y_train, rbfSVM)
        plt.show()

    plt.title('Accuray of rbf SVM as a function of gamma')
    plt.xlabel('gamma value (log10 scale)')
    plt.ylabel('Accuracy')
    plt.plot(*zip(*scores))
    # plt.show()

    return best_gamma


def main():
    X, y, x_val, y_val = get_points()
    n_support_array = train_three_kernels(X, y, None, None)
    print('the number of support vectors:', n_support_array)

    best_C = linear_accuracy_per_C(X, y, x_val, y_val)
    print('the best C found is:', best_C)

    best_gamma = rbf_accuracy_per_gamma(X, y, x_val, y_val)
    print('the best gamma found is:', best_gamma)


if __name__ == '__main__':
    main()
