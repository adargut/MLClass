#################################
# Your name: Adar Gutman 316265065
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper_hinge():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def helper_ce():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = labels[train_idx[:6000]]

    validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
    validation_labels = labels[train_idx[6000:8000]]

    test_data_unscaled = data[8000 + test_idx, :].astype(float)
    test_labels = labels[8000 + test_idx]

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    """
    # TODO: Implement me
    pass


def SGD_ce(data, labels, eta_0, T):
    """
    Implements multi-class cross entropy loss using SGD.
    """
    # TODO: Implement me
    pass


#################################

# Place for additional code
CLASSIFICATIONS = \
    {
        "DIGIT_ZERO": -1,
        "DIGIT_EIGHT": 1
    }


def _SGD_step(eta_0, prev_w, C, t, train_images, train_labels):
    """
    One step of the SGD algorithm: w_{t+1} = (1-eta_t) * w_t + C * eta_t * y_i * x_i
    In other words, the new classifier is a linear combination of the previous classifier and the gradient
    """
    eta = eta_0 / t
    i = int(np.random.uniform(low=0, high=len(train_images)))
    y_i, x_i = train_labels[i], train_images[i]
    if np.dot(y_i * prev_w, x_i) < 1:
        return (1 - eta) * prev_w + eta * C * y_i * x_i

    return (1 - eta) * prev_w


def _init_classifier(N):
    """
    Initializes the classifier
    :param N: length of classifier
    :return: np.zeros array of size N
    """
    return np.zeros(shape=(N,), dtype=np.float64)


def _SGD(C, T, train_images, train_labels, validation_images, validation_labels, eta_0):
    N = len(train_images[0])
    w = _init_classifier(N)

    for t in range(1, T + 1):
        w = _SGD_step(eta_0=eta_0,
                      prev_w=w, C=C, t=t,
                      train_images=train_images,
                      train_labels=train_labels)

    # Measure classifier on validation data
    accuracy = _measure_classifier(validation_images=validation_images,
                                   validation_labels=validation_labels,
                                   classifier=w)
    print('Accuracy achieved by classifier', accuracy)
    return accuracy


def _classify(image, classifier):
    if np.dot(image, classifier) < 0:
        return CLASSIFICATIONS["DIGIT_ZERO"]

    return CLASSIFICATIONS["DIGIT_EIGHT"]


def _measure_classifier(validation_images, validation_labels, classifier):
    correct_predictions = 0

    for image, label in zip(validation_images, validation_labels):
        classification = _classify(image, classifier)
        if classification == label:
            correct_predictions += 1

    return correct_predictions / len(validation_labels)


def main():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    results = {}

    for _ in range(10):
        for i in range(-5, 6):
            eta_0 = 10 ** i
            if eta_0 not in results:
                results[eta_0] = []
            print('Trying out eta ', eta_0)
            accuracy = _SGD(C=1,
                            T=1000,
                            train_images=train_data,
                            train_labels=train_labels,
                            validation_images=validation_data,
                            validation_labels=validation_labels,
                            eta_0=eta_0)
            results[eta_0].append(accuracy)

    optimal_eta = None
    optimal_accuracy = 0
    for eta_0 in results.keys():
        average_accuracy = np.average(results[eta_0])
        if average_accuracy > optimal_accuracy:
            optimal_eta = eta_0

    print('The optimal eta_0 we got is', optimal_eta)


if __name__ == '__main__':
    main()

#################################
