#################################
# Your name: Adar Gutman 316265065
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
from matplotlib import pyplot as plt

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

    N = len(data[0])
    w = _init_classifier(N)

    for t in range(1, T + 1):
        w = _SGD_hinge_step(eta_0=eta_0,
                            prev_w=w, C=C, t=t,
                            train_images=data,
                            train_labels=labels)

    return w


def SGD_ce(data, labels, eta_0, T):
    """
    Implements multi-class cross entropy loss using SGD.
    """
    N = len(data[0])
    classifiers = _init_classifiers(N)

    for i in range(1, T + 1):
        classifiers = _SGD_ce_step(eta_0, classifiers, train_images=data, train_labels=labels)

    return classifiers


#################################

# Place for additional code
CLASSIFICATIONS = \
    {
        "DIGIT_ZERO": -1,
        "DIGIT_EIGHT": 1
    }


def _SGD_hinge_step(eta_0, prev_w, C, t, train_images, train_labels):
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


def _SGD_ce_step(eta_0, classifiers, train_images, train_labels):
    """
    One step of the SGD algorithm using ce loss
    """
    i = int(np.random.uniform(low=0, high=len(train_images)))
    gradients = ce_gradient(classifiers=classifiers, sample=train_images[i], label=int(train_labels[i]))

    for i in range(len(classifiers)):
        classifiers[i] -= (gradients[i] * eta_0)

    return classifiers


def _init_classifier(N):
    """
    Initializes the classifier
    :param N: length of classifier
    :return: np.zeros array of size N
    """
    return np.zeros(shape=(N,), dtype=np.float64)


def _classify(image, classifier):
    """
    Classify an image
    """
    if np.dot(image, classifier) < 0:
        return CLASSIFICATIONS["DIGIT_ZERO"]

    return CLASSIFICATIONS["DIGIT_EIGHT"]


def _measure_classifier(validation_images, validation_labels, classifier):
    """
    Calculate % of correct predictions for classifier
    """
    correct_predictions = 0

    for image, label in zip(validation_images, validation_labels):
        classification = _classify(image, classifier)
        if classification == label:
            correct_predictions += 1

    return correct_predictions / len(validation_labels)


def SGD_hinge_and_validate(C, T, train_images, train_labels, validation_images, validation_labels, eta_0):
    """
    Perform SGD algorithm and measure accuracy together
    """
    w = SGD_hinge(data=train_images, labels=train_labels, C=C, eta_0=eta_0, T=T)

    # Measure classifier on validation data
    accuracy = _measure_classifier(validation_images=validation_images,
                                   validation_labels=validation_labels,
                                   classifier=w)

    return accuracy, w


def _init_classifiers(N):
    """
    :param N: length of weights vector
    :return: weights initialized to 1
    """
    return np.ones(shape=(10, N))


def softmax(classifiers, sample):
    """
    Implement softmax over classifiers
    """
    nominators = [np.dot(classifier, sample) for classifier in classifiers]
    nominators = nominators - np.max(nominators)  # prevent overflow
    nominators = np.exp(nominators)

    return nominators / np.sum(nominators)


def ce_gradient(classifiers, sample, label):
    """
    Update gradients
    """
    _softmax = softmax(classifiers=classifiers, sample=sample)
    _softmax[label] -= 1

    return [p * sample for p in _softmax]


def measure_ce_accuracy(classifiers, validation_data, validation_labels):
    """
    Check accuracy of cross entropy SGD classifier
    """
    correct_predictions = 0

    for image, label in zip(validation_data, validation_labels):
        nominators = [np.dot(classifier, image) for classifier in classifiers]
        prediction = np.argmax(nominators)

        if prediction == int(label):
            correct_predictions += 1

    return correct_predictions / len(validation_data)


def main():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()

    print('## Part 1 ##')
    # Find optimal eta
    eta_results = {}

    for _ in range(10):
        for i in range(-5, 6):
            eta_0 = 10 ** i
            if eta_0 not in eta_results:
                eta_results[eta_0] = []
            accuracy, _ = SGD_hinge_and_validate(C=1,
                                                 T=1000,
                                                 train_images=train_data,
                                                 train_labels=train_labels,
                                                 validation_images=validation_data,
                                                 validation_labels=validation_labels,
                                                 eta_0=eta_0)
            eta_results[eta_0].append(accuracy)

    optimal_eta = None
    optimal_accuracy = 0
    averages = []
    for eta_0 in eta_results.keys():
        average_accuracy = np.average(eta_results[eta_0])
        averages.append((np.log10(eta_0), average_accuracy,))
        if average_accuracy > optimal_accuracy:
            optimal_eta = eta_0
            optimal_accuracy = average_accuracy

    print('The optimal eta_0 we got is', optimal_eta)
    plt.plot(*zip(*averages), '.-')
    plt.title('Average accuracy as a function of eta_0')
    plt.xlim([-5, 5])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('Log 10 scale of eta')
    plt.savefig('part-1-accuracy-wrt-eta')
    plt.clf()

    # Find optimal C value given optimal eta value
    C_results = {}

    for _ in range(10):
        for i in range(-5, 6):
            C = 10 ** i
            if C not in C_results:
                C_results[C] = []
            accuracy, _ = SGD_hinge_and_validate(C=C,
                                                 T=1000,
                                                 train_images=train_data,
                                                 train_labels=train_labels,
                                                 validation_images=validation_data,
                                                 validation_labels=validation_labels,
                                                 eta_0=optimal_eta)
            C_results[C].append(accuracy)

    averages = []
    optimal_C = None
    optimal_accuracy = 0
    for C in C_results.keys():
        average_accuracy = np.average(C_results[C])
        averages.append((np.log10(C), average_accuracy,))
        if average_accuracy > optimal_accuracy:
            optimal_C = C
            optimal_accuracy = average_accuracy

    print('The optimal C we got is', optimal_C, 'with accuracy', optimal_accuracy)
    plt.plot(*zip(*averages), '.-')
    plt.xlim([-5, 5])
    plt.ylim([-0.1, 1.1])
    plt.title('Average accuracy as a function of C')
    plt.savefig('part-1-accuracy-wrt-C')
    plt.clf()

    # Train classifier on optimal C and eta_0
    T = 20000
    accuracy, optimal_classifier = SGD_hinge_and_validate(C=optimal_C,
                                                          T=T,
                                                          train_images=train_data,
                                                          train_labels=train_labels,
                                                          validation_images=test_data,
                                                          validation_labels=test_labels,
                                                          eta_0=optimal_eta)

    print('The best accuracy achieved over test set given optimal eta_0 and C in part 1 is', accuracy)
    # Draw classifier as image
    reshaped_img = np.reshape(optimal_classifier, newshape=(28, 28,))
    plt.imsave('part-1-model.jpg', reshaped_img)

    print('## Part 2 ##')
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
    eta_results = {}
    for _ in range(10):
        for i in range(-15, 6):
            eta_0 = 10 ** i
            if eta_0 not in eta_results:
                eta_results[eta_0] = []
            T = 1000
            classifiers = SGD_ce(data=train_data, labels=train_labels, eta_0=eta_0, T=T)
            accuracy = measure_ce_accuracy(classifiers=classifiers, validation_data=validation_data,
                                           validation_labels=validation_labels)
            eta_results[eta_0].append(accuracy)

    averages = []
    optimal_eta = None
    optimal_accuracy = 0
    for eta_0 in eta_results.keys():
        average_accuracy = np.average(eta_results[eta_0])
        averages.append((np.log10(eta_0), average_accuracy,))
        if average_accuracy > optimal_accuracy:
            optimal_eta = eta_0
            optimal_accuracy = average_accuracy

    print('The optimal eta_0 we got is', optimal_eta, 'with accuracy', optimal_accuracy)
    plt.plot(*zip(*averages), '.-')
    plt.xlim([-15, 5])
    plt.ylim([-0.1, 1.1])
    plt.title('Average accuracy as a function of eta_0')
    plt.xlabel('Log 10 scale of eta')
    plt.savefig('part-2-accuracy-wrt-eta_0')
    plt.clf()

    # Train the classifier using optimal eta_0
    T = 20000
    classifiers = SGD_ce(data=train_data, labels=train_labels, eta_0=optimal_eta, T=T)

    for i, classifier in enumerate(classifiers):
        reshaped_classifier_img = np.reshape(classifier, newshape=(28, 28))
        filename = "part-2-model-digit-" + str(i) + ".jpg"
        plt.imsave(filename, reshaped_classifier_img)

    accuracy = measure_ce_accuracy(classifiers=classifiers, validation_data=test_data, validation_labels=test_labels)
    print('The best accuracy achieved given optimal eta_0 on test set is in part 2', accuracy)


if __name__ == '__main__':
    main()

#################################
