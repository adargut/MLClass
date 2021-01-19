#################################
# Your name: Adar Gutman 316265065
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""


def helper():
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


def perceptron(data, labels):
    """
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """
    assert len(data) > 0, "Perceptron needs data to learn!"
    N = len(data[0])
    weights = initialize_weights(N=N)

    for sample, label in zip(data, labels):
        prediction = predict(weights=weights, sample=sample)

        # Perceptron made a mistake, therefore needs to update weights
        if prediction != label:
            weights = update_weights(weights=weights, label=label, sample=sample)

    # The final weights represent the perceptron classifier
    return weights


#################################

# Place for additional code

def initialize_weights(N):
    return np.zeros(shape=(N,))


def update_weights(weights, label, sample):
    return weights + label * sample


def predict(weights, sample):
    return np.sign(np.dot(weights, sample))


def calculate_accuracy(classifier, data, labels):
    errors = 0
    misclassified_samples = []

    for sample, label in zip(data, labels):
        prediction = predict(classifier, sample)
        if prediction != label:
            errors += 1
            misclassified_samples.append(sample)

    error_probability = errors / len(data)
    return 1 - error_probability, misclassified_samples


def update_accuracies(accuracy_dict, accuracy, n):
    if n not in accuracy_dict:
        accuracy_dict[n] = []
    accuracy_dict[n].append(accuracy)


def plot_perceptron_accuracy(X, Y_mean, Y_5_perc, Y_95_perc):
    plt.plot(X, Y_mean, label='Mean Accuracy')
    plt.plot(X, Y_5_perc, label='5 Percentile Accuracy')
    plt.plot(X, Y_95_perc, label='95 Percentile Accuracy')
    plt.legend(loc='best')
    plt.title('Perceptron accuracy on test data as a function of n')
    plt.xlabel('Value of n (number of sample)')
    plt.ylabel('Accuracy %')
    plt.show()


def permute_data_and_labels(data, labels):
    assert len(data) == len(labels), 'Length of data must correspond to length of labels'
    perm = np.random.permutation(len(data))
    data = data[perm]
    labels = labels[perm]

    return data, labels


def show_image(image):
    plt.imshow(image, interpolation='nearest')
    plt.show()


def main():
    train_data, train_labels, _, _, test_data, test_labels = helper()

    # part a
    accuracy_dict = dict()
    X = [5, 10, 50, 100, 500, 1000, 5000]
    Y_mean, Y_5_perc, Y_95_perc = list(), list(), list()
    for n in X:
        partial_train_data, partial_train_labels = train_data[:n], train_labels[:n]
        for _ in range(100):
            partial_train_data, partial_train_labels = permute_data_and_labels(data=partial_train_data,
                                                                               labels=partial_train_labels)

            classifier = perceptron(data=partial_train_data, labels=partial_train_labels)
            accuracy, _ = calculate_accuracy(classifier, data=test_data, labels=test_labels)
            update_accuracies(accuracy_dict, accuracy, n)

        mean_accuracy_for_n = np.average(accuracy_dict[n])
        percentile_5_for_n = np.percentile(accuracy_dict[n], 5)
        percentile_95_for_n = np.percentile(accuracy_dict[n], 95)
        Y_mean.append(mean_accuracy_for_n)
        Y_5_perc.append(percentile_5_for_n)
        Y_95_perc.append(percentile_95_for_n)

    plot_perceptron_accuracy(X=X, Y_mean=Y_mean, Y_5_perc=Y_5_perc, Y_95_perc=Y_95_perc)

    # part b
    classifier = perceptron(data=train_data, labels=train_labels)
    show_image(np.reshape(classifier, newshape=(28, 28,)))

    # part c
    accuracy, misclassified_samples = calculate_accuracy(classifier, data=test_data, labels=test_labels)
    print('The classifier trained on the full training set achieved accuracy of', accuracy)

    # part d
    if len(misclassified_samples) == 0:
        print('Perceptron was 100% correct!')
    else:
        show_image(np.reshape(misclassified_samples[0], newshape=(28, 28,)))


if __name__ == "__main__":
    main()

#################################
