import numpy.random
from collections import Counter
from sklearn.datasets import fetch_openml
from scipy.spatial import distance
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import dataset
mnist = fetch_openml('mnist_784', data_home="/data")
data = mnist['data']
labels = mnist['target']

# Split into test/train data
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]


def preprocess_data(train_data, train_labels):
    """Preprocess neighbors for every image in train data"""
    processed_data = dict()
    for image in train_data:
        neighbors = sorted(list(zip(train_labels, train_data)), key=lambda x: distance.euclidean(x[1], image))
        processed_data[image] = neighbors
    return processed_data


def predict(train_data, train_labels, query, k):
    """Classify query image using K nearest neighbors from train data"""
    k_neighbors = sorted(list(zip(train_labels, train_data)), key=lambda x: distance.euclidean(x[1], query))[:k]
    labels_counter = Counter([neighbor[0] for neighbor in k_neighbors])
    return labels_counter.most_common(1)[0][0]


def make_predictions(test_data, train_sample_labels, train_sample_data, K):
    """Make predictions for given test data"""""
    predictions = []
    for test_image in test_data:
        prediction = predict(train_sample_data, train_sample_labels, test_image, K)
        predictions.append(prediction[0])
    return predictions


def measure_loss(predictions, true_labels):
    """Measure KNN algorithm loss using 0-1 loss"""
    loss = 0
    for prediction, true_label in zip(predictions, true_labels):
        if prediction != true_label:
            loss += 1
    return loss


def measure_accuracy(loss, N):
    """Measure KNN accuracy"""
    return (N - loss) / N


def evaluate_classifier(predictions, true_labels):
    """Evaluate how good our classifier really is"""
    loss = measure_loss(predictions=predictions, true_labels=true_labels)
    accuracy = measure_accuracy(loss=loss, N=len(true_labels))
    return accuracy


def main():
    # KNN algorithm parameters
    N = 1000
    K = 10

    # Sample data for KNN
    train_sample_data = train[:N]
    train_sample_labels = train_labels[:N]

    # Make predictions on test data
    predictions = make_predictions(test_data=test[:N],
                                   train_sample_labels=train_sample_labels,
                                   train_sample_data=train_sample_data,
                                   K=K)

    # Evaluate KNN on test data
    accuracy = evaluate_classifier(predictions=predictions, true_labels=test_labels[:N])
    print("Achieved accuracy: ", accuracy)

    # Accuracy as a function of k
    left, right = 1, 101
    results = []
    for i in tqdm(range(left, right)):
        predictions = make_predictions(test_data=test[:N],
                                       train_sample_labels=train_sample_labels,
                                       train_sample_data=train_sample_data,
                                       K=i)
        accuracy = evaluate_classifier(predictions=predictions, true_labels=test_labels[:N])
        results.append((i, accuracy,))

    # Plot the results
    plt.title('Accuracy as a function of k')
    plt.plot(*zip(*results))
    plt.show()

    # Accuracy as a function of n
    left, right = 100, 5000
    results = []
    for i in tqdm(range(left, right+1, 500)):
        predictions = make_predictions(test_data=test[:N],
                                       train_sample_labels=train_labels[:i+1],
                                       train_sample_data=train[:i+1],
                                       K=1)
        accuracy = evaluate_classifier(predictions=predictions, true_labels=test_labels[:N])
        results.append((left, accuracy,))
        left += 100

    # Plot the results
    plt.title('Accuracy as a function of n')
    plt.plot(*zip(*results))
    plt.show()


if __name__ == '__main__':
    main()
