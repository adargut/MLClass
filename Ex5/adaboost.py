#################################
# Your name: Adar Gutman 316265065
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data
from tqdm import tqdm

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    num_reviews = 5000
    theta_vals = get_theta_vals(X=X_train)
    weights = [1 / num_reviews] * num_reviews
    alpha_vals = []
    weak_learners = []

    iter = tqdm(range(1, T + 1), desc='adaboost')
    for t in iter:  # T+1): todo remove tqdm, for debug only
        optimal_wl = select_optimal_WL(weights=weights, X=X_train, Y=y_train, theta_vals=theta_vals)
        optimal_wl_error = compute_weighted_error(optimal_wl, weights, X_train, y_train)
        optimal_wl_weight = optimal_wl_alpha(err=optimal_wl_error)
        weights = update_weights(weights, X_train, y_train, optimal_wl, optimal_wl_weight)

        weak_learners.append(optimal_wl)
        alpha_vals.append(optimal_wl_weight)

    return weak_learners, alpha_vals


##############################################
# You can add more methods here, if needed.

def plot_test_train_error(test_data, test_labels, train_data, train_labels, alpha_vals, weak_learners):
    """
    Plot the test and the train error achieved by adaboost, as the algorithm progresses in iterations
    :param test_data: the test data
    :param test_labels: the labels for it
    :param train_data: the train data
    :param train_labels: the labels for it
    :param alpha_vals: the weights given to the weak learners
    :param weak_learners: the weak learners
    :return: none, draw & show a nice plot of the error as a function of t (index of adaboost iteration)
    """
    num_iters = len(alpha_vals)
    X = range(1, num_iters + 1)
    Y_test = []
    Y_train = []

    for i in range(1, num_iters + 1):
        partial_error_test = calc_error(data=test_data, labels=test_labels, learners=weak_learners[:i],
                                        alpha_vals=alpha_vals[:i])
        Y_test.append(partial_error_test)
        partial_error_train = calc_error(data=train_data, labels=train_labels, learners=weak_learners[:i],
                                         alpha_vals=alpha_vals[:i])
        Y_train.append(partial_error_train)

    plt.plot(X, Y_test, label='Test Error')
    plt.plot(X, Y_train, label='Train Error')
    plt.legend(loc='best')
    plt.title('Test and train error by ensemble of weak learners as adaboost progresses')
    plt.show()


def plot_test_train_average_exp_loss(test_data, test_labels, train_data, train_labels, alpha_vals, weak_learners):
    """
    Plot the average exponential loss achieved by adaboost as it progresses
    :param test_data: the test data
    :param test_labels: the test labels
    :param train_data: the train data
    :param train_labels: the train labels
    :param alpha_vals: the weights given to the weak learners
    :param weak_learners: the weak learners
    :return: nothing, show a plot of exp loss
    """
    num_iters = len(alpha_vals)
    X = range(1, num_iters + 1)
    Y_test = []
    Y_train = []

    for i in range(1, num_iters + 1):
        partial_error_test = calc_exp_loss(data=test_data, labels=test_labels, weak_learners=weak_learners,
                                           alpha_vals=alpha_vals)
        Y_test.append(partial_error_test)
        partial_error_train = calc_exp_loss(data=train_data, labels=train_labels, weak_learners=weak_learners,
                                            alpha_vals=alpha_vals)
        Y_train.append(partial_error_train)

    plt.plot(X, Y_test, label='Test Exponential Loss')
    plt.plot(X, Y_train, label='Train Exponential Loss')
    plt.legend(loc='best')
    plt.title('Test and train exponential loss by ensemble of weak learners as adaboost progresses')
    plt.show()


def calc_error(data, labels, learners, alpha_vals):
    """
    Calculate error probability on data by ensemble of weak learners
    :param data: the sample data
    :param labels: the sample labels
    :param learners: the weak learners
    :param alpha_vals: the weights of each weak learner
    :return: probability of error
    """
    data_len = len(data)
    errors_made = 0

    for i in range(data_len):
        ensemble_prediction = ensemble_learners_predict(data, learners, alpha_vals, i)
        if ensemble_prediction != labels[i]:
            errors_made += 1

    return errors_made / data_len


def calc_exp_loss(data, labels, weak_learners, alpha_vals):
    """
    Calculate the average exponential loss of the weak learners, which adaboost minimizes
    :param data: the data
    :param labels: the labels
    :param alpha_vals: the weights of weak learners
    :param weak_learners: the weak learners
    :return: average exponential loss
    """
    m = len(data)
    T = len(weak_learners)
    ans = 0

    for i in range(m):
        s = 0
        for j in range(T):
            s += alpha_vals[j] * weak_learner_predict(data, i, weak_learners[j])

        exponent = np.exp(-labels[i] * s)
        ans += exponent

    return (1 / m) * ans


def ensemble_learners_predict(data, learners, alpha_vals, i):
    """
    Make a prediction according to the linear combination of the weak learners ensemble
    :param data: the data to predict on
    :param learners: the weak learners
    :param alpha_vals: weight of each weak learner
    :param i: index of review we're attempting to predict: positive or negative
    :return: The ensemble prediction
    """
    s = np.sum(alpha * weak_learner_predict(data, i, learner) for alpha, learner in zip(alpha_vals, learners))
    return np.sign(s)


def get_theta_vals(X):
    """
    Generate the theta used for the hypotheses class
    :param X: sample data
    :return: the relevant thetas we inspect
    """
    min_theta = 2 ** 31
    max_theta = (-2) ** 31

    for review in X:
        for word_count in review:
            min_theta = min(min_theta, word_count)
            max_theta = max(max_theta, word_count)

    return range(int(min_theta), int(max_theta) + 1)


def weak_learner_predict(X, i, h):
    """
    Make some prediction with a weak learner
    :param h: The weak learner, given by i, j, flip and theta params
    :param X: train data
    :return: the prediction, either 1 for positive review or -1 for negative review
    """
    j, theta, flip = h
    if X[i][j] <= theta:
        ans = 1
    else:
        ans = -1

    return ans if not flip else -ans


def select_optimal_WL(weights, X, Y, theta_vals):
    """
    Select the optimal weak learner over the class of hypotheses
    :param weights: error distribution
    :param X: train data
    :param Y: labels
    :param theta_vals: theta parameters attempted for weak learners
    :return: the best WL in the class, given the weights
    """
    optimal_wl = None
    min_error = 2 ** 31
    review_len = len(X[0])

    iter = tqdm(range(review_len), desc='optimal wl selection')
    for word_idx in iter:
        for theta in theta_vals:
            for flip in [True, False]:
                h = (word_idx, theta, flip,)
                WL_error = compute_weighted_error(h=h, weights=weights, X=X, Y=Y)
                if WL_error < min_error:
                    min_error = WL_error
                    optimal_wl = word_idx, theta, flip

    return optimal_wl


def compute_weighted_error(h, weights, X, Y):
    """
    Compute the weighted error of some weak learner to select the optimal weak learner
    :param h: the weak learner
    :param weights: the distribution to compute error prob
    :param X: train data (IMDB reviews)
    :param Y: labels (negative or positive review)
    :return: error of weak learner h on dataset according to distribution weights
    """
    error_prob = 0
    num_reviews = len(X)

    for review_idx in range(num_reviews):
        prediction = weak_learner_predict(X=X, i=review_idx, h=h)
        if prediction != Y[review_idx]:
            error_prob += weights[review_idx]

    return error_prob


def compute_normalizing_factor(X, Y, h, alpha, weights):
    """
    Ensure the weights sum up to 1
    :param weights: weights for distribution
    :param X: train data
    :param Y: train labels
    :param h: The optimal weak learner picked by adaboost
    :param alpha: Optimal weak learner's weight in the ensemble
    :return: Normalizing factor, allows the weights to sum up to 1 after division by it
    """
    factor = 0

    for i in range(len(X)):
        prediction = weak_learner_predict(X=X, i=i, h=h)
        factor += (weights[i] * np.exp(-Y[i] * prediction * alpha))

    return factor


def optimal_wl_alpha(err):
    """
    The weight given to some weak learner in the ensemble
    :param err: the weak learner error probability
    :return: weight of weak learner on ensemble (alpha val)
    """
    return 0.5 * np.log((1 - err) / err)


def update_weights(weights, X, Y, h, alpha):
    """
    Update the old weights distribution according to new optimal weak learner
    :param weights: old weights
    :param X: train data
    :param Y: train labels
    :param h: optimal weak learner
    :param alpha: optimal weak learner's weight
    :return: updated weights
    """
    # (X, Y, h, alpha):
    normalizing_factor = compute_normalizing_factor(X, Y, h, alpha, weights)
    updated_weights = []
    num_reviews = len(X)

    for i in range(num_reviews):
        prediction = weak_learner_predict(X=X, i=i, h=h)
        updated_weights_at_i = (weights[i] * np.exp(-Y[i] * alpha * prediction)) / normalizing_factor
        updated_weights.append(updated_weights_at_i)

    return updated_weights


##############################################


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    # section a
    T = 80
    weak_learners, alpha_vals = run_adaboost(X_train, y_train, T)
    plot_test_train_error(test_data=X_test, test_labels=y_test, train_data=X_train, train_labels=y_train,
                          alpha_vals=alpha_vals, weak_learners=weak_learners)

    # section b
    T = 10
    first_10_weak_learners, first_10_alpha_vals = weak_learners[:10], alpha_vals[:10]
    print('Learners picked by adaboost in section b', first_10_weak_learners)
    words_predicted_by = []
    for _, idx, _ in weak_learners:
        word = vocab[idx]
        words_predicted_by.append(word)

    print('The first 10 weak learners predicted by the words', words_predicted_by)

    # section c
    plot_test_train_average_exp_loss(test_data=X_test, test_labels=y_test, train_data=X_train, train_labels=y_train,
                                     alpha_vals=alpha_vals, weak_learners=weak_learners)


if __name__ == '__main__':
    main()
