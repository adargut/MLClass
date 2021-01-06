#################################
# Your name: Adar Gutman 316265065
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

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
    theta_vals = 5000
    weights = [1 / num_reviews] * num_reviews
    alpha_vals = []
    weak_learners = []

    for t in range(1, 11):  # T+1):
        optimal_wl = select_optimal_WL(weights=weights, X=X_train, Y=y_train, theta_vals=theta_vals)
        optimal_wl_error = compute_weighted_error(optimal_wl, weights, X_train, y_train)
        optimal_wl_weight = optimal_wl_alpha(err=optimal_wl_error)
        weights = update_weights(weights, X_train, y_train, optimal_wl, optimal_wl_weight)

        weak_learners.append(optimal_wl)
        alpha_vals.append(optimal_wl_weight)

    return weak_learners, alpha_vals


##############################################
# You can add more methods here, if needed.

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
    min_error = sys.maxsize
    review_len = len(X[0])

    for word_idx in range(review_len):
        for theta in range(theta_vals):
            for flip in [True, False]:
                h = (word_idx, theta, flip,)
                WL_error = compute_weighted_error(h=h, weights=weights, X=X, Y=Y)
                if WL_error < min_error:
                    min_error = WL_error
                    optimal_wl = (word_idx, theta, flag,)

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


def compute_normalizing_factor(X, Y, h, alpha):
    """
    Ensure the weights sum up to 1
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
    normalizing_factor = compute_normalizing_factor(X, Y, weights, h)
    updated_weights = []
    num_reviews = len(X)

    for i in range(num_reviews):
        i, theta, flip = h
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

    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    print(hypotheses, alpha_vals)  # TODO: remove me, for debug

    ##############################################
    # You can add more methods here, if needed.

    ##############################################


if __name__ == '__main__':
    main()
