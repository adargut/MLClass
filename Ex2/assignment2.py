
#################################
# Your name: Adar Gutman 316265065
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        samples = np.ndarray(shape=(m, 2,))

        for i in range(m):
            # Sample x uniformly from [0, 1]
            x = np.random.uniform(0, 1)

            # Probability of Y=1 given X=x for the x we just sampled
            if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
                p = 0.8
            else:
                p = 0.1

            # Generate a value for Y following Bernouli distribution with p from earlier
            y = np.random.binomial(1, p)

            # Add the new (x,y) sample
            samples[i] = (x, y,)

        samples = sorted(samples, key=lambda k: k[0])
        return samples

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        # Generate sample
        samples = self.sample_from_D(m)

        # Scatter plot
        plt.scatter(*zip(*samples))

        # Adjust y-axis for plot
        axes = plt.gca()
        axes.set_ylim([-0.1, 1.1])

        # Draw vertical lines on plot
        plt.axvline(0.2, color='r', linestyle='-')
        plt.axvline(0.4, color='r', linestyle='-')
        plt.axvline(0.6, color='r', linestyle='-')
        plt.axvline(0.8, color='r', linestyle='-')

        # Get x, y values of samples
        xs = [x[0] for x in samples]
        ys = [x[1] for x in samples]

        # Find best intervals on the above x, y values and plot them
        best_intervals, _ = intervals.find_best_interval(xs=xs, ys=ys, k=k)
        for a, b in best_intervals:
            plt.axhline(xmin=a, xmax=b, color='orange', linestyle='-')

        # Show the plot
        # plt.show() # todo remove this if you want to see the plot

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        avg_true_err = dict()
        avg_emp_err = dict()

        for i in range(T):
            X = []
            for m in range(m_first, m_last + 1, step):
                X.append(m)
                # Generate samples
                samples = self.sample_from_D(m)
                xs = [x[0] for x in samples]
                ys = [x[1] for x in samples]

                # Use ERM algorithm to determine hypothesis
                best_intervals, _ = intervals.find_best_interval(xs=xs, ys=ys, k=k)

                # Calculate empirical and true error on hypothesis
                emp_error = self._calculate_empirical_error(best_intervals, xs, ys)
                true_error = self._calculate_true_error(best_intervals)

                # Average out errors as a function of m
                if m not in avg_true_err:
                    avg_true_err[m] = []
                if m not in avg_emp_err:
                    avg_emp_err[m] = []
                avg_true_err[m].append(true_error)
                avg_emp_err[m].append(emp_error)

        # Prepare X,Y vectors for plots
        Y1 = []
        for m in range(m_first, m_last + 1, step):
            Y1.append(np.average(avg_emp_err[m]))
        Y2 = []
        for m in range(m_first, m_last + 1, step):
            Y2.append(np.average(avg_true_err[m]))

        # Plot average empirical and true errors as a function of m
        plt.plot(X, Y1, label="Average Empirical Error")
        plt.plot(X, Y2, label="Average True Error")
        plt.xlabel("Value of m")
        plt.legend()

        # Plot the graph
        # plt.show() # todo remove this if you want to see the plot

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        # Generate the samples required
        samples = self.sample_from_D(m=m)
        xs = [x[0] for x in samples]
        ys = [x[1] for x in samples]
        true_results = []
        emp_results = []
        best_k = -1
        best_error = 9999

        # Calculate hypothesis for every k in range, and calculate its error
        for k in range(k_first, k_last + 1, step):
            best_intervals, _ = intervals.find_best_interval(xs, ys, k)
            emp_error = self._calculate_empirical_error(best_intervals, xs, ys)
            true_error = self._calculate_true_error(best_intervals)

            # Check if we found a better k
            if emp_error < best_error:
                best_k = k

            true_results.append((k, true_error,))
            emp_results.append((k, emp_error,))

        # Plot the errors as a function of k
        plt.plot(*zip(*true_results), label="True Error")
        plt.plot(*zip(*emp_results), label="Empirical Error")
        plt.xlabel("Value of k")
        plt.legend()

        # Show the plot
        # plt.show() # todo remove this if you want to see the plot

        return best_k

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        # Generate the samples required
        samples = self.sample_from_D(m=m)
        xs = [x[0] for x in samples]
        ys = [x[1] for x in samples]
        true_results = []
        emp_results = []
        penalty_results = []
        emp_results_with_penalty = []
        best_k = -1
        best_error = 9999

        # Calculate hypothesis for every k in range, and calculate its error
        for k in range(k_first, k_last + 1, step):
            best_intervals, _ = intervals.find_best_interval(xs, ys, k)
            emp_error = self._calculate_empirical_error(best_intervals, xs, ys)
            true_error = self._calculate_true_error(best_intervals)

            # Use srm_penalty this time
            srm_penalty = self._calculate_penalty(delta=0.1, k=k, m=m)
            emp_error_with_penalty = emp_error + srm_penalty

            # Check if we found a better k
            if emp_error_with_penalty < best_error:
                best_k = k

            true_results.append((k, true_error,))
            emp_results.append((k, emp_error,))
            penalty_results.append((k, srm_penalty,))
            emp_results_with_penalty.append((k, emp_error_with_penalty,))

        # Plot the errors as a function of k
        plt.plot(*zip(*true_results), label="True Error")
        plt.plot(*zip(*emp_results), label="Empirical Error")
        plt.plot(*zip(*emp_results_with_penalty), label="Empirical Error With SRM Penalty")
        plt.plot(*zip(*penalty_results), label="SRM Penalty")
        plt.xlabel("Value of k")
        plt.legend()

        # Show the plot
        # plt.show() todo remove this is you want to see the plot

        return best_k

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # Generate samples from given distribution
        samples = self.sample_from_D(m)
        min_error = 9999
        best_k = None

        # Repeat the cross validation T times
        for i in range(T):
            # Randomly split samples into holdout and train data
            # Holdout: 20% of the data
            # Train: 80% of the data
            indices = np.random.permutation(m)
            train_idx = indices[:int(m * 0.8)]
            holdout_idx = indices[int(m * 0.8):]
            holdout, train = [], []
            for idx in train_idx:
                train.append(samples[idx])
            for idx in holdout_idx:
                holdout.append(samples[idx])
            train = sorted(train, key=lambda x: x[0])  # Sort train data by x labels
            train_x, train_y = [x[0] for x in train], [x[1] for x in train]
            holdout_x, holdout_y = [x[0] for x in holdout], [x[1] for x in holdout]

            for k in range(1, 11):
                # Perform ERM on train data
                best_intervals, _ = intervals.find_best_interval(xs=train_x, ys=train_y, k=k)

                # Calculate error on holdout data
                emp_error = self._calculate_empirical_error(best_intervals, x_labels=holdout_x, y_labels=holdout_y)

                # Check if we have found a smaller empirical error on holdout data
                if emp_error < min_error:
                    min_error = emp_error
                    best_k = k

        print('the best k found: ', best_k)
        return best_k

    #################################
    # Place for additional methods

    def _calculate_empirical_error(self, intervals, x_labels, y_labels):
        errors = 0
        num_labels = len(x_labels)

        # Check if label of x agrees with label classifier gave x
        for i, x in enumerate(x_labels):
            x_prediction = 0
            true_prediction = y_labels[i]

            # Check if x is contained in some interval => h(x)=1
            for a, b in intervals:
                if a <= x <= b:
                    x_prediction = 1
                    break

            # Classifier made a mistake, count it as error
            if x_prediction != true_prediction:
                errors += 1

        # Return ratio of errors to correct predictions
        return errors / num_labels

    def _calculate_true_error(self, intervals):
        # Positive intervals likely labelled as 1, negative likely labelled as 0
        positive_intervals = [(0.0, 0.2), (0.4, 0.6), (0.8, 1.0)]
        negative_intervals = [(0.2, 0.4), (0.6, 0.8)]
        error = 0

        # Intersection of the hypothesis with the positive intervals
        positive_intersection = self._find_intersection(positive_intervals, intervals)
        error += positive_intersection * 0.2
        error += (0.6 - positive_intersection) * 0.8

        # Intersection of the hypothesis with the negative intervals
        negative_intersection = self._find_intersection(negative_intervals, intervals)
        error += negative_intersection * 0.9
        error += (0.4 - negative_intersection) * 0.1

        return error

    def _find_intersection(self, first_intervals, other_intervals):
        intersection = 0

        # Calculate length of intersection of two sets of intervals
        for a, b in first_intervals:
            for c, d in other_intervals:
                x = max(a, c)
                y = min(b, d)
                length = max(y - x, 0)
                intersection += length

        return intersection

    def _calculate_penalty(self, delta, k, m):
        # Penalty for SRM: minimizing this keeps model complexity small to avoid overfitting
        # VC dimension: 2k
        x = 8 / m
        y = (m * np.exp(1)) / k
        w = np.log(4 / delta)
        z = 2 * k * np.log(y)

        return np.sqrt(x * (z + w))

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)