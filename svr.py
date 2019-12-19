import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessor import DataPreProcessor
from cross_validation import CrossValidation
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


# Support Vector Regression machine
class Svr:
    def __init__(self, data: DataPreProcessor, kernel_type, gamma,
                 tolerance, epsilon, degree, coef0, c, shrinking):
        self.data = data
        self.regressor = SVR(kernel=kernel_type, gamma=gamma, tol=tolerance, epsilon=epsilon,
                             degree=degree, coef0=coef0, C=c, shrinking=shrinking)

    # Fits the SVR
    def fit(self, train_x, train_y):
        self.regressor.fit(train_x, train_y)

    # Makes a prediction for an input x
    def predict(self, x):
        return self.regressor.predict([x])

    # Trains the model and returns the different losses
    def training(self, data: DataPreProcessor):
        # Retrieving the different sets
        train_x = self.data.train.features.to_numpy()
        train_y = self.data.train.labels.to_numpy().ravel()
        test_x = self.data.test.features.to_numpy()
        # test_y = svr.data.test.labels

        # Fitting the model
        self.fit(train_x, train_y)

        # Computing the predictions and losses
        train_predictions = np.array([self.predict(x) for x in train_x])
        test_predictions = np.array([self.predict(x) for x in test_x])
        avg_diff, nb_above, loss = data.evaluate(data.train, train_predictions)
        avg_diff_te, nb_above_te, loss_te = data.evaluate(data.test, test_predictions)
        return avg_diff, nb_above, loss, avg_diff_te, nb_above_te, loss_te


# Trains the model and returns the different losses with a cross validation process
def cross_validation_training(dataset, kernel_type, gamma, tolerance, epsilon,
                              degree, coef0, c, shrinking):
    losses = []
    relative_diffs = []
    nbs_above = []

    for data in CrossValidation(dataset, 4):
        svr = Svr(data, kernel_type, gamma, tolerance, epsilon, degree, coef0, c, shrinking)
        avg_diff, nb_above, loss, avg_diff_te, nb_above_te, loss_te = svr.training(data)

        # Appending the results
        losses.append((loss, loss_te))
        relative_diffs.append((avg_diff, avg_diff_te))
        nbs_above.append((nb_above, nb_above_te))

    # Computing the mean of the results
    mean_loss, mean_loss_te = compute_mean(losses, 4)
    mean_avg_diff, mean_avg_diff_te = compute_mean(relative_diffs, 4)
    mean_nb_above, mean_nb_above_te = compute_mean(nbs_above, 4)

    return mean_loss, mean_loss_te, mean_avg_diff, mean_avg_diff_te, mean_nb_above, mean_nb_above_te


# Computes the mean in a list of length k. Used for the cross validation process.
def compute_mean(l, k):  # def compute_mean(l:list[tuple[float, float]], k:list[tuple[float, float]]):
    mean_train = 0
    mean_te = 0
    for (u, v) in l:
        mean_train += u
        mean_te += v
    return mean_train / k, mean_te / k


# Grid search algorithm to tune the hyper parameters
def grid_search(dataset):
    data = DataPreProcessor(dataset)
    features = data.train.features.to_numpy()
    labels = data.train.labels.to_numpy().ravel()

    # Parameters of SVR
    kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']  # Specifies the kernel type to be used
    gammas = ['scale', 'auto']  # Kernel coefficient. Use if kernel_type = ‘rbf’, ‘poly’ or ‘sigmoid’
    degrees = np.arange(10)  # Degree of the polynomial kernel function. Use if kernel_type = poly
    coef0s = np.arange(0, 1, 0.1)  # Independent term in kernel function. Use if kernel_type = poly or sigmoid
    epsilons = np.arange(0.05, 0.21, 0.02)  # Epsilon in the epsilon-SVR model
    cs = np.arange(0.1, 3.1, 0.5)  # regularization parameter
    shrinkings = [True, False]  # whether to use the shrinking heuristic
    tolerances = np.arange(0.0001,0.002,0.0001)
    parameters = {'kernel': kernel_types,
                  'gamma': gammas,
                  'degree': degrees,
                  'coef0': coef0s,
                  'epsilon': epsilons,
                  'C': cs,
                  'shrinking': shrinkings,
                  'tol': tolerances
                  }

    """
    # Second gridsearch
    parameters = {'kernel': ['poly'],
                  'gamma': ['scale'],
                  'degree': [5, 6, 7],
                  'coef0': np.arange(0.59, 0.61, 0.01),
                  'epsilon': np.arange(0.01,0.12,0.01),
                  'C': np.arange(1,1.2,0.01),
                  'shrinking': [True],
                  'tol': np.arange(0.001,0.01,0.001)
                  }
    """

    svr = SVR()
    clf = GridSearchCV(svr, parameters)
    clf.fit(features, labels)
    print(clf.best_params_)


# Runs the training and returns the losses for a DataPreProcessor input
def run_training(data : DataPreProcessor):
    svr = Svr(data, 'poly', 'scale', 0.001, 0.11, 5, 0.61, 1.19, True)
    # Retrieving the different sets
    train_x = svr.data.train.features.to_numpy()
    train_y = svr.data.train.labels.to_numpy().ravel()
    test_x = svr.data.test.features.to_numpy()
    # test_y = svr.data.test.labels

    # Fitting the model
    svr.fit(train_x, train_y)

    # Computing the predictions and losses
    train_predictions = np.array([svr.predict(x) for x in train_x])
    test_predictions = np.array([svr.predict(x) for x in test_x])
    avg_diff, _, _ = data.evaluate(data.train, train_predictions)
    avg_diff_te, _, _ = data.evaluate(data.test, test_predictions)
    return avg_diff, avg_diff_te


# Makes a graph of the lnQ error depending on the kernel type
def make_graph(dataset):
    kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']
    te_errors = []
    for kernel_type in kernel_types :
        ln_q_te_errs = []
        for data in CrossValidation(dataset, 4):
            svr = Svr(data, kernel_type, 'scale', 0.001, 0.11, 5, 0.61, 1.19, True)

            # Retrieving the different sets
            train_x = svr.data.train.features.to_numpy()
            train_y = svr.data.train.labels.to_numpy().ravel()
            test_x = svr.data.test.features.to_numpy()

            # Fitting the model
            svr.fit(train_x, train_y)

            # Computing the test prediction and test ln q error
            test_predictions = np.array([svr.predict(x) for x in test_x])
            ln_q_te_err, _, _ = data.evaluate(data.test, test_predictions)

            # Appending the ln q error to the list
            ln_q_te_errs.append(ln_q_te_err)

        # Appending the ln q error for the given kernel type to the errors list
        sum_er = 0
        for e in ln_q_te_errs :
            sum_er += e
        te_errors.append(sum_er/4)

    # Plotting
    plt.xlabel("Kernel Type")
    plt.ylabel("log Ln Q error")
    plt.grid(True)
    plt.plot(kernel_types, np.log(te_errors), color='blue', marker='o',linewidth=1, markersize=4)
    plt.show()