import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessor import DataPreProcessor
from cross_validation import CrossValidation
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


class Svr:
    def __init__(self, data: DataPreProcessor, kernel_type, gamma,
                 tolerance, epsilon, degree, coef0, c, shrinking):
        self.data = data
        self.regressor = SVR(kernel=kernel_type, gamma=gamma, tol=tolerance, epsilon=epsilon,
                             degree=degree, coef0=coef0, C=c, shrinking=shrinking)

    def fit(self, train_x, train_y):
        self.regressor.fit(train_x, train_y)

    def predict(self, x):
        return self.regressor.predict([x])

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


def compute_mean(l, k):  # def compute_mean(l:list[tuple[float, float]], k:list[tuple[float, float]]):
    mean_train = 0
    mean_te = 0
    for (u, v) in l:
        mean_train += u
        mean_te += v
    return mean_train / k, mean_te / k


def grid_search(dataset):
    # Load the data without splitting it for GridsearchCV
    data = DataPreProcessor(dataset)
    features = data.train.features.to_numpy()
    labels = data.train.labels.to_numpy().ravel()
    # features = (data.train.features + data.test.features).to_numpy()
    # labels = (data.train.labels + data.test.labels).to_numpy().ravel()

    """
    # First gridsearch
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
                  #'tol': tolerances
                  }
    """

    # Second gridsearch
    parameters = {'kernel': ['poly'],
                  'gamma': ['scale'],
                  'degree': [5, 6, 7],
                  'coef0': np.arange(0.59, 0.61, 0.01),
                  'epsilon': np.arange(0.1,0.12,0.001),
                  'C': np.arange(1,1.2,0.01),
                  'shrinking': [True],
                  'tol': np.arange(0.001,0.01,0.001)
                  }

    svr = SVR()
    clf = GridSearchCV(svr, parameters)
    clf.fit(features, labels)
    print(clf.best_params_)

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


if __name__ == '__main__':
    # Importing the dataset
    dataset = pd.read_csv('data/sanitized_complete.csv').set_index('EGID')
    dataset.dropna(inplace=True)

    #for data in CrossValidation(dataset, 4):
    #    print(run_training(data))

    # Grid search
    #grid_search(dataset)

    # Best parameters
    tolerance = 0.001
    kernel_type = 'poly'
    gamma = 'scale'
    degree = 5
    coef0 = 0.61
    epsilon = 0.11
    c = 1.19
    shrinking = True

    # Training
    mean_loss, mean_loss_te, mean_avg_diff, mean_avg_diff_te, mean_nb_above, mean_nb_above_te \
        = cross_validation_training(dataset, kernel_type, gamma, tolerance, epsilon,
                                    degree, coef0, c, shrinking)

    print("\nWith 4-fold Cross Validation : \n"
          "ln Q error on train: {e:.2f}%, on test: {te:.2f}%. \n"
          "Training avg loss: {l} \n"
          "Training above target: {a:.2f}%"
          .format(e=mean_avg_diff * 100, te=100 * mean_avg_diff_te,
                  l=mean_loss, a=mean_nb_above * 100.0))
