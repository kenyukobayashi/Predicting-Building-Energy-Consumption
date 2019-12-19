import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from annual.preprocessor import DataPreProcessor


class Svr:
  """Support Vector Regression machine"""

  def __init__(self, data: DataPreProcessor, kernel_type, gamma,
               tolerance, epsilon, degree, coef0, c, shrinking):
    self.data = data
    self.regressor = SVR(kernel=kernel_type, gamma=gamma, tol=tolerance, epsilon=epsilon,
                         degree=degree, coef0=coef0, C=c, shrinking=shrinking)

  def fit(self, train_x, train_y):
    """Fits the SVR"""
    self.regressor.fit(train_x, train_y)

  def predict(self, x):
    """Makes a prediction for an input x"""
    return self.regressor.predict([x])

  def training(self):
    """
    Trains the model
    :returns: the average Ln Q error for training and testing at the end
    """
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
    ln_q_tr = self.data.evaluate(self.data.train, train_predictions)
    ln_q_te = self.data.evaluate(self.data.test, test_predictions)
    return ln_q_tr, ln_q_te


def grid_search(dataset):
  """Grid search algorithm to tune the hyper parameters"""
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
  tolerances = np.arange(0.0001, 0.002, 0.0001)
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


def run_training(data: DataPreProcessor):
  """
  Runs the training on the given dataset with the best SVR model
  :returns: the average Ln Q error for training and testing at the end
  """
  svr = Svr(data, 'poly', 'scale', 0.001, 0.11, 5, 0.61, 1.19, True)
  return svr.training()
