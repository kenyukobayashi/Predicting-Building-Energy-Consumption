import numpy as np
from annual.preprocessor import DataPreProcessor


def ridge_regression(y, tx, lambda_):
  """implement ridge regression."""
  aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
  a = tx.T.dot(tx) + aI
  b = tx.T.dot(y)
  return np.linalg.solve(a, b)


def least_squares(y, tx):
  """Calculate the least squares solution."""
  a = tx.T.dot(tx)
  b = tx.T.dot(y)
  return np.linalg.solve(a, b)


def run_training_least_square(data: DataPreProcessor):
  """
  Run the training for the least square
  :returns: the average Ln Q error for training and testing at the end
  """
  # Extract data
  x_test = data.test.features.to_numpy()
  x_train = data.train.features.to_numpy()
  y_train = data.train.labels.to_numpy()

  # Compute weights of least square
  weight = least_squares(y_train, x_train)

  # Compute relative diff
  rel_dif_test = data.evaluate(data.test, x_test.dot(weight))
  rel_dif_train = data.evaluate(data.train, x_train.dot(weight))
  return rel_dif_train, rel_dif_test
