import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from preprocessor import DataPreProcessor


def ridge_regression(y, tx, lambda_):
  """implement ridge regression."""
  aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
  a = tx.T.dot(tx) + aI
  b = tx.T.dot(y)
  return np.linalg.solve(a, b)


def least_squares(y, tx):
  """calculate the least squares solution."""
  a = tx.T.dot(tx)
  b = tx.T.dot(y)
  return np.linalg.solve(a, b)


def build_model_data(height, weight):
  """Form (y,tX) to get regression data in matrix form."""
  y = weight
  x = height
  num_samples = len(y)
  tx = np.c_[np.ones(num_samples), x]
  return y, tx


def compute_mse(y, tx, w):
  """compute the loss by mse."""
  e = y - tx.dot(w)
  mse = e.dot(e.T) / (2 * len(e))
  return mse


def build_poly(x, degree):
  """polynomial basis functions for input data x, for j=0 up to j=degree."""
  poly = np.ones((len(x), 1))
  for deg in range(1, degree + 1):
    poly = np.c_[poly, np.power(x, deg)]
  return poly


if __name__ == '__main__':
  dataset = pd.read_csv('data/features.csv').set_index('EGID')
  dataset.dropna(inplace=True)
  data = DataPreProcessor(dataset)
  x_test = data.test.features.to_numpy()
  x_train = data.train.features.to_numpy()
  y_train = data.train.labels.to_numpy()
  lambdas = np.logspace(-5, 0, 15)

  weight = least_squares(y_train, x_train)
  rel_dif, nb_above, mse = data.evaluate(data.test, x_test.dot(weight))
  print("Least squares", "Relative diff: ", rel_dif, "Nb above: ", nb_above, "MSE: ", mse)

  relDiffTab = []
  lambdaTab = []
  for i in lambdas:
    weight = ridge_regression(y_train, x_train, i)
    rel_dif, nb_above, mse = data.evaluate(data.test, x_test.dot(weight))
    relDiffTab.append(rel_dif)
    print("Ridge regression for ", i, "Relative diff: ", rel_dif, "Nb above: ", nb_above, "MSE: ", mse)

  plt.xlabel("Lambas")
  plt.ylabel("Relative Diff")
  plt.title("Ridge regression")
  plt.plot(lambdas, relDiffTab)
  plt.show()
