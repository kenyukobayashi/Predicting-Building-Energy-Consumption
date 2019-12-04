import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessor import DataPreProcessor
from sklearn.svm import SVR


class Svr:
  def __init__(self, data: DataPreProcessor, tolerance, epsilon, kernel_type='rbf', gamma='auto'):
    self.data = data
    self.regressor = SVR(kernel=kernel_type, gamma=gamma, tol=tolerance, epsilon=epsilon)

  def fit(self, train_x, train_y):
    self.regressor.fit(train_x, train_y)

  def predict(self, x):
    return self.regressor.predict([x])


if __name__ == '__main__':
  # Importing the dataset
  dataset = pd.read_csv('data/features.csv').set_index('EGID')
  dataset.dropna(inplace=True)

  # Instanciating the data preprocessor and the model
  data = DataPreProcessor(dataset)
  """
  On peut choisir parmis : 
  'linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ pour kernel type
  'scale' ou 'auto' ou float pour gamma
  """
  svr = Svr(data, tolerance=0.0000001, epsilon=0.00000001, kernel_type='rbf', gamma='auto')

  # Retrieving the different sets
  train_x = svr.data.train.features.to_numpy()
  train_y = svr.data.train.labels.to_numpy().ravel()
  test_x = svr.data.test.features.to_numpy()
  # test_y = svr.data.test.labels

  # Fitting the model
  svr.fit(train_x, train_y)

  # Visualising the Support Vector Regression results
  train_predictions = np.array([svr.predict(x) for x in train_x])
  test_predictions = np.array([svr.predict(x) for x in test_x])
  avg_diff, nb_above, loss = data.evaluate(data.train, train_predictions)
  avg_diff_te, nb_above_te, loss_te = data.evaluate(data.test, test_predictions)
  print("Relative average error on train: {e:.2f}%, on test: {te:.2f}%. "
        "Training avg loss: {l}, above target: {a:.2f}%"
        .format(e=avg_diff * 100, te=100 * avg_diff_te,
                l=loss, a=nb_above * 100.0))
