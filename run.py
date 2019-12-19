import pandas as pd
import numpy as np

import svr
import regression
import ann
from daily.ann import run_daily_ann
from cross_validation import CrossValidation
from sys import argv

BUILDINGS_FILE = 'data/sanitized_complete.csv'
DAILY_WEATHER_FILE = 'data/daily_forecast.csv'
DAILY_PREDICTIONS_FILE = 'data/daily_predictions.csv'


def annual_predictions(method):
  """Run the `method` for annual prediction with 4-fold cross validation"""
  dataset = pd.read_csv(BUILDINGS_FILE).set_index('EGID').dropna()
  ln_q = []
  for k, data in enumerate(CrossValidation(dataset, 4)):
    print('Cross validation step k=%i' % k)
    if method == 'regression':
      ln_q.append(regression.run_training_least_square(data)[1])
    if method == 'svr':
      ln_q.append(svr.run_training(data)[1])
    if method == 'ann':
      ln_q.append(ann.run_ann(data)[1])
  return ln_q


def daily_prediction():
  """Run the ANN for daily prediction with 4-fold cross validation"""
  building_features = pd.read_csv(BUILDINGS_FILE).set_index('EGID').drop(columns=['heating', 'cooling'])
  weather_forecast = pd.read_csv(DAILY_WEATHER_FILE).set_index('timestamp')

  labels = pd.read_csv(DAILY_PREDICTIONS_FILE).set_index('timestamp')
  return run_daily_ann(building_features, weather_forecast, labels)


def main(arg):
  """
  Usage: `python3 run.py [regression|svr|ann|daily]`
  Runs the models with 4-fold cross validation:
    - for 'regression', 'svr' or 'ann', runs the annual prediction for the regression (least square, baseline), svr or the neural network
    - for 'daily', runs the neural network for daily prediction
  Note that the ANNs can take a long time to run, so their progresses are printed during learning.
  At the end, prints the average Ln Q error on testing and the standard deviation.
  """

  if len(arg) < 2:
    print('Missing argument')
    print(main.__doc__)
    return 1

  method = arg[1]

  if method in ['regression', 'svr', 'ann']:
    ln_q = annual_predictions(method)
  elif method == 'daily':
    print('WARNING: daily ANN prediction with cross validation takes a very long time to run, up to 30 min.')
    ln_q = daily_prediction()
  else:
    print('Incorrect argument')
    print(main.__doc__)
    return 2

  print('Mean Ln Q error: %f, standard deviation: %f' % (np.mean(ln_q), np.std(ln_q)))
  return 0


if __name__ == '__main__':
  err = main(argv)
  exit(err)
