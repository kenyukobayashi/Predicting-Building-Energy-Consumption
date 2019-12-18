import multiprocessing

import pandas as pd

import svr
import regression
import ann
from cross_validation import CrossValidation


def run(data):
  _, regr = regression.run_training_least_square(data)
  _, sv = svr.run_training(data)
  print('.')
  _, an = ann.run_ann(data)
  print(regr, sv, an)
  return regr, sv, an


if __name__ == '__main__':
  dataset = pd.read_csv('data/sanitized_complete.csv').set_index('EGID')
  dataset.dropna(inplace=True)

  k = 16
  results = multiprocessing.Pool(k).map(run, CrossValidation(dataset, k))

  reg = [x for x, _, _ in results]
  sv = [x for _, x, _ in results]
  an = [x for _, _, x in results]
  print(reg)
  print(sv)
  print(an)
