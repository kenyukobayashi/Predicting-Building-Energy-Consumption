import numpy as np
import pandas as pd
import multiprocessing

from hourlyData import HourlyPreprocessor
from preprocessor import DataPreProcessor


class CrossValidation:

  def __init__(self, df, k):
    indices = np.random.permutation(df.shape[0])
    split_step = int(df.shape[0] / k)
    self.indices = np.array(np.split(indices[:split_step * k], indices_or_sections=k))
    self.df = df

  def __iter__(self):
    for ind in range(len(self.indices)):
      tr_ind = np.delete(self.indices, ind, axis=0).flatten()
      te_ind = self.indices[ind].flatten()
      yield DataPreProcessor(self.df, split=(tr_ind, te_ind))
