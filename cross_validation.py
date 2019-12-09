import numpy as np
import pandas as pd

from preprocessor import DataPreProcessor


class CrossValidation:

  def __init__(self, df, k):
    indices = np.random.permutation(df.shape[0])
    split_step = int(df.shape[0] / k)
    self.indices = np.array(np.split(indices[:split_step * k], indices_or_sections=k))
    self.df = df

  def __iter__(self):
    for ind in range(len(self.indices)):
      tr_ind = np.delete(self.indices, ind).flatten()
      te_ind = self.indices[ind].flatten()
      yield DataPreProcessor(self.df, split=(tr_ind, te_ind))


if __name__ == '__main__':
  dataset = pd.read_csv('data/features.csv').set_index('EGID')
  dataset.dropna(inplace=True)
  for i in CrossValidation(dataset.iloc[:4], 4):
    print(i)