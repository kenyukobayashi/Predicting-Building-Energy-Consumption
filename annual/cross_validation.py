import numpy as np

from annual.preprocessor import DataPreProcessor


class CrossValidation:
  """Cross validation for annual prediction models"""
  def __init__(self, df, k):
    """Initialize a cross validation with `k` folds"""
    indices = np.random.permutation(df.shape[0])
    split_step = int(df.shape[0] / k)
    self.indices = np.array(np.split(indices[:split_step * k], indices_or_sections=k))
    self.df = df

  def __iter__(self):
    """Iterate on each fold of the cross validation"""
    for ind in range(len(self.indices)):
      tr_ind = np.delete(self.indices, ind).flatten()
      te_ind = self.indices[ind].flatten()
      yield DataPreProcessor(self.df, split=(tr_ind, te_ind))
