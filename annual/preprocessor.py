import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class DataPreProcessor:
  """Preprocess and gives access to train and test data"""

  def __init__(self, df: pd.DataFrame, split=0.75, use_log=True):
    """
    Initialize this preprocessor
    :param df: Dataframe that contains buildings features and consumption labels
    :param split:
                  If it is a float, it randomly split the dataset between training and testing
                  where split is the proportion (between 0 and 1) of data in training
                  If it is a pair of arrays of EGIDs, it assigns the EGIDs of the first array to the training set,
                  and the one from the second array to testing set
    :param use_log: whether the targets should be changed to their log values
    """
    # split the data
    if isinstance(split, float):
      indices = np.random.permutation(df.shape[0])
      split = int(df.shape[0] * split)
      tr = indices[:split]
      te = indices[split:]
    else:
      tr, te = split
    train = df.iloc[tr].copy()
    test = df.iloc[te].copy()

    self.use_log = use_log
    # Normalize the data and create pytorch dataloader
    self.norm_factors = self.compute_normal(train)
    self.train = PandaDataset(self.normalize(train))
    self.train_loader = DataLoader(self.train, batch_size=16, shuffle=True)
    self.test = PandaDataset(self.normalize(test))
    self.test_loader = DataLoader(self.test, batch_size=1, shuffle=True)

  @staticmethod
  def compute_normal(tr):
    """Compute normalization values for the columns to normalize, based on the `tr` dataset"""
    columns_to_normalize = ['GASTW', 'GAREA']
    norm_factors = {'heating': (0, 1)}
    for c in columns_to_normalize:
      col = tr[c]
      norm_factors[c] = (col.min(), col.max() - col.min())
    return norm_factors

  def normalize(self, df):
    """Normalize the dataframe using the parameters already computed"""
    for c, (sub, div) in self.norm_factors.items():
      if c == 'heating' and self.use_log:
        df[c] = np.log10(df[c])
      else:
        df[c] = (df[c] - sub) / div
    return df

  def denormalize(self, df, column='heating'):
    """Denormalize a column/value to reverse `normalize`"""
    if column == 'heating' and self.use_log:
      return np.power(10, df)
    else:
      sub, div = self.norm_factors.get(column, (0, 1))
      df = df * div + sub
      return df

  def evaluate(self, panda_dataset, predictions):
    """
    Evaluate a prediction `predictions` compared to the expected value in the dataframe `panda_dataset`
    :returns: average Ln Q error
    """
    predictions = self.denormalize(predictions)
    expected = self.denormalize(panda_dataset.labels_t.numpy())

    diff = predictions - expected
    ln_q_err = np.abs(np.log(predictions / expected)).mean()
    return ln_q_err


class PandaDataset(Dataset):
  """
  Wrapper to convert a panda dataframe with labels (heating and cooling)
  and building features to a pytorch Dataset
  """

  def __init__(self, df):
    super(PandaDataset, self).__init__()
    self.n = df.shape[0]
    self.features = df.drop(columns=['heating', 'cooling'])  # Dataframe with the buildings features
    self.labels = df[['heating']]  # Dataframe with the heating predictions
    self.features_t = PandaDataset.df_to_tensor(self.features)
    self.labels_t = PandaDataset.df_to_tensor(self.labels)

  def __len__(self):
    return self.n

  def __getitem__(self, index):
    """:returns: two pytorch tensors for buildings feature and labels"""
    return self.features_t[index], self.labels_t[index]

  @staticmethod
  def df_to_tensor(df):
    return torch.from_numpy(df.values).float()
