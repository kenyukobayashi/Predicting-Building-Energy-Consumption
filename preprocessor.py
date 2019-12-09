import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DataPreProcessor:
  def __init__(self, df, split=0.75, columns_to_normalize=None):
    if isinstance(split, float):
      indices = np.random.permutation(df.shape[0])
      split = int(df.shape[0] * split)
      tr = indices[:split]
      te = indices[split:]
    else:
      tr, te = split
    train = df.iloc[tr].copy()
    test = df.iloc[te].copy()

    print("Train log distribution:")
    print(np.log10(train['heating']).astype(int).value_counts())

    print("Test log distribution:")
    print(np.log10(test['heating']).astype(int).value_counts())

    self.compute_normal(train, columns_to_normalize)
    self.train = PandaDataset(self.normalize(train))
    self.train_loader = DataLoader(self.train, batch_size=16, shuffle=True)
    self.test = PandaDataset(self.normalize(test))
    self.test_loader = DataLoader(self.test, batch_size=1, shuffle=True)

  def compute_normal(self, tr, columns_to_normalize):
    if columns_to_normalize is None:
      # columns_to_normalize = ['heating', 'cooling', 'GASTW', 'GAREA']
      columns_to_normalize = ['GASTW', 'GAREA']
    self.norm_factors = {'heating': (0, 1e8)}
    for c in columns_to_normalize:
      col = tr[c]
      # self.norm_factors[c] = (0, col.max())
      self.norm_factors[c] = (col.min(), col.max() - col.min())

  def normalize(self, df):
    for c, (sub, div) in self.norm_factors.items():
      if c == 'heating':
        df[c] = np.log10(df[c])
      else:
        df[c] = (df[c] - sub) / div
    return df

  def denormalize(self, df, column='heating'):
    # sub, div = self.norm_factors.get(column, (0, 1))
    # df = df * div + sub
    # return df
    return np.power(df, 10)

  def evaluate(self, panda_dataset, predictions):
    predictions = self.denormalize(predictions)
    expected = self.denormalize(panda_dataset.labels_t.numpy())

    diff = predictions - expected
    rel_diff = (np.abs(diff) / expected).mean()
    mse = np.square(diff).mean()
    nb_above = 1.0 * np.sum(predictions > expected) / len(diff)
    return rel_diff, nb_above, mse


class PandaDataset(Dataset):
    def __init__(self, df):
      super(PandaDataset, self).__init__()
      self.n = df.shape[0]
      self.features = df.drop(columns=['heating', 'cooling'])
      self.labels = df[['heating']]
      self.features_t = PandaDataset.df_to_tensor(self.features)
      self.labels_t = PandaDataset.df_to_tensor(self.labels)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.features_t[index], self.labels_t[index]

    @staticmethod
    def df_to_tensor(df):
      return torch.from_numpy(df.values).float()
