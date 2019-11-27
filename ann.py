import pandas as pd
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset


def df_to_tensor(df):
  return torch.from_numpy(df.values).float()


class DataPreProcessor:
  def __init__(self, df, split_factor=0.75, columns_to_normalize=None):
    if columns_to_normalize is None:
      columns_to_normalize = ['heating', 'cooling', 'GASTW', 'GAREA']
    indices = np.random.permutation(df.shape[0])
    split = int(df.shape[0] * split_factor)
    tr = indices[:split]
    te = indices[split:]
    train = df.iloc[tr].copy()
    test = df.iloc[te].copy()

    self.norm_factors = {}
    for c in columns_to_normalize:
      col = train[c]
      self.norm_factors[c] = (col.min(), col.max() - col.min())
    self.train = DataLoader(PandaDataset(self.normalize(train)), batch_size=16, shuffle=True)
    self.test = DataLoader(PandaDataset(self.normalize(test)), batch_size=1, shuffle=True)

  def normalize(self, df):
    for c, (sub, div) in self.norm_factors.items():
      df[c] = (df[c] - sub) / div
    return df

  def denormalize(self, df, column='heating'):
    sub, div = self.norm_factors[column]
    df = df * div + sub
    return df


class PandaDataset(Dataset):
    def __init__(self, df):
      super(PandaDataset, self).__init__()
      self.n = df.shape[0]
      self.features = df_to_tensor(df.drop(columns=['heating', 'cooling']))
      self.labels = df_to_tensor(df[['heating']])

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def evaluate(panda_dataset, preprocessor, model):
  predictions = model(panda_dataset.features).detach().numpy()
  predictions = preprocessor.denormalize(predictions)
  expected = preprocessor.denormalize(panda_dataset.labels.numpy())
  diff = np.abs(predictions - expected) / expected
  # print(np.concatenate((predictions, expected), axis=1))
  return diff.mean()


if __name__ == '__main__':
    dataset = pd.read_csv('data/features.csv').set_index('EGID')
    dataset.dropna(inplace=True)

    n_features = dataset.iloc[1:2].drop(columns=['heating', 'cooling']).shape[1]
    n_output = 1
    n_hidden = int((n_features + n_output) / 2)

    data = DataPreProcessor(dataset)

    model = nn.Sequential(nn.Linear(n_features, n_hidden),
                          nn.Sigmoid(),
                          nn.Linear(n_hidden, n_hidden),
                          nn.Sigmoid(),
                          nn.Linear(n_hidden, n_output),
                          nn.Sigmoid())
    criterionH = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=0.003)

    for j in range(20000):
      losses = 0
      for feat, target in data.train:
        optimizer.zero_grad()

        out = model(feat)

        loss = criterionH(out, target)
        loss.backward()
        optimizer.step()

        out_d = data.denormalize(out)
        target_d = data.denormalize(target)
        # losses += torch.mean(abs((out_d - target_d) / target_d))
        losses += loss.item()
      if j % 100 == 0:
        print("Step {j}: relative average error on train: {e:.03}%. Training avg loss: {l}"
              .format(j=j, e=evaluate(data.train.dataset, data, model) * 100, l=losses / len(data.train)))
