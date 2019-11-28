import pandas as pd
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


# try SVM
# get baseline
# ReLu and no normalize
#

def df_to_tensor(df):
  return torch.from_numpy(df.values).float()


class DataPreProcessor:
  def __init__(self, df, split_factor=0.75, columns_to_normalize=None):
    indices = np.random.permutation(df.shape[0])
    split = int(df.shape[0] * split_factor)
    tr = indices[:split]
    te = indices[split:]
    train = df.iloc[tr].copy()
    test = df.iloc[te].copy()

    self.compute_normal(train, columns_to_normalize)
    self.train = DataLoader(PandaDataset(self.normalize(train)), batch_size=16, shuffle=True)
    self.test = DataLoader(PandaDataset(self.normalize(test)), batch_size=1, shuffle=True)

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
      df[c] = (df[c] - sub) / div
    return df

  def denormalize(self, df, column='heating'):
    sub, div = self.norm_factors.get(column, (0, 1))
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

  diff = predictions - expected
  rel_diff = (np.abs(diff) / expected).mean()
  mse = np.square(diff).mean()
  nb_above = 1.0 * np.sum(predictions > expected) / len(diff)
  return rel_diff, nb_above, mse


if __name__ == '__main__':
    dataset = pd.read_csv('data/features.csv').set_index('EGID')
    dataset.dropna(inplace=True)

    n_features = dataset.iloc[1:2].drop(columns=['heating', 'cooling']).shape[1]
    n_output = 1
    n_hidden = 8 #int((n_features + n_output) / 2)

    data = DataPreProcessor(dataset)

    model = nn.Sequential(nn.Linear(n_features, n_hidden),
                          nn.LeakyReLU(),
                          # nn.Linear(n_hidden, n_hidden),
                          # nn.Sigmoid(),
                          nn.Linear(n_hidden, n_output),
                          nn.LeakyReLU()
                          # nn.Linear(n_output, n_output),
                          # nn.ELU()
                          )
    criterionH = nn.MSELoss()

    # adam optim
    optimizer = optim.SGD(model.parameters(), lr=0.010)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1000, cooldown=5000, verbose=True, )

    losses = []
    for j in range(20_000):
      for feat, target in data.train:
        optimizer.zero_grad()

        out = model(feat)

        loss = criterionH(out, target)
        loss.backward()
        optimizer.step()

        out_d = data.denormalize(out)
        target_d = data.denormalize(target)

      avg_diff, nb_above, loss = evaluate(data.train.dataset, data, model)
      avg_diff_te, nb_above_te, loss_te = evaluate(data.test.dataset, data, model)
      losses.append((loss, loss_te))
      scheduler.step(avg_diff)
      if j % 100 == 0:
        print("Step {j}: relative average error on train: {e:.2f}%, on test: {te:.2f}%. LR={lr:.2} Training avg loss: {l}, above target: {a:.2f}%"
              .format(j=j, e=avg_diff * 100, te=100 * avg_diff_te, l=loss,
                      lr=[group['lr'] for group in optimizer.param_groups][0], a=nb_above * 100.0))
    plt.plot(losses)
    plt.ylabel("MSE")
    plt.xlabel("Epoch")
    plt.legend(["train", "test"])
    plt.show()
