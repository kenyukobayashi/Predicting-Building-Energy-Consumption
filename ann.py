import multiprocessing

import numpy as np
import pandas as pd
from torch import nn
from torch import optim

from cross_validation import CrossValidation
from preprocessor import DataPreProcessor


class Ann:
  def __init__(self, data: DataPreProcessor, n_features, n_output, n_hidden):
    self.model = nn.Sequential(
      nn.Linear(n_features, n_hidden),
      nn.LeakyReLU(),
      nn.Linear(n_hidden, n_output),
      nn.LeakyReLU()
    )
    self.data = data
    self.criterionH = nn.MSELoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)

  def do_epoch(self, evaluate=True):
    for feat, target in self.data.train_loader:
      self.optimizer.zero_grad()
      out = self.model(feat)
      loss = self.criterionH(out, target)
      loss.backward()
      self.optimizer.step()

    if evaluate:
      train_predictions = self.model(self.data.train.features_t).detach().numpy()
      test_predictions = self.model(self.data.test.features_t).detach().numpy()
      avg_diff, nb_above, loss = self.data.evaluate(self.data.train, train_predictions)
      avg_diff_te, nb_above_te, loss_te = self.data.evaluate(self.data.test, test_predictions)
      return avg_diff, nb_above, loss, avg_diff_te, nb_above_te, loss_te
    else:
      return None, None, None, None, None, None

  def get_lr(self):
    return self.optimizer.param_groups[0]['lr']


def fabc(data):
  ann = Ann(data,
            n_features=dataset.iloc[1:2].drop(columns=['heating', 'cooling']).shape[1],
            n_output=1,
            n_hidden=nh)

  for j in range(8000):
    avg_diff, nb_above, loss, avg_diff_te, nb_above_te, loss_te = ann.do_epoch( j % 1000 == 0)
    if j % 1000 == 0:
      # losses.append((loss, loss_te))
      # relative_diff.append((avg_diff, avg_diff_te))
      print("Step {j}: relative average error on train: {e:.2f}%, on test: {te:.2f}%. "
            "LR={lr:.2} Training avg loss: {l}, above target: {a:.2f}%"
            .format(j=j, e=avg_diff * 100, te=100 * avg_diff_te,
                    l=loss, lr=ann.get_lr(), a=nb_above * 100.0))
  avg_diff, _, _, avg_diff_te, _, _ = ann.do_epoch(True)
  return (avg_diff, avg_diff_te)


if __name__ == '__main__':
    dataset = pd.read_csv('data/sanitized_complete.csv').set_index('EGID')
    dataset.dropna(inplace=True)

    # n = [22, 24, 26, 28, 30]
    n = [10]
    te_losses = []
    tr_losses = []
    te_stds = []
    tr_stds = []
    for nh in n:
      te_loss = []
      tr_loss = []
      loss = multiprocessing.Pool(4).map(fabc, CrossValidation(dataset, 4))
      tr_loss = [x for x, _ in loss]
      te_loss = [x for _, x in loss]

      te_losses.append(np.mean(te_loss))
      tr_losses.append(np.mean(tr_loss))
      te_stds.append(np.std(te_loss))
      tr_stds.append(np.std(tr_loss))
      print(nh)
      print(tr_losses)
      print(tr_stds)
      print(te_losses)
      print(te_stds)
