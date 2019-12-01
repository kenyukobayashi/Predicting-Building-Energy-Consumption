import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from preprocessor import DataPreProcessor


# try SVM
# get baseline
# ReLu and no normalize
# adam optim


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
    self.optimizer = optim.SGD(self.model.parameters(), lr=0.010)
    self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=1000, cooldown=5000, verbose=True)

  def do_epoch(self):
    for feat, target in data.train_loader:
      self.optimizer.zero_grad()
      out = self.model(feat)
      loss = self.criterionH(out, target)
      loss.backward()
      self.optimizer.step()
    train_predictions = self.model(self.data.train.features_t).detach().numpy()
    test_predictions = self.model(self.data.test.features_t).detach().numpy()
    avg_diff, nb_above, loss = data.evaluate(self.data.train, train_predictions)
    avg_diff_te, nb_above_te, loss_te = data.evaluate(self.data.test, test_predictions)
    self.scheduler.step(avg_diff)
    return avg_diff, nb_above, loss, avg_diff_te, nb_above_te, loss_te

  def get_lr(self):
    return self.optimizer.param_groups[0]['lr']


if __name__ == '__main__':
    dataset = pd.read_csv('data/features.csv').set_index('EGID')
    dataset.dropna(inplace=True)
    data = DataPreProcessor(dataset)

    ann = Ann(data,
              n_features=dataset.iloc[1:2].drop(columns=['heating', 'cooling']).shape[1],
              n_output=1,
              n_hidden=8)

    losses = []
    relative_diff = []
    for j in range(20_000):
      avg_diff, nb_above, loss, avg_diff_te, nb_above_te, loss_te = ann.do_epoch()
      if j % 100 == 0:
        losses.append((loss, loss_te))
        relative_diff.append((avg_diff, avg_diff_te))
        print("Step {j}: relative average error on train: {e:.2f}%, on test: {te:.2f}%. "
              "LR={lr:.2} Training avg loss: {l}, above target: {a:.2f}%"
              .format(j=j, e=avg_diff * 100, te=100 * avg_diff_te,
                      l=loss, lr=ann.get_lr(), a=nb_above * 100.0))

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(losses[10:])
    ax2.plot(relative_diff[10:])

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('MSE')
    ax2.set_ylabel('relative difference')
    fig.legend(["train", "test"], loc='center left')

    plt.show()
