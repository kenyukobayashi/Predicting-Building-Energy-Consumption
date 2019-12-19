from torch import nn
from torch import optim

from annual.preprocessor import DataPreProcessor


class Ann:
  """Best ANN version for annual predictions of heating needs"""
  def __init__(self, data: DataPreProcessor, n_features: int, n_output: int, n_hidden: int):
    """Initialize an ANN for the given data and with provided number of neurons in each layer"""
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
    """
    Do one epoch of learning
    If `evaluate` is True, will return the Ln Q error for training and testing set at the end of the epoch
    """
    for feat, target in self.data.train_loader:
      self.optimizer.zero_grad()
      out = self.model(feat)
      loss = self.criterionH(out, target)
      loss.backward()
      self.optimizer.step()

    if evaluate:
      train_predictions = self.model(self.data.train.features_t).detach().numpy()
      test_predictions = self.model(self.data.test.features_t).detach().numpy()
      ln_q_tr = self.data.evaluate(self.data.train, train_predictions)
      ln_q_te = self.data.evaluate(self.data.test, test_predictions)
      return ln_q_tr, ln_q_te
    else:
      return None, None


def run_ann(data: DataPreProcessor):
  """
  Runs the best ANN model for yearly heating prediction,
  :returns: the average Ln Q error for training and testing at the end
  """
  ann = Ann(data,
            n_features=data.train.features.shape[1],
            n_output=1,
            n_hidden=8)

  for j in range(5000):
    ln_q_tr, ln_q_te = ann.do_epoch(j % 1000 == 0)
    if j % 1000 == 0:
      print('Epoch %d/5000: Ln Q error on test: %f' % (j, ln_q_te))
  ln_q_tr, ln_q_te = ann.do_epoch(True)
  return ln_q_tr, ln_q_te
