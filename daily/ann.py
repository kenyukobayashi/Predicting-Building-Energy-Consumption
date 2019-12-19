import pandas as pd
from torch import nn
from torch import optim

from daily.data import DailyPreprocessor, DailyDataset, DailyCrossValidation


class Ann:
  """Neural network optimized for daily prediction"""
  def __init__(self, data: DailyPreprocessor, n_features, n_output, n_hidden):
    """Initialize the ANN with the best parameters found"""
    self.model = nn.Sequential(
      nn.Linear(n_features, n_hidden), nn.LeakyReLU(),
      nn.Linear(n_hidden, n_output), nn.LeakyReLU()
    )
    self.data = data
    self.criterionH = nn.MSELoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-8)

  def do_epoch(self):
    """Execute single epoch of learning on the training set"""
    for feat, target in self.data.train_loader:
      self.optimizer.zero_grad()
      out = self.model(feat)
      loss = self.criterionH(out, target)
      loss.backward()
      self.optimizer.step()

  def fit(self, preprocessor: DailyPreprocessor):
    """Executes the optimum number of epochs to learn the model from the data of `preprocessor`"""
    for j in range(5):
      self.do_epoch()
      print('Epoch:', j, 'Ln Q error on test:', preprocessor.evaluate(self.model, train=False))


def get_summer_days(df):
  """Returns warm days that should be removed when predicting heating needs"""
  start_summer = pd.Timestamp(year=2017, month=3, day=15)
  end_summer = pd.Timestamp(year=2017, month=11, day=15)
  df = df[start_summer < df.index.map(pd.Timestamp)]
  return df[df.index.map(pd.Timestamp) < end_summer].index


def run_daily_ann(building_features, weather_forecast, labels):
  """
  Runs 4-fold cross validation of daily prediction for the best ANN model with provided data
  :returns: Ln Q errors on testing set for each fold
  """
  useful_weather = ['G_Dh_mean', 'G_Bn_mean', 'G_Dh_var', 'G_Bn_var', 'Ta_mean', 'Ta_var', 'Ta_min',
                    'FF_mean', 'FF_var', 'h_day']
  weather_forecast = weather_forecast[useful_weather].copy()
  summer = get_summer_days(weather_forecast)
  weather_forecast.drop(summer, inplace=True)
  labels.drop(summer, inplace=True)

  dataset = DailyDataset(
      building_features=building_features,
      weather_forecast=weather_forecast,
      labels=labels
    )

  ln_q = []
  for k, data in enumerate(DailyCrossValidation(dataset, 4)):
    print('Cross validation step:', k)
    ann = Ann(data,
              n_features=data.train.nb_features,
              n_output=1,
              n_hidden=50)
    ann.fit(data)
    ln_q.append(data.evaluate(ann.model, train=False))
  return ln_q
