import multiprocessing

import numpy as np
import pandas as pd
from torch import nn
from torch import optim

from cross_validation import CrossValidation
from hourlyData import HourlyPreprocessor, HourlyDataset


class Ann:
  def __init__(self, data: HourlyPreprocessor, n_features, n_output, n_hidden):
    self.model = nn.Sequential(
      nn.Linear(n_features, n_hidden), nn.LeakyReLU(),
      nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(),

      nn.Linear(n_hidden, n_output), nn.LeakyReLU()
    )
    self.data = data
    self.criterionH = nn.MSELoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-8)

  def do_epoch(self):
    i = 0
    losses = 0
    mses = 0
    abv = 0
    for feat, target in self.data.train_loader:
      self.optimizer.zero_grad()
      out = self.model(feat)
      loss = self.criterionH(out, target)
      loss.backward()
      self.optimizer.step()

      i += 1
      t = self.data.denormalize("labels", target.detach().numpy())
      o = self.data.denormalize("labels", out.detach().numpy())
      o2 = o[t != 0]
      t2 = t[t != 0]
      losses += np.abs(np.log(o2/t2)).mean()
      mses += loss.data.item()
      t3 = target.detach().numpy()
      o3 = out.detach().numpy()[t3 != 0]
      t3 = t3[t3 != 0]
      abv += len(np.where(o3 > t3)[0]) / float(len(t3))

      if i % 500 == 0:
        print(i, '\t', losses / 500, '\t', mses / 500, '\t', abv / 500)
        losses = 0
        mses = 0
        abv = 0

  def fit(self, preprocessor):
    for j in range(10):
      self.do_epoch()
      print(j, preprocessor.evaluate(self.model, train=False))


def summer(df):
  start_summer = pd.Timestamp(year=2017, month=3, day=15)
  end_summer = pd.Timestamp(year=2017, month=11, day=15)
  df = df[start_summer < df.index.map(pd.Timestamp)]
  return df[df.index.map(pd.Timestamp) < end_summer].index


def svrf():
  x_tr, y_tr = preprocessor.train.to_numpy()
  x_te, y_te = preprocessor.test.to_numpy()
  # train_dataset_array = preprocessor.train_loader.numpy() #next(iter(preprocessor.train_loader))[0].numpy()
  print(x_tr.shape, y_tr.shape)

  # Best parameters
  tolerance = 0.001
  kernel_type = 'poly'
  gamma = 'scale'
  degree = 5
  coef0 = 0.61
  epsilon = 0.11
  c = 1.19
  shrinking = True
  svr = svr.Svr(None, kernel_type, gamma, tolerance, epsilon, degree, coef0, c, shrinking)
  svr.fit(x_tr, y_tr)
  print('fitted')
  pred = svr.regressor.predict(x_te)

  expected = preprocessor.denormalize('labels', y_te)
  predictions = preprocessor.denormalize('labels', pred)[expected != 0]
  expected = expected[expected != 0]

  rel_diff = (np.abs(predictions - expected) / expected).mean()
  abv = (predictions > expected).mean()
  lnq = np.abs(np.log(expected / predictions)).mean()

  print(rel_diff)
  print(abv)
  print(lnq)


if __name__ == '__main__':
  useful_weather = ['G_Dh_mean', 'G_Bn_mean', 'G_Dh_var', 'G_Bn_var', 'Ta_mean', 'Ta_var', 'Ta_min',
                    'FF_mean', 'FF_var', 'h_day']
  buildings = pd.read_csv('data/sanitized_complete.csv') \
    .set_index('EGID') \
    .drop(columns=['heating', 'cooling'])
  forecast = pd.read_csv('data/daily_forecast.csv').set_index('timestamp')[useful_weather]
  summer = summer(forecast)

  forecast.drop(summer, inplace=True)

  print(forecast.shape)
  predictions = pd.read_csv('data/daily_predictions.csv').set_index('timestamp').drop(summer)
  dataset = HourlyDataset(
      building_features=buildings,
      weather_forecast=forecast,
      labels=predictions
    )

  print('loaded')

  def inner(data: HourlyPreprocessor):
    ann = Ann(data,
              n_features=data.train.nb_features,
              n_output=1,
              n_hidden=50)
    ann.fit(data)
    return data.evaluate(ann.model, train=True), data.evaluate(ann.model, train=False)

  loss = multiprocessing.Pool(4).map(inner, CrossValidation(dataset, 4))
  tr_loss = [x for x, _ in loss]
  te_loss = [x for _, x in loss]

  print(np.mean(te_loss))
  print(np.mean(tr_loss))
  print(np.std(te_loss))
  print(np.std(tr_loss))

  # losses = []
  # relative_diff = []

  # load_forecast()
  # print(pd.Timestamp(year=2017, month=1, day=1, hour=0) + pd.Timedelta(hours=1))
  # filter_columns()
  # filter_forecast()
