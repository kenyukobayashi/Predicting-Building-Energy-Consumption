import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
from torch import optim
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from preprocessor import PandaDataset


class HourlyDataset(Dataset):
  def __init__(self, building_features, weather_forecast, labels):
    super(HourlyDataset, self).__init__()
    self.features = building_features.dropna().copy()
    self.forecast = weather_forecast.copy()

    non_nan_indices = [str(egid) for egid in self.features.index.tolist()]
    self.labels = labels[non_nan_indices].copy()

    self.h_in_years = self.forecast.shape[0]
    self.nb_buildings = self.features.shape[0]
    self.nb_features = self.features.shape[1] + self.forecast.shape[1]

  def __len__(self):
    return self.h_in_years * self.nb_buildings

  def __getitem__(self, index):
    building, hour = divmod(index, self.h_in_years)
    egid = self.labels.columns[building]
    timestamp = self.labels.index[hour]

    forecast_tensor = torch.tensor(self.forecast.loc[timestamp])
    building_tensor = torch.tensor(self.features.loc[int(egid)])
    return torch.cat((building_tensor, forecast_tensor), 0), torch.tensor([self.labels.loc[timestamp, egid]])

  def split_buildings(self, tr_egid, te_egid):
    return HourlyDataset(self.features.loc[tr_egid], self.forecast, self.labels), \
           HourlyDataset(self.features.loc[te_egid], self.forecast, self.labels)

  def apply_to_each(self, functions):
    for col, f in functions.items():
      for df in [self.features, self.forecast]:
        if col in df.columns:
          df[col] = df[col].apply(f)

    if 'labels' in functions:
      self.labels = self.labels.apply(functions['labels'])

  def get_min_max(self, col):
    for df in [self.features, self.forecast]:
      if col in df.columns:
        return df[col].min(), df[col].max()


class HourlyPreprocessor:
  def __init__(self, dataset: HourlyDataset, split=0.75):
    if isinstance(split, float):
      egids = dataset.features.index.tolist()
      np.random.shuffle(egids)
      split = int(len(egids) * split)
      tr = egids[:split]
      te = egids[split:]
    else:
      tr, te = split
    self.train, self.test = dataset.split_buildings(tr, te)
    self.compute_normal()
    self.normalize()
    self.train_loader = DataLoader(self.train, batch_size=16, shuffle=True)
    self.test_loader = DataLoader(self.train, batch_size=256, shuffle=True)

    # labels = np.log10(labels)

  def compute_normal(self):
    columns_to_normalize = ['GASTW', 'GAREA'] + useful_weather
    self.norm_factors = {}
    for col in columns_to_normalize:
      mini, maxi = self.train.get_min_max(col)
      self.norm_factors[col] = (mini, maxi - mini)

  def normalize(self):
    self.norm_factors['labels'] = (0, 100_000)
    functions = {col: lambda x: (x - subi) / divi for col, (subi, divi) in self.norm_factors.items()}
    functions['labels'] = lambda x: np.log10(x + 1) # if x != 0 else 0
    self.train.apply_to_each(functions)
    self.test.apply_to_each(functions)

  def denormalize(self, col, x):
    if col == 'labels':
      return np.power(10, x) - 1 # if x == 0 else np.exp(x)
    if col in self.norm_factors:
      subi, divi = self.norm_factors[col]
      return x * divi + subi
    return x

  def evaluate(self, model, nb_sample=10, train=True):
    rel_diff = 0
    c = 0
    for feat, target in (self.train_loader if train else self.test_loader):
      expected = self.denormalize('labels', target.detach().numpy())
      predictions = self.denormalize('labels', model(feat).detach().numpy())[expected != 0]
      expected = expected[expected != 0]
      diff = predictions - expected
      rel_diff += (np.abs(diff) / expected).mean()
      c += 1
      if c == nb_sample:
        return rel_diff / nb_sample


class Ann:
  def __init__(self, data: HourlyPreprocessor, n_features, n_output, n_hidden):
    self.model = nn.Sequential(
      nn.Linear(n_features, n_hidden),
      nn.LeakyReLU(),
      nn.Linear(n_hidden, n_hidden),
      nn.LeakyReLU(),
      nn.Linear(n_hidden, n_output),
      nn.LeakyReLU()
    )
    self.data = data
    self.criterionH = nn.MSELoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=0.000005, betas=(0.9, 0.999), eps=1e-8)
    # self.optimizer = optim.SGD(self.model.parameters(), lr=0.010)
    # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=1000, cooldown=5000, verbose=True)

  def do_epoch(self, evaluate=True):
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

      # print('.', end='', flush=True)
      i += 1
      t = self.data.denormalize("labels", target.detach().numpy())
      o = self.data.denormalize("labels", out.detach().numpy())
      o2 = o[t != 0]
      t2 = t[t != 0]
      losses += np.abs((t2 - o2) / t2).mean()
      mses += loss.data.item()
      t3 = target.detach().numpy()
      o3 = out.detach().numpy()[t3 != 0]
      t3 = t3[t3 != 0]
      abv += len(np.where(o3 > t3)[0]) / float(len(t3))

      if i % 100 == 0:
        print(losses / 100, '\t', mses / 100, '\t', abv / 100)
        # print(np.where(o3 > t3))
        # print(t.flatten().tolist())
        # print(o.flatten().tolist())
        losses = 0
        mses = 0
        abv = 0
        # return
        # print(self.data.evaluate(self.model))

    # if evaluate:
    #   train_predictions = self.model(self.data.train.features_t).detach().numpy()
    #   test_predictions = self.model(self.data.test.features_t).detach().numpy()
    #   avg_diff, nb_above, loss = data.evaluate(self.data.train, train_predictions)
    #   avg_diff_te, nb_above_te, loss_te = data.evaluate(self.data.test, test_predictions)
    #   self.scheduler.step(avg_diff)
      # return avg_diff, nb_above, loss, avg_diff_te, nb_above_te, loss_te
    # else:
    #   return None, None, None, None, None, None

  def get_lr(self):
    return self.optimizer.param_groups[0]['lr']


def summer(df):
  start_summer = pd.Timestamp(year=2017, month=3, day=15)
  end_summer = pd.Timestamp(year=2017, month=11, day=15)
  df = df[start_summer < df.index.map(pd.Timestamp)]
  return df[df.index.map(pd.Timestamp) < end_summer].index


if __name__ == '__main__':
  useful_weather = ['G_Dh_mean', 'G_Bn_mean', 'G_Dh_var', 'G_Bn_var', 'Ta_mean', 'Ta_var', 'Ta_min',
                    'FF_mean', 'FF_var', 'h_day']
  buildings = pd.read_csv('data/sanitized_complete.csv')\
    .set_index('EGID')\
    .drop(columns=['heating', 'cooling'])
  forecast = pd.read_csv('data/daily_forecast.csv').set_index('timestamp')[useful_weather]
  summer = summer(forecast)

  forecast.drop(summer, inplace=True)
  print(forecast)
  predictions = pd.read_csv('data/daily_predictions.csv').set_index('timestamp').drop(summer)
  preprocessor = HourlyPreprocessor(
    dataset=HourlyDataset(
      building_features=buildings,
      weather_forecast=forecast,
      labels=predictions
    ),
    split=0.80
  )
  print('loaded')
  print(preprocessor.train.nb_features)
  ann = Ann(preprocessor,
            n_features=preprocessor.train.nb_features,
            n_output=1,
            n_hidden=20)

  losses = []
  relative_diff = []
  for j in range(100):
    ann.do_epoch(j % 100 == 0)
    print(j, preprocessor.evaluate(ann.model, train=False))

  # load_forecast()
  # print(pd.Timestamp(year=2017, month=1, day=1, hour=0) + pd.Timedelta(hours=1))
  # filter_columns()
  # filter_forecast()
