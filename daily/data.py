import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DailyCrossValidation:
  def __init__(self, df, k):
    np.random.seed(4)
    indices = np.random.permutation(df.nb_buildings)
    split_step = int(df.nb_buildings / k)
    self.indices = np.array(np.split(indices[:split_step * k], indices_or_sections=k))
    self.df = df
    self.k = k

  def __iter__(self):
    for ind in range(len(self.indices)):
      tr_ind = np.delete(self.indices, ind, axis=0).flatten()
      te_ind = self.indices[ind].flatten()
      yield DailyPreprocessor(self.df, split=(tr_ind, te_ind))


class DailyDataset(Dataset):
  def __init__(self, building_features, weather_forecast, labels):
    super(DailyDataset, self).__init__()
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
    return DailyDataset(self.features.iloc[tr_egid], self.forecast, self.labels), \
           DailyDataset(self.features.iloc[te_egid], self.forecast, self.labels)

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

  def to_numpy(self):
    return np.array([x.numpy() for x, y in self]), np.array([y.numpy() for x, y in self]).ravel()


class DailyPreprocessor:
  def __init__(self, dataset: DailyDataset, split=0.75):
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
    self.test_loader = DataLoader(self.test, batch_size=256, shuffle=True)

  def compute_normal(self):
    columns_to_normalize = ['GASTW', 'GAREA'] + ['G_Dh_mean', 'G_Bn_mean', 'G_Dh_var', 'G_Bn_var', 'Ta_mean', 'Ta_var', 'Ta_min',
                    'FF_mean', 'FF_var', 'h_day']
    self.norm_factors = {}
    for col in columns_to_normalize:
      mini, maxi = self.train.get_min_max(col)
      self.norm_factors[col] = (mini, maxi - mini)

  def normalize(self):
    self.norm_factors['labels'] = (0, 100_000)
    functions = {col: lambda x: (x - subi) / divi for col, (subi, divi) in self.norm_factors.items()}
    functions['labels'] = lambda x: np.log10(x + 1) / 10.0  # if x != 0 else 0
    self.train.apply_to_each(functions)
    self.test.apply_to_each(functions)

  def denormalize(self, col, x):
    if col == 'labels':
      return np.power(10, x * 10.0) - 1  # if x == 0 else np.exp(x)
    if col in self.norm_factors:
      subi, divi = self.norm_factors[col]
      return x * divi + subi
    return x

  def evaluate(self, model, train=True):
    rel_diff = 0
    c = 0
    for feat, target in (self.train_loader if train else self.test_loader):
      expected = self.denormalize('labels', target.detach().numpy())
      predictions = self.denormalize('labels', model(feat).detach().numpy())[expected != 0]
      expected = expected[expected != 0]
      rel_diff += np.abs(np.log(predictions/expected)).sum()
      c += len(expected)
    return rel_diff / c
