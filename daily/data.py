import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class DailyDataset(Dataset):
  """Dataset holding all the data for daily prediction"""
  def __init__(self, building_features: pd.DataFrame, weather_forecast: pd.DataFrame, labels: pd.DataFrame):
    """
    Initialize the dataset
    :param building_features: features for each building (such as GAREA, period of construction, ...)
    :param weather_forecast: weather information aggregated for each day (such as minimum and mean temperature, irradiation, ...)
    :param labels: targets for each building for each day
    """
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
    """
    Construct all the feature for a given day and building,
    by combining the building's features and the weather information of this day.
    """
    building, hour = divmod(index, self.h_in_years)
    egid = self.labels.columns[building]
    timestamp = self.labels.index[hour]

    forecast_tensor = torch.tensor(self.forecast.loc[timestamp])
    building_tensor = torch.tensor(self.features.loc[int(egid)])
    return torch.cat((building_tensor, forecast_tensor), 0), torch.tensor([self.labels.loc[timestamp, egid]])

  def split_buildings(self, tr_egid, te_egid):
    """Split data set over buildings, based on EGID to put in train (`tr_egid`) and in test (`te_egid`)"""
    return DailyDataset(self.features.iloc[tr_egid], self.forecast, self.labels), \
           DailyDataset(self.features.iloc[te_egid], self.forecast, self.labels)

  def apply_to_each(self, functions):
    """Apply the function corresponding to the columns in the `functions` dict"""
    for col, f in functions.items():
      for df in [self.features, self.forecast]:
        if col in df.columns:
          df[col] = df[col].apply(f)
    if 'labels' in functions:
      self.labels = self.labels.apply(functions['labels'])

  def get_min_max(self, col):
    """
    :returns: min and max of the columns `col`
    """
    for df in [self.features, self.forecast]:
      if col in df.columns:
        return df[col].min(), df[col].max()

  def to_numpy(self):
    """Converts the Dataset to a numpy array"""
    return np.array([x.numpy() for x, y in self]), np.array([y.numpy() for x, y in self]).ravel()


class DailyPreprocessor:
  """Preprocess and gives access to train and test data"""

  def __init__(self, dataset: DailyDataset, split=0.75):
    """
    Initialize this preprocessor
    :param dataset: DailyDataset that contains all the datapoints
    :param split:
                  If it is a float, it randomly split the dataset between training and testing
                  where split is the proportion (between 0 and 1) of data in training
                  If it is a pair of arrays of EGIDs, it assigns the EGIDs of the first array to the training set,
                  and the one from the second array to testing set
    """
    # Split the data
    if isinstance(split, float):
      egids = dataset.features.index.tolist()
      np.random.shuffle(egids)
      split = int(len(egids) * split)
      tr = egids[:split]
      te = egids[split:]
    else:
      tr, te = split
    self.train, self.test = dataset.split_buildings(tr, te)

    # Compute the normalization factors and normalize
    self.norm_factors = self.compute_normal(self.train)
    self.normalize()
    # Define pytorch DataLoader
    self.train_loader = DataLoader(self.train, batch_size=16, shuffle=True)
    self.test_loader = DataLoader(self.test, batch_size=256, shuffle=True)

  @staticmethod
  def compute_normal(train):
    """Compute normalization factors for columns that should be normalized based on training set"""
    columns_to_normalize = ['GASTW', 'GAREA'] + ['G_Dh_mean', 'G_Bn_mean', 'G_Dh_var', 'G_Bn_var', 'Ta_mean', 'Ta_var',
                                                 'Ta_min', 'FF_mean', 'FF_var', 'h_day']
    norm_factors = {}
    for col in columns_to_normalize:
      mini, maxi = train.get_min_max(col)
      norm_factors[col] = (mini, maxi - mini)
    return norm_factors

  def normalize(self):
    """Normalize columns, and transform targets to their log version (divided by 10 for faster convergence)"""
    functions = {col: lambda x: (x - subi) / divi for col, (subi, divi) in self.norm_factors.items()}
    functions['labels'] = lambda x: np.log10(x + 1) / 10.0
    self.train.apply_to_each(functions)
    self.test.apply_to_each(functions)

  def denormalize(self, col, x):
    """Denormalize `col` (which could be a numpy array, pandas dataframe, or number) based on what it is"""
    if col == 'labels':
      return np.power(10, x * 10.0) - 1
    if col in self.norm_factors:
      subi, divi = self.norm_factors[col]
      return x * divi + subi
    return x

  def evaluate(self, model, train=True):
    """
    Evaluate a model on testing or training set (based on `train`)
    :returns: average Ln Q error
    """
    ln_q = 0
    c = 0
    for feat, target in (self.train_loader if train else self.test_loader):
      expected = self.denormalize('labels', target.detach().numpy())
      predictions = self.denormalize('labels', model(feat).detach().numpy())[expected != 0]
      expected = expected[expected != 0]
      ln_q += np.abs(np.log(predictions/expected)).sum()
      c += len(expected)
    return ln_q / c


class DailyCrossValidation:
  """Cross validation for daily prediction"""
  def __init__(self, data: DailyDataset, k: int):
    """Initialize a cross validation with `k` folds"""
    np.random.seed(4)
    indices = np.random.permutation(data.nb_buildings)
    split_step = int(data.nb_buildings / k)
    self.indices = np.array(np.split(indices[:split_step * k], indices_or_sections=k))
    self.df = data

  def __iter__(self):
    """Iterate on each fold of the cross validation"""
    for ind in range(len(self.indices)):
      tr_ind = np.delete(self.indices, ind, axis=0).flatten()
      te_ind = self.indices[ind].flatten()
      yield DailyPreprocessor(self.df, split=(tr_ind, te_ind))
