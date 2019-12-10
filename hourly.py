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
    self.features = building_features.dropna()
    self.forecast = weather_forecast

    non_nan_indices = [str(egid) for egid in self.features.index.tolist()]
    self.labels = labels[non_nan_indices]

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


def filter_columns():
  # prediction = pd.read_csv('data/hourly_predictions.csv')
  prediction = pd.read_csv('data/jonctionnofloors_TH.tsv', sep='\t')
  data_top = prediction.columns.tolist()
  heatings_col = [s for s in data_top if 'Heating' in s]
  prediction = prediction[heatings_col]
  new_columns = {str(col): int(col[col.find('(') + 1: col.find(')')]) for col in heatings_col}

  prediction['timestamp'] = prediction.index.map(lambda t: pd.Timestamp(year=2017, month=1, day=1, hour=0) + pd.Timedelta(hours=t))
  prediction.set_index('timestamp', inplace=True)
  prediction.rename(columns=new_columns).to_csv('data/hourly_predictions.csv')


def filter_forecast():
  df = pd.read_csv('data/Geneva.cli', sep='\t', header=3)
  row_to_timestamp = lambda row: pd.Timestamp(
    year=2017,
    month=int(row['m']),
    day=int(row['dm']),
    hour=int(row['h']) - 1  # needed because hours go from 1 to 24, instead of 0-23
  )
  df['timestamp'] = df.apply(row_to_timestamp, axis=1)
  df = df[['timestamp', 'h', 'G_Dh', 'G_Bn', 'Ta', 'FF', 'DD']]
  df.set_index('timestamp', inplace=True)
  df.to_csv('data/weather_forecast.csv')


if __name__ == '__main__':
  buildings = pd.read_csv('data/sanitized_complete.csv')\
    .set_index('EGID')\
    .drop(columns=['heating', 'cooling'])
  forecast = pd.read_csv('data/weather_forecast.csv').set_index('timestamp')
  predictions = pd.read_csv('data/hourly_predictions.csv').set_index('timestamp')
  ds = HourlyDataset(
    building_features=buildings,
    weather_forecast=forecast,
    labels=predictions
  )

  print(ds[1])
  # load_forecast()
  # print(pd.Timestamp(year=2017, month=1, day=1, hour=0) + pd.Timedelta(hours=1))
  # filter_columns()
  # filter_forecast()
