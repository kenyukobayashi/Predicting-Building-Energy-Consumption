import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
from torch import optim
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class HourlyDataset(Dataset):
  def __init__(self, df, forecast):
    super(HourlyDataset, self).__init__()
    self.n = df.shape[0]
    self.features = df.drop(columns=['heating', 'cooling'])
    self.labels = df[['heating']]
    self.features_t = HourlyDataset.df_to_tensor(self.features)
    self.labels_t = HourlyDataset.df_to_tensor(self.labels)

  def __len__(self):
    return self.n

  def __getitem__(self, index):
    return self.features_t[index], self.labels_t[index]

  @staticmethod
  def df_to_tensor(df):
    return torch.from_numpy(df.values).float()


def filter_columns():
  prediction = pd.read_csv('data/jonctionnofloors_TH.tsv', sep='\t')
  data_top = prediction.columns.tolist()
  heatings_col = [s for s in data_top if 'Heating' in s]
  prediction = prediction[heatings_col]
  prediction['timestep'] = prediction.index
  prediction = prediction.reset_index()
  prediction.to_csv('data/hourly_predictions.csv', index=False)


def flatten():
  prediction = pd.read_csv('data/hourly_predictions.csv')
  result = pd.DataFrame(columns=['timestep', 'EGID', 'heating'])

  time_col = 'timestep'
  for col in prediction.columns.tolist():
    if 'Heating' in col:
      df = prediction[[time_col, col]].copy()
      egid = int(col[col.find('(') + 1: col.find(')')])
      df['EGID'] = egid
      df.rename(columns={col: 'heating'}, inplace=True)
      result = result.append(df, ignore_index=True, sort=False)
  result.to_csv('data/hourly_predictions_flatten.csv', index=False)


if __name__ == '__main__':
  # filter_columns()
  filter_columns()
  flatten()
