import pandas as pd
import numpy as np


def filter_columns():
  # prediction = pd.read_csv('data/hourly_predictions.csv')
  prediction = pd.read_csv('data/jonctionnofloors_TH.tsv', sep='\t')
  data_top = prediction.columns.tolist()
  heatings_col = [s for s in data_top if 'Heating' in s]
  prediction = prediction[heatings_col]
  new_columns = {str(col): int(col[col.find('(') + 1: col.find(')')]) for col in heatings_col}
  prediction['timestamp'] = prediction.index.map(
    lambda t: pd.Timestamp(year=2017, month=1, day=1) + pd.Timedelta(hours=t))
  prediction.set_index('timestamp', inplace=True)
  prediction.rename(columns=new_columns).to_csv('data/hourly_predictions.csv')


def pass_to_daily():
  df = pd.read_csv('data/hourly_predictions.csv')
  df2 = pd.read_csv(r'data/weather_forecast.csv')

  df = df.groupby(np.arange(len(df)) // 24).sum()
  df['timestamp'] = df.index.map(lambda t: pd.Timestamp(year=2017, month=1, day=1) + pd.Timedelta(days=t))
  df.set_index('timestamp').to_csv('data/daily_predictions.csv')

  df2 = df2.drop(['h', 'timestamp'], axis=1)
  header = df2.columns.tolist()
  df2['h_day'] = df2['G_Dh'].groupby(np.arange(len(df2)) // 24).apply(np.count_nonzero)
  for head in header:
    df2[head + '_min'] = df2[head].groupby(np.arange(len(df2)) // 24).min()
    df2[head + '_mean'] = df2[head].groupby(np.arange(len(df2)) // 24).mean()
    df2[head + '_var'] = df2[head].groupby(np.arange(len(df2)) // 24).std()
    df2 = df2.drop(head, axis=1)
  df2 = df2.dropna()
  df2['timestamp'] = df2.index.map(lambda t: pd.Timestamp(year=2017, month=1, day=1) + pd.Timedelta(days=t))

  df2.set_index('timestamp').to_csv('data/daily_forecast.csv')


def filter_forecast():
  df = pd.read_csv('data/Geneva.cli', sep='\t', header=3)

  def row_to_timestamp(row):
    pd.Timestamp(
      year=2017,
      month=int(row['m']),
      day=int(row['dm']),
      hour=int(row['h']) - 1  # needed because hours go from 1 to 24, instead of 0-23
    )

  df['timestamp'] = df.apply(row_to_timestamp, axis=1)
  df = df[['timestamp', 'h', 'G_Dh', 'G_Bn', 'Ta', 'FF', 'DD']]
  df.set_index('timestamp', inplace=True)
  df.to_csv('data/weather_forecast2.csv')


if __name__ == '__main__':
  pass_to_daily()
