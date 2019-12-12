import pandas as pd


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
  df.to_csv('data/weather_forecast.csv')
