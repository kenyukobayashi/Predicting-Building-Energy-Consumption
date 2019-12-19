"""
This file contains methods that were used to produce the files in data/
based on other larger files that were provided but are not included in the submission
"""
import pandas as pd
import numpy as np

"""Annual parsing"""

def find_building_features_and_labels():
  """
  Takes one file that contains Swiss building information
  and one file that contains simulation of heating and cooling needs
  And assemble them to have a file with only simulated buildings and their relevant features
  """
  HEADERS = 'EGID,GBAUP,GASTW,GKAT,AREA'.split(',')
  COL_TO_KEEP = ['EGID', 'heatingNeeds(Wh)', 'coolingNeeds(Wh)', 'GBAUP', 'GASTW', 'GAREA', 'GKLAS']

  # GKAT number that represents categories
  gkat_habit = [1010, 1021, 1025, 1030, 1040]
  # Mapping of construction periods to group
  gbaup = {'b19': [8011], '19-45': [8012], '46-60': [8013], '61-70': [8014], '71-80': [8015],
           '81-90': [8016, 8017], '91-2000': [8018, 8019], 'a2000': [8020, 8021, 8022]}

  # Gets the output of the simulation
  output = pd.read_csv('data/jonctionnofloors_YearlyResultsPerBuilding.out', delimiter='\t')
  output['EGID'] = output['#buildingId(key)'].apply(lambda s: int(s[s.find('(') + 1: s.find(')')]))
  output.drop(columns='#buildingId(key)', inplace=True)
  output = output.set_index('EGID')

  # Gets the building information
  info = pd.read_csv('data/complete_RegBL.csv', delimiter=',')
  info.drop_duplicates(subset='EGID', inplace=True)
  info.set_index('EGID', inplace=True)

  # Join the datasets
  joined = output.join(info, on='EGID', lsuffix='_', how='left')
  joined.rename(columns={'heatingNeeds(Wh)': 'heating', 'coolingNeeds(Wh)': 'cooling', 'AREA': 'GAREA'}, inplace=True)

  # Create binary features
  joined['habit'] = joined['GKAT'].apply(lambda s: 1 if s in gkat_habit else 0)
  for label, values in gbaup.items():
    joined[label] = joined['GBAUP'].apply(lambda s: 1 if s in values else 0)

  # Drop irrelevant features and write to file
  joined.drop(columns=['GKAT', 'GBAUP'], inplace=True)
  joined.to_csv('data/sanitized_complete.csv')


"""Daily parsing"""

def filter_columns():
  """
  Filter output of the simulation to keep only heating prediction
  The simulation contained a lot of other predictions that were not useful
  """
  prediction = pd.read_csv('../data/jonctionnofloors_TH.tsv', sep='\t')
  data_top = prediction.columns.tolist()
  heatings_col = [s for s in data_top if 'Heating' in s]
  prediction = prediction[heatings_col]
  new_columns = {str(col): int(col[col.find('(') + 1: col.find(')')]) for col in heatings_col}
  prediction['timestamp'] = prediction.index.map(
    lambda t: pd.Timestamp(year=2017, month=1, day=1) + pd.Timedelta(hours=t))
  prediction.set_index('timestamp', inplace=True)
  prediction.rename(columns=new_columns).to_csv('data/hourly_predictions.csv')


def pass_to_daily():
  """Convert hourly prediction and weather information to daily"""
  df = pd.read_csv('../data/hourly_predictions.csv')
  df2 = pd.read_csv(r'../data/weather_forecast.csv')

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
  """Filter the weather information file to keep only relevant features"""
  df = pd.read_csv('../data/Geneva.cli', sep='\t', header=3)

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
