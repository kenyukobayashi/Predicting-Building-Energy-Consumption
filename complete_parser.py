import pandas as pd

HEADERS = 'EGID,GBAUP,GASTW,GKAT,AREA'.split(',')
COL_TO_KEEP = ['EGID', 'heatingNeeds(Wh)', 'coolingNeeds(Wh)', 'GBAUP', 'GASTW', 'GAREA', 'GKLAS']

gkat_habit = [1010, 1021, 1025, 1030, 1040]
gbaup = {'b19': [8011], '19-45': [8012], '46-60': [8013], '61-70': [8014], '71-80': [8015],
         '81-90': [8016, 8017], '91-2000': [8018, 8019], 'a2000': [8020, 8021, 8022]}

if __name__ == '__main__':
  output = pd.read_csv('data/jonctionnofloors_YearlyResultsPerBuilding.out', delimiter='\t')
  output['EGID'] = output['#buildingId(key)'].apply(lambda s: int(s[s.find('(') + 1: s.find(')')]))
  output.drop(columns='#buildingId(key)', inplace=True)
  output = output.set_index('EGID')

  info = pd.read_csv('data/complete_RegBL.csv', delimiter=',')
  info.drop_duplicates(subset='EGID', inplace=True)
  info.set_index('EGID', inplace=True)
  joined = output.join(info, on='EGID', lsuffix='_', how='left')

  joined.rename(columns={'heatingNeeds(Wh)': 'heating', 'coolingNeeds(Wh)': 'cooling', 'AREA': 'GAREA'}, inplace=True)
  joined['habit'] = joined['GKAT'].apply(lambda s: 1 if s in gkat_habit else 0)

  for label, values in gbaup.items():
    joined[label] = joined['GBAUP'].apply(lambda s: 1 if s in values else 0)

  joined.drop(columns=['GKAT', 'GBAUP'], inplace=True)
  joined.to_csv('data/sanitized_complete.csv')

  print(joined.shape)
  print(joined.dropna().shape)
