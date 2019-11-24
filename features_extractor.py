import pandas as pd

gbaup = {'b19': [8011], '19-45': [8012], '46-60': [8013], '61-70': [8014], '71-80': [8015],
         '81-90': [8016, 8017], '91-2000': [8018, 8019], 'a2000': [8020, 8021, 8022]}
gklas = {'habit': [1122, 1130, 1121, 1110, 1211, 1212], 'office': [1220, 1230, 1263, 1264],
         'industrial': [1251, 1252], 'hobbies': [1272, 1261, 1265]}

if __name__ == '__main__':
  sanitized = pd.read_csv('data/sanitized.csv', delimiter=',')
  sanitized.rename(columns={'heatingNeeds(Wh)': 'heating', 'coolingNeeds(Wh)': 'cooling'}, inplace=True)

  for label, values in gbaup.items():
    sanitized[label] = sanitized['GBAUP'].apply(lambda s: 1 if s in values else 0)

  useful_class = []
  for label, values in gklas.items():
    useful_class += values
    sanitized[label] = sanitized['GKLAS'].apply(lambda s: 1 if s in values else 0)
  sanitized['others'] = sanitized['GKLAS'].apply(lambda s: 0 if s in useful_class else 1)

  sanitized.drop(columns=['GKLAS', 'GBAUP'], inplace=True)
  sanitized.to_csv('data/features.csv', index=False)
