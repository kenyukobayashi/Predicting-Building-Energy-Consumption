import numpy as np
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

from preprocessor import DataPreProcessor



if __name__ == '__main__':
  features_labels = ['EGID', 'heatingNeeds(Wh)', 'coolingNeeds(Wh)', 'GBAUP', 'GASTW', 'GAREA', 'GKLAS']

  dataset = pd.read_csv('data/features.csv').set_index('EGID')
  dataset.dropna(inplace=True)
  data = DataPreProcessor(dataset)
  x_test = data.test.features.to_numpy()
  x_train = data.train.features.to_numpy()
  y_train = data.train.labels.to_numpy().ravel()

  clf = RandomForestRegressor(n_estimators=10000,random_state=1,n_jobs=-1)
  clf.fit(x_train,y_train)

  for feature in zip(features_labels, clf.feature_importances_):
      print(feature)
