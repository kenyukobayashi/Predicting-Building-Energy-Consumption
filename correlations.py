import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the datasets
building_features = pd.read_csv("data/sanitized_complete.csv")

# Dropping the index and the labels
building_features = building_features[['GASTW','GAREA','habit','b19','19-45','46-60','61-70','71-80','81-90','91-2000','a2000']]
weather_features = pd.read_csv('data/Geneva.cli', sep='\t', header=3)

# Creating the figure instance and the two subplots
fig = plt.figure(figsize = (40,40))
ax1 = fig.add_subplot(1, 2, 1) # row, column, position
ax2 = fig.add_subplot(1, 2, 2)

# Using Pearson Correlation
building_cor = building_features.corr()
weather_cor = weather_features.corr()

# We use ax parameter to tell seaborn which subplot to use for this plot
sns.heatmap(data=building_cor, ax=ax1, cmap=plt.cm.Reds, square=True, cbar_kws={"shrink": 0.6}, annot=True, annot_kws={'fontsize': 4})
sns.heatmap(data=weather_cor, ax=ax2, cmap=plt.cm.Reds, square=True, cbar_kws={"shrink": 0.6}, annot=True, annot_kws={'fontsize': 4})

# Adjusting the heatmap sides
bottom, top = ax1.get_ylim()
ax1.set_ylim(bottom + 0.5, top - 0.5)
bottom, top = ax2.get_ylim()
ax2.set_ylim(bottom + 0.5, top - 0.5)

# Rotating the axis labels
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=5)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=45, horizontalalignment='right', fontsize=5)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=5)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=45, horizontalalignment='right', fontsize=5)

# Adding titles
ax1.set_title('Building features correlations', fontsize=10)
ax2.set_title('Weather features correlations', fontsize=10)

# Displaying the plots
plt.show()
