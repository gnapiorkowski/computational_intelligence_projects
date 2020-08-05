import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from matplotlib import pyplot as plt

df = pd.read_csv('titanic.csv')

items = set((df['Class'].unique()))
for i in df['Sex']:
    items.add(i)
for i in df['Age']:
    items.add(i)
for i in df['Survived']:
    items.add(i)
print(items)
encodedVals = []
for index, row in df.iterrows():
    labels = {}
    uncommons = list(items - set(row))
    commons = list(items.intersection(row))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encodedVals.append(labels)
ohe_df = pd.DataFrame(encodedVals)
print(df)
print(ohe_df)
freq_items = apriori(ohe_df, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.005)
print('-'*40, '\nRules:\n', rules.loc[rules['confidence'] >= 0.8])
plt.subplot(2, 2, 1)
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.subplot(2, 2, 3)
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.subplot(2, 2, 2)
fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], fit_fn(rules['lift']))
plt.xlabel('lift')
plt.ylabel('confidence')
plt.show()