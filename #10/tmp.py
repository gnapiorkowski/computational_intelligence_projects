import pandas as pd

df = pd.read_csv('dogsVScats.csv')
df.loc[(df['label'] >= 0.5) & df['id'].str.contains('cat'), 'true'] = 0
df.loc[(df['label'] < 0.5) & df['id'].str.contains('cat'), 'true'] = 1
df.loc[(df['label'] < 0.5) & df['id'].str.contains('dog'), 'true'] = 1
df.loc[(df['label'] >= 0.5) & df['id'].str.contains('dog'), 'true'] = 0
print('Accuracy: ', 1. - len(df.loc[df['true'] == 0].index)/len(df.index))
print(df.loc[df['true'] == 0])