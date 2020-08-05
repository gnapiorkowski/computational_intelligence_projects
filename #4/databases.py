import pandas as pd
import numpy as np

df = pd.read_csv('iris_with_errors.csv', na_values=['n/a', '-','--', 'nan'])

print('-'*40)
print("Błędy w kolumnach: \n", df.isna().sum())
print('Błędów w sumie: ', df.isna().sum().sum())
print('Mean: ' df.describe().iloc[2,:])
print('-'*40)
print(df.dtypes)

for column in df:
    if df.dtypes[column] == 'float64':
        print(df.index[df[column].between(0,15)].tolist())
