import pandas as pd
import numpy as np

df = pd.read_csv('iris_with_errors.csv', na_values=['n/a', '-','--', 'nan'])

print('-'*40)
print("Błędy w kolumnach: \n", df.isna().sum())
print('Błędów w sumie: ', df.isna().sum().sum())
print('Mean: ', df.describe().iloc[2,:])
print('-'*40)
print(df.dtypes)

print(df[df['sepal.width'] < 0])

for column in df:
    if df.dtypes[column] == 'float64':
        poprawneDane = df.index[df[column].between(0,15)].tolist()
        #print ('Wrong: ', df[df[column] < 0.0],'\n', 'Wrong2: ', df[df[column] > 15.0])
        #df[df[column] < 0.0][column] = df.describe().iloc[2,:][column]
        #df[df[column] > 15.0] = df.describe().iloc[2,:][column]
        #print ('Right: ', df[df[column] < 0.0], '\n', 'Right2: ', df[df[column] > 15.0])
        df[column] = df[column].mask(df[column] < 0, df.describe().iloc[2,:][column])
        df[column] = df[column].mask(df[column] > 15, df.describe().iloc[2,:][column])
variety = df['variety']
etosa = variety.str.contains('etosa')
irginica = variety.str.contains('irginica')
ersicolor = variety.str.contains('ersicolor') 

df['variety'] = np.where(etosa, 'Setosa',
                        np.where(ersicolor, 'Versicolor',
                                np.where(irginica, 'Virginica', None)))

print(df.loc[[47]])