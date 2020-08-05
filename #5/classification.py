import pandas as pd
import random

df = pd.read_csv('iris.csv')

def myPredictRow(sl, sw, pl, pw):
    if(pw < 1):
        return 'Iris-setosa'
    elif(pl < 5 and pw < 1.8):
        return 'Iris-versicolor'
    else:
        return 'Iris-virginica'
good = 0
for row in df.itertuples():
    if row[5] == myPredictRow(row[1], row[2], row[3], row[4]): good += 1

print('My accuracy: ', good/150, '%\n')
