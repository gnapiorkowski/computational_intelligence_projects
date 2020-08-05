import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

iris = pd.read_csv('iris.csv')
iris.loc[iris['class']=='Iris-virginica','virginica']=1
iris.loc[iris['class']=='Iris-versicolor','versicolor']=1
iris.loc[iris['class']=='Iris-setosa','setosa'] = 1
iris.loc[iris['class']!='Iris-virginica','virginica']=0
iris.loc[iris['class']!='Iris-versicolor','versicolor']=0
iris.loc[iris['class']!='Iris-setosa','setosa'] = 0

tmpSetosa = iris['setosa']
tmpVirginica = iris['virginica']
tmpVersicolor = iris['versicolor']
iris = iris.drop(['class'], axis=1)
x = iris.values
minMaxScaler = preprocessing.MinMaxScaler()
x_scaled = preprocessing.MinMaxScaler().fit_transform(x)
iris = pd.DataFrame(x_scaled)
iris['setosa'] = tmpSetosa
iris['virginica'] = tmpVirginica
iris['versicolor'] = tmpVersicolor
allInputs = iris[[0, 1, 2, 3]].values
allClasses = iris[['virginica', 'versicolor', 'setosa']].values
(trainInputs, testInputs, trainClasses, testClasses) = train_test_split(allInputs, allClasses, train_size=0.7, random_state=1)

model = Sequential()
model.add(Dense(32, input_dim=4, activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(trainInputs, trainClasses, epochs=100, batch_size=64)

y_pred = model.predict(testInputs)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
test = list()
for i in range(len(testClasses)):
    test.append(np.argmax(testClasses[i]))
a = accuracy_score(pred,test)
print(confusion_matrix(pred, test))
print('Accuracy:', a*100, '%')