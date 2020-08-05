import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import naive_bayes



df = pd.read_csv('diabetes.csv')
df.loc[df['class']=='tested_positive','tested_positive'] = 1
df.loc[df['class']=='tested_positive','tested_negative'] = 0
df.loc[df['class']=='tested_negative','tested_negative'] = 1
df.loc[df['class']=='tested_negative','tested_positive'] = 0
tmppos = df['tested_positive']
tmpneg = df['tested_negative']
df = df.drop(['class'], axis=1)
df = df.drop(['tested_positive'], axis=1)
df = df.drop(['tested_negative'], axis=1)
x = df.values
minMaxScaler = preprocessing.MinMaxScaler()
x_scaled = preprocessing.MinMaxScaler().fit_transform(x)
df = pd.DataFrame(x_scaled)
df['tested_positive'] = tmppos 
df['tested_negative'] = tmpneg

allInputs = df[[0, 1, 2, 3, 4, 5, 6, 7]].values
allClasses = df[['tested_negative', 'tested_positive']].values
(trainInputs, testInputs, trainClasses, testClasses) = train_test_split(allInputs, allClasses, train_size=0.7, random_state=1)

model = Sequential()
model.add(Dense(128, input_dim=8, activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(2, activation='softmax'))
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
print('neural network Accuracy:', a*100, '%')


df = pd.read_csv('diabetes.csv')

all_inputs = df[['pregnant-times', 'glucose-concentr', 'blood-pressure', 'skin-thickness', 'insulin', 'mass-index', 'pedigree-func', 'age']].values
all_classes = df['class'].values
(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.67, random_state=1)

FalseNegative = {}

from sklearn.metrics import classification_report
knn3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn3.fit(train_inputs, train_classes)
knn3Score = knn3.score(test_inputs, test_classes)
print('KNN 3 score: ', knn3Score)
y_pred = knn3.predict(test_inputs)
print(confusion_matrix(test_classes, y_pred))
FalseNegative['knn3'] = confusion_matrix(test_classes, y_pred)[0][1]

knn5 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn5.fit(train_inputs, train_classes)
knn5Score = knn5.score(test_inputs, test_classes)
print('KNN 5 score: ', knn5Score)
y_pred = knn5.predict(test_inputs)
print(confusion_matrix(test_classes, y_pred))
FalseNegative['knn5'] = confusion_matrix(test_classes, y_pred)[0][1]

knn11 = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
knn11.fit(train_inputs, train_classes)
knn11Score = knn11.score(test_inputs, test_classes)
print('KNN 11 score: ', knn11Score)
y_pred = knn11.predict(test_inputs)
print(confusion_matrix(test_classes, y_pred))
FalseNegative['knn11'] = confusion_matrix(test_classes, y_pred)[0][1]

dtc = tree.DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)
dtcScore = dtc.score(test_inputs, test_classes)
print('Score of decision tree: ', dtcScore)
y_pred = dtc.predict(test_inputs)
print(confusion_matrix(test_classes, y_pred))
FalseNegative['Drzewo decyzyjne'] = confusion_matrix(test_classes, y_pred)[0][1]

gnb = naive_bayes.GaussianNB()
gnb.fit(train_inputs, train_classes)
gnbScore = gnb.score(test_inputs, test_classes)
print('GNB score: ', gnbScore)
y_pred = gnb.predict(test_inputs)
print(confusion_matrix(test_classes, y_pred))
FalseNegative['Naiwny bayesowski'] = confusion_matrix(test_classes, y_pred)[0][1]