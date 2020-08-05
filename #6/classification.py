import pandas as pd
import sklearn as sk
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import naive_bayes
from sklearn.metrics import confusion_matrix

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

print('Najwięcej fałszywych negatywów miał: ', max(FalseNegative, key=FalseNegative.get))
print('Najmniej fałszywych negatywów miał: ', min(FalseNegative, key=FalseNegative.get))

plt.bar(['knn3', 'knn5', 'knn11', 'Drzewo', 'N. bayesowski'], [knn3Score, knn5Score, knn11Score, dtcScore, gnbScore])
plt.show()