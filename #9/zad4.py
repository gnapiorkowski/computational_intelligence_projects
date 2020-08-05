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
from math import pow
import random

PRZEDZIAL = 5

def f(x):
    return pow(x, 3)+3*pow(x, 2)-5*x+7

def fRes(B):
    x, y = [], []
    for i in range(-100, 100):
        if (i == 0):
            x.append(0)
            y.append(f(0)/200)
        else:
            x.append(i*B/100/200)
            y.append(f(i*B/100)/200)
    return [x, y]

def rSeq(length):
    seq = []
    for i in range(length):
        x = 0.0001
        x = random.randint(-PRZEDZIAL*100, PRZEDZIAL*100)
        x = x/100
        seq.append([x/200, f(x)/200])
    return seq

df = pd.DataFrame(rSeq(200), columns=['arg', 'res'])
# print(df)

testInputs = fRes(PRZEDZIAL)
fY = np.array(testInputs[1]).astype('float32')
testInputs = np.array(testInputs[0]).astype('float32')

trainInputs = df['arg'].values.astype('float32')
trainClasses = df['res'].values.astype('float32')

trainClasses = keras.utils.np_utils.to_categorical(trainClasses, num_classes=200).astype('float32')
testClasses = keras.utils.np_utils.to_categorical(fY, num_classes=200).astype('float32')

print('testClasses: ', testClasses, 'fY:', fY, sep='\n')

model = Sequential()
model.add(Dense(128, input_dim=1, activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(200, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(trainInputs, trainClasses, epochs=10, batch_size=64)

y_pred = model.predict(testInputs)

print('testInputs: ', testInputs, 'y_pred: ', y_pred, 'testClasses:', testClasses, sep='\n')

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

y = []
for i in pred:
    y.append((i - 100)/100*PRZEDZIAL)
y = np.array(y)

print('y: \n', y)

plt.plot(testInputs*200, fY*200, 'b--', testInputs*200, y, 'r-')
plt.show()