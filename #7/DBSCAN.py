import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

df = pd.read_csv('iris2D.csv')
x = df.iloc[:, [1,2]].values
y_db = DBSCAN(eps=0.3, min_samples=3, ).fit_predict(x)
# plt.show()
# Building the label to colour mapping 
colours = {} 
colours[0] = 'r'
colours[1] = 'g'
colours[2] = 'b'
colours[-1] = 'k'

# Building the colour vector for each data point 
cvec = [colours[label] for label in y_db] 

# For the construction of the legend of the plot 
r = plt.scatter(x[0], x[1], color ='r') 
g = plt.scatter(x[0], x[1], color ='g') 
b = plt.scatter(x[0], x[1], color ='b') 
k = plt.scatter(x[0], x[1], color ='k') 

# Plotting P1 on the X-Axis and P2 on the Y-Axis 
# according to the colour vector defined 
plt.scatter(x[0], x[1], c = cvec) 

# Building the legend 
plt.legend((r, g, b, k), ('Label 0', 'Label 1', 'Label 2', 'Label -1')) 

plt.show() 
