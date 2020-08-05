import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

X_principal = pd.read_csv('iris2D.csv')
print(X_principal.head()) 
#WYRZUĆ KOLUMNĘ ID
db_default = DBSCAN(eps = 2.5, min_samples = 3).fit(X_principal) 
labels = db_default.labels_ 
print(labels)
colours = {} 
colours[0] = 'r'
colours[1] = 'g'
colours[2] = 'b'
colours[-1] = 'k'

cvec = [colours[label] for label in labels] 

r = plt.scatter(X_principal['PC1'], X_principal['PC2'], color ='r'); 
g = plt.scatter(X_principal['PC1'], X_principal['PC2'], color ='g'); 
b = plt.scatter(X_principal['PC1'], X_principal['PC2'], color ='b'); 
k = plt.scatter(X_principal['PC1'], X_principal['PC2'], color ='k'); 

plt.scatter(X_principal['PC1'], X_principal['PC2'], c = cvec) 

plt.legend((r, g, b, k), ('Label 0', 'Label 1', 'Label 2', 'Label -1')) 

plt.show() 
