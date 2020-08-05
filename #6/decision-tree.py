import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
import io

from graphviz import Source
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('diabetes.csv')

all_inputs = df[['pregnant-times', 'glucose-concentr', 'blood-pressure', 'skin-thickness', 'insulin', 'mass-index', 'pedigree-func', 'age']].values
all_classes = df['class'].values
(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=1)

dtc = tree.DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)
print('Score of decision tree: ', dtc.score(test_inputs, test_classes))

# text_graph = tree.export.export_text(dtc)
# print(text_graph)

# dotfile = tree.export_graphviz(dtc, out_file=None, class_names=df['class'], feature_names=list(df.columns[:-1]))
# graph  = pydotplus.graph_from_dot_data(dotfile)
# graph.write_png('dtree.png')

# y_pred = dtc.predict(test_inputs)
# print(confusion_matrix(test_classes, y_pred))
# print(classification_report(test_classes, y_pred))
