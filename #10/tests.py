import os, cv2, re, random
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras import layers, models, optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split
import random

model = models.load_model('model_keras.h5')

with open('test_images_dogs_cats', 'r') as F:
    test_images_dogs_cats = F.readlines()

X_test, Y_test = prepare_data(test_images_dogs_cats) #Y_test in this case will be []

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = val_datagen.flow(np.array(X_test), batch_size=batch_size)
prediction_probabilities = model.predict_generator(test_generator, verbose=1)

# counter = range(1, len(test_images_dogs_cats) + 1)
counter = test_images_dogs_cats

solution = pd.DataFrame({"id": counter, "label":list(prediction_probabilities)})
cols = ['label']

for col in cols:
    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)

solution.to_csv("dogsVScats.csv", index = False)

print('Finito :D')