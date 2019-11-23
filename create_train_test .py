import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os
import pickle

size = 120
alpha = [chr(i) for i in range(65, 91)]
alpha.append('nothing')
alpha.append('space')
alpha.append('del')

images = 'C:\\Users\\Vikas Thapliyal\\Desktop\\WOW\\asl_alphabet_train\\asl_alphabet_train'

training_data = []

for alphabet in alpha:
    label = alpha.index(alphabet)
    print(label)
    path = os.path.join(images, alphabet)
    for image in os.listdir(path):
        img = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (size, size))    
        training_data.append([img_resized, label])

import random

random.shuffle(training_data)

X_train = []
Y_train = []

for features, label in training_data:
    X_train.append(features)
    Y_train.append(label)

pickle_out = open('Y_train.pickle', 'wb')
pickle.dump(Y_train, pickle_out)
pickle_out.close()

pickle_out = open('X_train.pickle', 'wb')
pickle.dump(X_train, pickle_out)
pickle_out.close() 

