import tensorflow as tf 
import pandas as pd 
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np  

X_train = pickle.load(open('X_train.pickle', 'rb'))
Y_train = pickle.load(open('Y_train.pickle', 'rb'))

name = 'sign_recognition'

tensorboard = TensorBoard(log_dir = f'logs\\{name}')

size = 120

X_train = np.array(X_train, dtype = np.uint8).reshape(-1, size, size, 1)
Y_train = np.array(Y_train)
X_train = X_train/255

model = Sequential()

model.add(Conv2D(128, (3, 3), input_shape = X_train.shape[1:]))
model.add(Activation('tanh'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(.4))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(.4))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(.4))

model.add(Dense(32))
model.add(Activation('tanh'))
model.add(Dropout(.4))

model.add(Flatten())

model.add(Dense(29))
model.add(Activation('softmax'))


opt = tf.keras.optimizers.SGD(lr = .01)

model.compile(optimizer = opt,
    loss ='sparse_categorical_crossentropy',
    metrics = ['accuracy', 'sparse_categorical_accuracy'])

model.fit(X_train, Y_train, validation_split = .3, epochs = 6, batch_size = 32, callbacks = [tensorboard])


model.save('new_model_2.model')