import cv2
import tensorflow
import numpy as np 
import matplotlib.pyplot as plt 
model = tensorflow.keras.models.load_model('first_model.model')


alpha = [chr(i) for i in range(65, 91)]
alpha.append('nothing')
alpha.append('space')
alpha.append('del')

size = 120

def prepare():
    image = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)  
    image = image/255
    image = cv2.resize(image, (size, size))
    return image.reshape(-1, size, size, 1)


predictions = model.predict([prepare()])

new_pred = list(predictions[0])

index = new_pred.index(max(new_pred))

print(new_pred)
print(index)