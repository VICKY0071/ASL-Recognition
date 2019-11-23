import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow
from gtts import gTTS



model = tensorflow.keras.models.load_model('second_model.model')

#url = 'your IP address here'


lowerbound = np.array([0, 48, 80])
upperbound = np.array([20, 255, 255])

size = 120

kernel_open = np.ones((5, 5))
kernel_close = np.ones((20, 20))

def prepare(frame):
    image = cv2.imread('C:\\Users\\Vikas Thapliyal\\Desktop\\final\\full_project\\train_image.jpg', cv2.IMREAD_GRAYSCALE)  
    image = image/255
    image = cv2.resize(image, (size, size))
    return image.reshape(-1, size, size, 1)           

from urllib.request import urlopen
alpha = [chr(i) for i in range(65, 91)]
alpha.append('nothing')
alpha.append('space')
alpha.append('del')

video = cv2.VideoCapture(0)
video.set(3, 480)
video.set(4, 800)

#video = cv2.VideoCapture('test_video.mp4')


while True:
    '''
    frame = urlopen(url)
    image_np = np.array(bytearray(frame.read()), dtype = np.uint8)
    frame = cv2.imdecode(image_np, -1)
    '''
    ret, frame = video.read()
    
    color_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(color_hsv, lowerbound, upperbound)

    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernel_close)

    maskFinal = maskClose

    conts, h =  cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    string = ''

    string_buffer = []
    if len(conts) == 1:
        x1, y1, w1, h1 = cv2.boundingRect(conts[0])
        cv2.rectangle(frame, (x1-50,y1-50), (x1+w1+50, y1+h1+50), (255, 0, 0), 3)
        cv2.imwrite('train_image.jpg', frame[y1:y1+h1+50, x1:x1+w1+50])

        try:    
            train_ = cv2.imread('train_image.jpg', cv2.IMREAD_GRAYSCALE)

            cv2.imshow('train_image', train_)
        except:
            pass

        predictions = model.predict([prepare(maskFinal)])
        print(predictions)
        new_pred= list(predictions[0])
        index = new_pred.index(max(new_pred))
        if index > .7:
            print(index)
            cv2.putText(frame, alpha[index], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            pass

    cv2.imshow('frame_window', frame)
    cv2.imshow('mask_close', maskClose)

    
    key = cv2.waitKey(1)
    if key == 27:
        break

    
cv2.destroyAllWindows() 