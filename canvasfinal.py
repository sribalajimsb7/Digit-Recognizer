import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

m_new=tf.keras.models.load_model('model.h5')

img = np.zeros((512,512,1),dtype ='uint8')
#img[50:350,50:350]=0

wname='Canvas'
cv.namedWindow(wname)

state = False

def shape(event,x,y,flags,param):
    global state
    if event == cv.EVENT_LBUTTONDOWN:
        state = True
        cv.circle(img,(x,y),10,(255,0,0),-1)
    elif event == cv.EVENT_MOUSEMOVE:
        if state == True:
            cv.circle(img,(x,y),10,(255,0,0),-1)

    else:
        state  = False

cv.setMouseCallback(wname,shape)

while True:
    cv.imshow(wname,img)
    k = cv.waitKey(1)  
    if k == ord('q'):
        break
    elif k == ord('c'):
        img = np.zeros((512,512,1),dtype ='uint8')
    elif k == ord('w'):
        #out = img[50:350,50:350]
        cv.imwrite('sample1.jpg',img)
    elif k == ord('p'):
        img2 = img/255.0
        img2 = cv.resize(img2,(28,28),cv.INTER_AREA)
        img2 = img2.reshape(1,28,28)
        pr = np.argmax(m_new.predict(img2),axis=-1)
        print(pr)


cv.destroyAllWindows()
