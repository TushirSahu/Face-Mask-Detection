import cv2
from keras.models import  load_model
import numpy as np

model=load_model('./model-007.model')
labels_dict={0:'NO MASK',1:'MASK'}
color_dict={0:(0,0,255),1:(0,255,0)}
source=cv2.VideoCapture(0)

face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

size=4
while(True):
    (ret,img)=source.read()
    img=cv2.flip(img,1,1)
    mini=cv2.resize(img,(img.shape[1] // size,img.shape[0]//size))

    # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(mini)
    for f in faces:
        (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
        # Save just the rectangle faces in SubRecFaces
        face_img = img[y:y + h, x:x + w]
        resized = cv2.resize(face_img, (150, 150))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 150, 150, 3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)
        print(result)

        label = np.argmax(result, axis=1)[0]
        #
        cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
         break

cv2.destroyAllWindows()
source.release()