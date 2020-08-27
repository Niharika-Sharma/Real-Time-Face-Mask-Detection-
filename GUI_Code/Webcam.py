import keras
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import keras
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tkinter


window = tkinter.Tk()

model = keras.models.load_model('CNN_face-mask.model')
#Provide path to your model

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict={0:'without_mask',1:'with_mask'}
color_dict={0:(0,0,255),1:(0,255,0)}
#size = 4

webcam = cv2.VideoCapture(0) #Use camera 0

    # We load the xml file
classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,1) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // 4, im.shape[0] // 4))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * 4 for v in f] #Scale the shapesize backup
        #Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(150,150))
        test_image = img_to_array(resized)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        
        
        #print(result, result[0][0])
        if result[0][0] == 1:
            label = 1
        else:
            label = 0
            
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    # Show the image
    cv2.imshow('Webcam',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27:
        window.destroy()
        #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()

window.mainloop()    
