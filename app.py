import os
# import datetime/
# from datetime import datetime
import av
import cv2
import time
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
from time import sleep
from aiortc.contrib.media import MediaPlayer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# def_values ={'conf': 70, 'nms': 50} 
# keys = ['conf', 'nms']

# def parameter_sliders(key, enabled = True, value = None):
#     conf = custom_slider("Model Confidence", 
#                         minVal = 0, maxVal = 100, InitialValue= value[0], enabled = enabled,
#                         key = key[0])
#     nms = custom_slider('Overlapping Threshold', 
#                         minVal = 0, maxVal = 100, InitialValue= value[1], enabled = enabled,
#                         key = key[1])

        
#     return(conf, nms)


# importing the necessary files
faceCascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
model =load_model(r'model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

#creating a function that predicts emotions using the files above
def result(img):
    frame=np.array(img)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = model.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y-10)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    return frame,label



@st.cache    

# a class that captures real time webcam feed
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray)
        if faces is ():
            return img

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = model.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y-10)
            cv2.putText(img,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)    
        return img




# main function
def main():
    st.title("Face Emotion Detection App ")
    st.header("Created by - Atul Chouhan ")

    activities = ["Play with camera"]
#     list_of_days = [1,2,3,4,5,6,7]
#     st.slider("Plotting window", 
#     value=(min(list_of_days).timestamp(), max(list_of_days).timestamp()), 
#     format='%s'.format(datetime.utcfromtimestamp().strftime('%d/%m/%y')))
    st.title('model confidance')
    temp_option = [70, 75]
    temp = st.select_slider('choose confidance level', options = temp_option)
    
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Play with camera":
        st.subheader("Real time face emotion detection")
        try:
            webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
        except Exception:
            st.error("oops there seems to be an error.")
    


# calling main function
main()
