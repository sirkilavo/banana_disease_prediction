import streamlit as st 
import keras
import numpy as np
from keras.preprocessing.image import img_to_array
import tensorflow_hub as hub
from keras.preprocessing import image
import tensorflow as tf
import cv2 
from PIL import Image
import joblib
st.title('Banana Disease Prediction System')
st.image('image/profile1.jpg', width = 650)
st.write('Banana Disease Prediction was the system that are used for predicting disease occured on banana crop.The system based on on three classes, two are diseases and one is health banana crops,diseases include Black Sigatoka and Fusarium wilt. ')


st.sidebar.title('Make Prediction')
with st.sidebar:
    uploaded_img = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])
    if uploaded_img is not None:
        st.image(uploaded_img)
        if st.button('Predict'):
            img = image.load_img(uploaded_img, target_size=(224, 224))
            #deep learning models expect a batches of images as input, so we create batch of images
            img = np.array(img)/255.0
            
            @keras.saving.register_keras_serializable(package="MyLayers")
            class CustomLayer(keras.layers.Layer):
                def __init__(self, factor):    
                   super().__init__()
                   self.factor = factor

                def call(self, x):
                   return x * self.factor

                def get_config(self):
                   return {"factor": self.factor}


            @keras.saving.register_keras_serializable(package="my_package", name="custom_fn")
            def custom_fn(x):
                
                return x**2
        
            #load a model
            model = tf.keras.models.load_model('model/my_model.keras', custom_objects={'KerasLayer':hub.KerasLayer, "custom_fn": custom_fn})
            result = model.predict(img[np.newaxis, ...])
           
            #Predict the image by using pretrained data
            if np.argmax(result[0]) == 0:
                st.write("Black Sigatoka")
            elif np.argmax(result[0]) == 1:
                st.write("Fusarium Wilt")    
            else:
                st.write("Health")
           
         