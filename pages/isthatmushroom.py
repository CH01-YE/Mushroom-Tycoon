import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import base64
from keras.preprocessing import image
from PIL import Image, ImageOps
import cv2
import os
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from skimage.io import imread, imshow
from tensorflow import keras
from keras import layers
from keras import models
from keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions

import tensorflow_hub as hub
from tensorflow.keras.activations import softmax

st.markdown('<h1 style="color:black;">Is that mushroom?</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies image into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> Mushroom or not</h3>', unsafe_allow_html=True)

# background image to streamlit

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('../streamlit/mushroom.webp')

file_uploded= st.file_uploader('Insert image for classification', type=['png','jpg', 'jpeg'])
c1, c2= st.columns(2)
if file_uploded is not None:
    im= Image.open(file_uploded)
    img= np.asarray(im)
    img= cv2.resize(img,(200, 200))
    img= preprocess_input(img)
    img= np.expand_dims(img, axis=0)
    c1.header('Input Image')
    c1.image(im)
    c1.write(img.shape)
    
      #load weights of the trained model.
    classifier_model = tf.keras.models.load_model(r'../streamlit/pages/cnn_mushroom.h5')
    shape = ((200, 200, 3))
    model = tf.keras.Sequential(hub.KerasLayer(classifier_model, input_shape=shape))

    class_names = ['Mushroom',
                  'Thing']
    
    # predictions = model.predict(img)
  
    images = np.vstack([img])
    classes = model.predict(images, batch_size=10)
    # print(classes[0])
    if classes[0]<0.5:
        c2.header('Output')
        c2.subheader('It looks like a mushroom \n 버섯이네요.')

    else:
        c2.header('Output')
        c2.subheader("It doesn't look like a mushroom. \n 버섯이 아닌 것 같아요.")
    
else: 
  st.write('Please upload image to be classified')