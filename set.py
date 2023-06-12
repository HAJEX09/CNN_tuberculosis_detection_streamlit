from xml.etree.ElementTree import PI
import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import emoji 

model_path= "mymodel.h5"

st.title("TuberCulosis Detection System 💻")
st.write()
st.write(" ")


header = st.container()
with header:
  st.write(">Below is the CNN(lenet) model which i used for the tuberculosis detection using human chest X-rays ")
  img = Image.open('mymodel.h5.png')
  
  st.image(img,caption='CNN model')


img = Image.open("hello.png")
st.image(img)


upload = st.file_uploader('Upload a chest X-ray image')

if upload is not None:
  file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  
  img = Image.open(upload)
  st.image(img,caption='Uploaded Image',width=300)
  if(st.button('Analyze 👈')):
    model = tf.keras.models.load_model(model_path)
    # x = cv2.resize(opencv_image,(28,28))
    # x = np.expand_dims(x,axis=0) 
    x=[] 
    img = cv2.resize(opencv_image,(28,28))
    if img.shape[2]==1:
      img=np.dstack([img,img,img])
    # opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB) # Color from BGR to RGB
    
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img =np.array(img)
    img=img/255
    x.append(img)  
    y = model.predict(np.array(x))
    # st.write(y)
    y=(y>0.5)
    ans=y[0,1]
    # st.write(ans)
    if(ans==0):
      st.success('Tuberculosis Not Detected !!')
    elif(ans==1):
      st.error('Tuberculosis Detected !!')
    else:
      st.error('Other Pulmonary Disorder')
