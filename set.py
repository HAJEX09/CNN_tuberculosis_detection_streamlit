import io
import streamlit as st
from PIL import Image
from keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.efficientnet import decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Загрузка модели
model = load_model('mymodel.h5')

def preprocess_image(img):
    img = img.resize((128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def print_predictions(preds):
    classes = decode_predictions(preds, top=2)[0]
    for cl in classes:
        st.write(cl[1], cl[2])


def print_predictions2(preds):
    if preds[[0]] < 0.5:
        st.write('Доброкачественная)')
    else:
        st.write('Злокачественная! Пожалуйста, обратитесь к доктору!')


st.title('Нейросеть для обнаружения рака кожи ')
img = load_image()
result = st.button('Распознать изображение')
if result:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write('**Результаты распознавания:**')
    print_predictions2(preds)
