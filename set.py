import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image

st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def loading_model():
    fp = "./model.h5"
    model_loader = load_model(fp)
    return model_loader


cnn = loading_model()
st.write("""
# Нейросеть для обнаружения туберкулеза по рентгеновскому снимку

""")


temp = st.file_uploader("Выберите изображение для распознавания")
#temp = temp.decode()

buffer = temp
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))


if buffer is None:
    st.text("Загрузите изображение")

else:

    img = image.load_img(temp_file.name, target_size=(
        500, 500), color_mode='grayscale')

    # Preprocessing the image
    pp_img = image.img_to_array(img)
    pp_img = pp_img/255
    pp_img = np.expand_dims(pp_img, axis=0)

    # predict
    preds = cnn.predict(pp_img)
    if preds >= 0.5:
        out = ('{:.2%}  '.format(
            preds[0][0]))

    else:
        out = ('{:.2%} '.format(
            1-preds[0][0]))

    st.success(out)

    image = Image.open(temp)
    st.image(image, use_column_width=True)
