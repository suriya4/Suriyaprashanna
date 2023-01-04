import cv2
import numpy as np
import streamlit as st
import sklearn
from keras.models import load_model


def image_process(img):
    image_1_resize = cv2.resize(img, (256, 256))
    image_1_b_w = cv2.cvtColor(image_1_resize, cv2.COLOR_BGR2GRAY)
    return image_1_b_w


def abs_diff(img1, img2):
    absdiff = cv2.absdiff(img1, img2)
    return absdiff


st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
loaded_model = load_model("E:\college\newmodel.h5")
font = cv2.FONT_HERSHEY_PLAIN

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    cv2.waitKey(3)
    _, frame1 = camera.read()
    img1 = image_process(frame)
    img2 = image_process(frame1)
    res = abs_diff(img1, img2)
    ans = np.dstack([res, res, res])
    ans = np.expand_dims(ans, axis=0)
    output = loaded_model.predict(ans)
    if output == 0:
        text = "Signing"
    else:
        text = "unsigning"

    display = cv2.putText(frame, text, (50, 50), font, 1,
                          (0, 255, 255),
                          2,
                          cv2.LINE_4)
    FRAME_WINDOW.image(display)

else:
    st.write('Stopped')
