import io
import streamlit as st
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import os
import time
import pygame
from gtts import gTTS
import speech_recognition as sr
from googletrans import LANGUAGES, Translator
import pyttsx3
from streamlit_cropperjs import st_cropperjs

engine = pyttsx3.init()
isTranslateOn = False

translator = Translator() # Initialize the translator module.
pygame.mixer.init()  # Initialize the mixer module.

# Create a mapping between language names and language codes
language_mapping = {name: code for code, name in LANGUAGES.items()}

def get_language_code(language_name):
    return language_mapping.get(language_name, language_name)

def translator_function(spoken_text, from_language, to_language):
    return translator.translate(spoken_text, src='{}'.format(from_language), dest='{}'.format(to_language))


def  text_to_voice(text_data, to_language):
    st.header("Translated Text...")
    st.markdown(f"<div style='background-color:black; font-size: 25px;'>{text_data}</div>", unsafe_allow_html=True)
    
    myobj = gTTS(text=text_data, lang='{}'.format(to_language), slow=False)
    myobj.save("cache_file.mp3")
    audio = pygame.mixer.Sound("cache_file.mp3")  # Load a sound.
    audio.play()
    # Wait for the sound to finish playing
    # while pygame.mixer.get_busy():
    #     pygame.time.delay(20)
    os.remove("cache_file.mp3")

class StreamlitApp:
    def __init__(self):
        self.font_path = r"ocr\PaddleOCR\doc\fonts\latin.ttf"
        self.ocr_model = PaddleOCR(lang="en") 

    def process_output(self, uploaded_image):
        npimg = np.fromstring(uploaded_image.read(), np.uint8)
        image_decode = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        res = self.ocr_model.ocr(image_decode)
        boxes = [box[0] for box in res]
        text = [tx[1][0] for tx in res]
        acc = [sc[1][1] for sc in res]
        return image_decode, boxes, text, acc
 
    def predict_text(self, uploaded_image):
        image_decode, _, text, _ = self.process_output(uploaded_image)
        return {"text": text, "image": image_decode}

    def predict_img(self, uploaded_image):
        image_decode, boxes, text, acc = self.process_output(uploaded_image)
        drawing = draw_ocr(image_decode, boxes, text, acc, font_path=self.font_path)
        return drawing

app = StreamlitApp()

# st.title("📷 OCR Streamlit App")

# Dropdowns for selecting languages 
def ocr_prediction():
    
    uploaded_image = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        # uploaded_image=uploaded_image.read()
        cropped_pic = st_cropperjs(pic=uploaded_image.read(), btn_text="Detect!", key="foo")
        if cropped_pic:

            from_language_name = st.selectbox("Select Source Language:", list(LANGUAGES.values()))
            to_language_name = st.selectbox("Select Target Language:", list(LANGUAGES.values()))
            print(type(cropped_pic))
            image = io.BytesIO(cropped_pic)
            # Convert language names to language codes
            from_language = get_language_code(from_language_name)
            to_language = get_language_code(to_language_name)
            if st.button("🔍 Predict Text"):
                result = app.predict_text(image)
                st.write("📖 Detected Text:")
                
                # Combine all the text into one string
                combined_text = " ".join(result["text"])

                # Translate and speak the combined text
                translated_text = translator_function(combined_text, from_language, to_language)
                # st.write(combined_text)
                st.markdown(f"<div style='background-color: black; font-size: 25px;'>{combined_text}</div>", unsafe_allow_html=True)

                text_to_voice(translated_text.text, to_language)

                # text_to_voice(translated_text.text, to_language)

            if st.button("🎨 Show Processed Image"):
                st.image(app.predict_img(image), caption="Processed Image", use_column_width=True)


# ocr_prediction()
