import streamlit as st
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from blip import ModelUtils

@st.cache_resource
def load_server_model():
    my_model = ModelUtils()
    processor, model, device = my_model.load_model("blip")
    return processor, model, device, my_model

def main():
    st.title("Pipeline Models")
    st.selectbox("Select the Model:", ["blip", "flux","SD"])
    img_url = st.text_input("Enter img url")

    if img_url:
        processor, model, device, my_model = load_server_model()
        caption = my_model.generate_caption(processor, model, device, img_url)
        st.image(img_url, caption="Input Image", use_column_width=True)
        st.write("**Caption:**", caption)

if __name__ == "__main__":
    main()

