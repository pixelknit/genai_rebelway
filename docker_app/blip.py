import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

"""
def blip_process(device, processor, model, img_url):
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt").to(device, torch.float16)

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)
"""

class ModelUtils:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    def load_model(self, model_type):
        device = self.device 
        processor = None
        model = None
        if model_type == "blip":
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to(device)

        return processor, model, device
    
    def generate_caption(self, processor, model, device, img_url):
        try:
            raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
            text = "a photography of"
            inputs = processor(raw_image, text, return_tensors="pt").to(device, torch.float16)

            out = model.generate(**inputs)
            return processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            return f"Error: {str(e)}"


if __name__ == "__main__":

    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    my_model = ModelUtils()
    processor, model, device = my_model.load_model("blip")
    result = my_model.generate_caption(processor, model, device, img_url)
    print(result)

