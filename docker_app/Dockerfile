#python parent image
FROM python:3.10-slim

# Install dep
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# ENV VARS
ENV TRANSFORMERS_CACHE=/models/huggingface

# Create app dir
WORKDIR /app

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy app files
COPY server.py blip.py flux_depth.py .

#Download model once
RUN python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large'); \
BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large')"

#expose server for st
EXPOSE 8501

#Run the app
CMD ["streamlit","run","server.py","--server.port=8501","--server.address=0.0.0.0"]
