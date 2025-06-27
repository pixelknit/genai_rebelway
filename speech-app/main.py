import sounddevice as sd
from scipy.io.wavfile import write
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

AUDIO_FILE = "test.wav"

@app.post("/record")
def record_audio():
    fs = 16000
    seconds = 5
    print("Recording...")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(AUDIO_FILE, fs, audio)
    print("Saved as ", AUDIO_FILE)
    return {"status": "recorded", "filename": AUDIO_FILE}

@app.get("/transcribe")
def transcribe_audio():
    device = torch.device("mps")
    torch_dtype = torch.float32

    model_id = "nyrahealth/CrisperWhisper"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )

    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps="word",
        torch_dtype=torch_dtype,
        device=device,
    )

    if not os.path.exists(AUDIO_FILE):
        return {"error": "Audio file not found"}

    result = pipe(AUDIO_FILE)
    transcript = result.get("text", "")
    return {"transcript": transcript}
    

