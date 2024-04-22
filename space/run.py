""" App Boilerplate

import torch
import gradio as gr
import tempfile
import soundfile as sf
from DClassifier import CNN2DAudioClassifier
from audioMNIST import AudioHandler
import numpy as np

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN2DAudioClassifier()
model.load_state_dict(torch.load('audio_classifier_model.pth', map_location=device))
model.to(device)
model.eval()

def predict(audio_data):
        # Process audio file
        waveform, _ = AudioHandler.open(tmpfile.name)
        waveform = AudioHandler.pad(waveform)          
        mfcc = AudioHandler.mfcc(waveform)            
        mfcc = mfcc.unsqueeze(0).to(device)            

    # Predict using the model
    with torch.no_grad():
        outputs = model(mfcc)
        predicted_index = outputs.argmax(1)
        predicted_class = predicted_index.item()

    return f'Predicted Class: {predicted_class}'

# Setup Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type='numpy', label="Record or Upload Audio"),
    outputs="text"
)


# Launch the app
iface.launch()

"""