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

    if isinstance(audio_data, tuple) and len(audio_data) == 2:
        data, sample_rate = audio_data
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            # Process data here as before
        else:
            return "Received non-array audio data"
    else:
        return f"Unexpected input type or structure: {type(audio_data)}"

    # Write the audio data to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmpfile:
        sf.write(tmpfile.name, data, sample_rate)
        tmpfile.flush()  # Make sure to flush so data is written before reading
        
        # Process audio file
        waveform, _ = AudioHandler.open(tmpfile.name)  # Load waveform
        waveform = AudioHandler.pad(waveform)          # Apply padding
        mfcc = AudioHandler.mfcc(waveform)             # Compute MFCCs
        mfcc = mfcc.unsqueeze(0).to(device)            # Add batch dimension and send to device

    # Predict using the model
    with torch.no_grad():
        outputs = model(mfcc)
        predicted_index = outputs.argmax(1)
        predicted_class = predicted_index.item()

    return f'Predicted Class: {predicted_class}'

# Setup Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type='numpy', label="Record or Upload Audio"),  # Ensure the correct type is set
    outputs="text"
)


# Launch the app
iface.launch()
