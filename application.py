import torch
from transformers import pipeline
import whisper
import gradio as gr
import os
from gtts import gTTS
from PIL import Image
import re
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Setup the image-to-text pipeline
model_id = "llava-hf/llava-1.5-7b-hf"

# Initialize pipeline for image-to-text with fallback to CPU if MPS is not available
if torch.backends.mps.is_available():
    device = 0  # Use device 0 for MPS if available
else:
    device = -1  # Use CPU

pipe = pipeline("image-to-text", model=model_id, device=device)

# Load the Whisper model
DEVICE = "cpu"  # Force the use of CPU for Whisper model
model = whisper.load_model("medium", device=DEVICE)

# Function to process image and text
def img2txt(input_text, input_image):
    # Load the image
    image = Image.open(input_image)

    if isinstance(input_text, tuple):
        prompt_instructions = """
        Describe the image using as much detail as possible, is it a painting, a photograph, what colors are predominant, what is the image about?
        """
    else:
        prompt_instructions = """
        Act as an expert in imagery descriptive analysis, using as much detail as possible from the image, respond to the following prompt:
        """ + input_text

    prompt = "USER: <image>\n" + prompt_instructions + "\nASSISTANT:"
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

    if outputs and len(outputs[0]["generated_text"]) > 0:
        match = re.search(r'ASSISTANT:\s*(.*)', outputs[0]["generated_text"])
        if match:
            reply = match.group(1)
        else:
            reply = "No response found."
    else:
        reply = "No response generated."

    return reply

# Function to transcribe audio using Whisper
def transcribe(audio):
    if audio is None or audio == '':
        return '', '', None

    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    result_text = result.text

    return result_text

# Function to convert text to speech using gTTS
def text_to_speech(text, file_path):
    language = 'en'
    audioobj = gTTS(text=text, lang=language, slow=False)
    audioobj.save(file_path)
    return file_path

# Main function to handle audio and image inputs
def process_inputs(audio_path, image_path):
    # Process the audio file
    speech_to_text_output = transcribe(audio_path)

    # Process the image file
    if image_path:
        chatgpt_output = img2txt(speech_to_text_output, image_path)
    else:
        chatgpt_output = "No image provided."

    # Convert the response to audio
    processed_audio_path = text_to_speech(chatgpt_output, "Temp3.mp3")

    return speech_to_text_output, chatgpt_output, processed_audio_path

# Create the Gradio interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="AI Output"),
        gr.Audio(label="Response Audio", type="filepath")
    ],
    title="Virtual AI Assistant",
    description="Upload an image and interact via voice input and audio response."
)

# Launch the interface
iface.launch(debug=True)
