# Virtual AI Assistant

This project provides a virtual AI assistant that integrates image-to-text and speech-to-text functionalities with audio responses. The assistant utilizes various models for processing inputs and generating responses.

## Features

- **Image-to-Text**: Describe images in detail using a pre-trained image-to-text model.
- **Speech-to-Text**: Convert spoken audio into text using the Whisper model.
- **Text-to-Speech**: Convert text responses into spoken audio using gTTS.
- **Gradio Interface**: A user-friendly web interface for interacting with the assistant via voice and images.

## Installation

To run this project, you need to install the necessary Python packages. Create a virtual environment and install the dependencies with:

```bash
# Create a virtual environment (optional)
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
