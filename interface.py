from transformers import pipeline
import gradio as gr
from gtts import gTTS
import os

# Initialize the Whisper model for Hindi speech recognition
pipe = pipeline(model="ihimanshu-soni/whisper-small-hi")

def transcribe_and_speak(audio):
    # Transcribe audio to text
    text = pipe(audio)["text"]

    # Convert text to speech using gTTS
    tts = gTTS(text=text, lang='hi', slow=False)
    audio_output_path = "output_speech.mp3"
    tts.save(audio_output_path)

    return text, audio_output_path

# Create Gradio interface
iface = gr.Interface(
    fn=transcribe_and_speak,
    inputs=gr.Audio(type="filepath"),
    outputs=["text", gr.Audio(type="filepath")],
    title="Whisper Small Hindi with Text-to-Speech",
    description="Realtime demo for Hindi speech recognition using a fine-tuned Whisper small model, with text-to-speech output.",
)

iface.launch()
