import streamlit as st
import os
import random
import string
import numpy as np
import librosa
from silero_vad import get_speech_timestamps, read_audio, load_silero_vad
from yt_dlp import YoutubeDL
from pydub import AudioSegment
import torch
import nemo.collections.asr as nemo_asr
from googletrans import Translator

# Helper function to generate a random string
def generate_random_string(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

# Function to download YouTube audio as a WAV file
def download_youtube_audio_as_wav(url):
    random_filename = generate_random_string()  # No .wav extension here
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': random_filename,  # No .wav extension in outtmpl
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        # yt-dlp will append .wav to the filename during post-processing
        downloaded_file = f"{random_filename}.wav"
        return os.path.abspath(downloaded_file)  # Return absolute path
    except Exception as e:
        st.error(f"Failed to download YouTube audio: {e}")
        return None

# Function to process WAV files and extract speech chunks
def process_wav_files(wav_file_path):
    try:
        # Load the Silero VAD model
        model = load_silero_vad()

        # Create output folder for speech chunks
        base_name = os.path.splitext(os.path.basename(wav_file_path))[0]
        output_folder = f"{base_name}_chunks"
        os.makedirs(output_folder, exist_ok=True)

        # Read the audio and get speech timestamps
        wav = read_audio(wav_file_path)
        speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)

        # Load the audio file using pydub
        audio = AudioSegment.from_file(wav_file_path)

        # Split the audio based on speech timestamps and save segments
        chunk_paths = []
        for i, timestamp in enumerate(speech_timestamps):
            start_ms = int(timestamp['start'] * 1000)
            end_ms = int(timestamp['end'] * 1000)
            segment = audio[start_ms:end_ms]

            # Save the segment as a WAV file in the subfolder
            output_filename = os.path.join(output_folder, f"segment_{i}.wav")
            segment.export(output_filename, format="wav")
            chunk_paths.append(output_filename)

        return chunk_paths
    except Exception as e:
        st.error(f"Failed to process WAV file: {e}")
        return []

# Function to preprocess audio
def preprocess_audio(audio_path):
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    return audio

# Function to process and transcribe audio
def process_and_transcribe(wav_file, asr_model):
    if wav_file is not None:
        # Preprocess and transcribe
        preprocessed_audio = preprocess_audio(wav_file)
        audio_array = np.array(preprocessed_audio)
        transcript = asr_model.transcribe([audio_array])
        return transcript[0]
    else:
        return None

# Function to translate Persian text to English
def translate_to_en(text):
    translator = Translator()
    translation = translator.translate(text[0], src='fa', dest='en')
    return translation.text

# Initialize your ASR model
@st.cache_resource
def load_asr_model():
    return nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="nvidia/stt_fa_fastconformer_hybrid_large")

# Streamlit UI starts here
st.title("YouTube Audio Downloader and Speech Chunk Extractor")

# Initialize session state for file persistence
if 'wav_file_path' not in st.session_state:
    st.session_state.wav_file_path = None
if 'chunk_paths' not in st.session_state:
    st.session_state.chunk_paths = []
if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = {}
if 'translations' not in st.session_state:
    st.session_state.translations = {}

# Load ASR model
asr_model = load_asr_model()

# Input for YouTube URL
youtube_url = st.text_input("Enter YouTube Video URL:")

if youtube_url:
    if st.button("Download and Process Audio"):
        with st.spinner("Downloading and extracting audio..."):
            # Step 1: Download YouTube audio as WAV
            wav_file_path = download_youtube_audio_as_wav(youtube_url)
            if wav_file_path and os.path.exists(wav_file_path):
                st.session_state.wav_file_path = wav_file_path  # Persist file path in session state
                st.success(f"Audio downloaded successfully: {wav_file_path}")
            else:
                st.error("Failed to download or locate the audio file.")

# Check if the file exists in session state
if st.session_state.wav_file_path and os.path.exists(st.session_state.wav_file_path):
    # Play the downloaded audio
    st.audio(st.session_state.wav_file_path, format="audio/wav")

    # Step 2: Process WAV file to extract speech chunks
    if st.button("Extract Speech Chunks"):
        with st.spinner("Extracting speech chunks..."):
            chunk_paths = process_wav_files(st.session_state.wav_file_path)
            st.session_state.chunk_paths = chunk_paths  # Persist chunk paths in session state
            num_chunks = len(chunk_paths)

        if num_chunks > 0:
            st.success(f"Number of voice chunks extracted: {num_chunks}")

            
    # Step 3: Display extracted chunks with playback options
    st.subheader("Extracted Voice Chunks")
    for i, chunk_path in enumerate(st.session_state.chunk_paths):
        st.write(f"Chunk {i + 1}:")
        st.audio(chunk_path, format="audio/wav")

        # Use a form for each chunk to avoid rerunning the entire script
        with st.form(key=f"form_{i}"):
            submit_button = st.form_submit_button(f"Transcribe and Translate Chunk {i + 1}")
            if submit_button:
                with st.spinner("Transcribing and translating..."):
                # Transcribe the chunk
                    transcript = process_and_transcribe(chunk_path, asr_model)
                if transcript:
                    # Store transcription in session state
                    st.session_state.transcriptions[i] = transcript
                    st.write(f"**Transcription (Persian):** {transcript}")

                    # Translate the transcript
                    translation = translate_to_en(transcript)
                    st.session_state.translations[i] = translation
                    st.write(f"**Translation (English):** {translation}")
                else:
                    st.error("Failed to transcribe the audio chunk.")

        # Display stored transcription and translation if available
        if i in st.session_state.transcriptions:
            st.write(f"**Stored Transcription (Persian):** {st.session_state.transcriptions[i]}")
        if i in st.session_state.translations:
            st.write(f"**Stored Translation (English):** {st.session_state.translations[i]}")