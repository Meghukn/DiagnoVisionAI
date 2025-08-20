import os
import logging
import speech_recognition as sr
from pydub import AudioSegment

from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path: str, timeout: int = 20, phrase_time_limit: int | None = None) -> str:
    """Record audio from microphone and save as MP3 at file_path."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        logging.info("Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=1)
        logging.info("Start speaking now...")
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        logging.info("Recording complete.")

    # Save WAV first
    wav_path = file_path if file_path.lower().endswith(".wav") else file_path.replace(".mp3", ".wav")
    with open(wav_path, "wb") as f:
        f.write(audio.get_wav_data())

    # Convert to MP3 (requires ffmpeg installed)
    try:
        sound = AudioSegment.from_wav(wav_path)
        mp3_path = file_path if file_path.lower().endswith(".mp3") else file_path.replace(".wav", ".mp3")
        sound.export(mp3_path, format="mp3")
        logging.info(f"Audio saved to {mp3_path}")
        return mp3_path
    except Exception as e:
        logging.warning(f"MP3 conversion failed ({e}); keeping WAV at {wav_path}")
        return wav_path

def transcribe_with_groq(stt_model: str, audio_filepath: str, api_key: str) -> str:
    """Transcribe audio using Groq Whisper API."""
    client = Groq(api_key=api_key)
    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en",
        )
    return transcription.text
