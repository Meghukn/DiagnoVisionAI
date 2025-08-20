from gtts import gTTS

def text_to_speech(input_text: str, output_filepath: str = "Temp.mp3") -> str:
    """Generate speech using gTTS and save to output_filepath (mp3)."""
    tts = gTTS(text=input_text or "I'm sorry, I have no response.", lang="en")
    tts.save(output_filepath)
    return output_filepath
