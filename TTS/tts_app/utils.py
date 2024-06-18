# tts_app/utils.py
import io
from suno import bark

def generate_tts_audio(text, language, speaker):
    # Replace this with the actual TTS generation logic using the bark model
    audio = bark.generate_audio(text, language=language, speaker=speaker)
    audio_data = io.BytesIO()
    audio.export(audio_data, format='wav')
    return audio_data.getvalue()
