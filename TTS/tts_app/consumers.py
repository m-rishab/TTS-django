# tts_app/consumers.py

import json
import base64
from channels.generic.websocket import WebsocketConsumer
from .tts import generate_speech

class TTSConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
    
    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        data = json.loads(text_data)
        text = data['text']
        language = data['language']

        # Generate speech audio
        audio_data = generate_speech(text, language)
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        self.send(text_data=json.dumps({
            'audio': audio_base64,
        }))
