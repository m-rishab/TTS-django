# tts_app/routing.py

from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/tts/', consumers.TTSConsumer.as_asgi()),
]
