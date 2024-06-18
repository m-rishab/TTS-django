# tts_app/tts.py
import os
import numpy as np
import torch
from transformers import BertTokenizer
from scipy.io.wavfile import write as write_wav
from bark import SAMPLE_RATE, generate_audio, preload_models
from encodec import EncodecModel
from IPython.display import Audio

SUPPORTED_LANGS = [
    ("English", "en"),
    ("German", "de"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Turkish", "tr"),
    ("Chinese", "zh"),
]

def generate_speech(text, language_code='en'):
    if language_code not in [lang[1] for lang in SUPPORTED_LANGS]:
        raise ValueError(f"Unsupported language code: {language_code}")

    preload_models()
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    semantic_tokens = generate_text_semantic(text, language_code, tokenizer)
    coarse_tokens = generate_coarse(semantic_tokens)
    fine_tokens = generate_fine(coarse_tokens)
    audio = decode_audio(fine_tokens)

    return audio

def generate_text_semantic(text, language_code, tokenizer):
    text = text.strip()
    assert len(text) > 0

    encoded_text = tokenizer.encode(text, add_special_tokens=False) + [10000]
    encoded_text = np.pad(encoded_text, (0, 256 - len(encoded_text)), constant_values=129595)
    
    encoded_text = torch.tensor(encoded_text).unsqueeze(0)
    model = load_model("text")
    with torch.no_grad():
        outputs = model(encoded_text)
    
    semantic_tokens = outputs[0].cpu().numpy().squeeze()
    return semantic_tokens

def generate_coarse(semantic_tokens):
    model = load_model("coarse")
    semantic_tokens = torch.tensor(semantic_tokens).unsqueeze(0)

    with torch.no_grad():
        outputs = model(semantic_tokens)
    
    coarse_tokens = outputs[0].cpu().numpy().squeeze()
    return coarse_tokens

def generate_fine(coarse_tokens):
    model = load_model("fine")
    coarse_tokens = torch.tensor(coarse_tokens).unsqueeze(0)

    with torch.no_grad():
        outputs = model(coarse_tokens)
    
    fine_tokens = outputs[0].cpu().numpy().squeeze()
    return fine_tokens

def decode_audio(fine_tokens):
    model = load_codec_model()
    fine_tokens = torch.tensor(fine_tokens).unsqueeze(0)

    with torch.no_grad():
        audio = model.decode(fine_tokens)
    
    return audio.cpu().numpy().squeeze()

def load_model(model_type):
    model_path = _get_ckpt_path(model_type)
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    return model

def _get_ckpt_path(model_type):
    base_dir = os.path.expanduser("~/.cache/suno/bark_v0")
    file_name = {
        "text": "text.pt",
        "coarse": "coarse.pt",
        "fine": "fine.pt"
    }[model_type]
    return os.path.join(base_dir, file_name)

def load_codec_model():
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.eval()
    return model
