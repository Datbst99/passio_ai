import json
import os.path
import subprocess
from datetime import datetime
from pathlib import Path

import ffmpeg

STORAGE = "storage"
VOICE_FILE_JSON = 'voice.json'
UPLOAD_FOLDER = 'storage/uploads'
AUDIO_SAMPLE_FOLDER = 'storage/audio_samples'
AUDIO_OUTPUT_FOLDER = 'storage/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
CURRENT_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = CURRENT_PATH.parent.parent

def get_voice_storage(file_name):
    full_path = os.path.join(AUDIO_OUTPUT_FOLDER, file_name)
    return os.path.join(PROJECT_DIR,full_path) if os.path.isfile(full_path) else None


def get_voice_path(voice_key):
    path_voice = os.path.join(STORAGE, VOICE_FILE_JSON)
    with open(path_voice, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if voice_key in data and 'audio_file' in data[voice_key]:
        audio_file = data[voice_key]['audio_file']
        if os.path.exists(audio_file):
            return audio_file

    return None


def upload_voice(file, name = None):
    mp3_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(mp3_path)

    subprocess.run(
        [
            "deepFilter",
            mp3_path,
            "-o",
        ]
    )

    timestamp = int(datetime.now().timestamp() * 1000)
    wav_output_path = os.path.join(AUDIO_SAMPLE_FOLDER, f"sample_{timestamp}.wav")
    ffmpeg.input(mp3_path).output(
        wav_output_path,
        ar=44100,
        ac=1,
        ab="384K",
        format="wav",
        acodec="libmp3lame",
        strict='normal'
    ).run(overwrite_output=True)

    if name is not None:
        voice_name = name
    else:
        voice_name = str(timestamp)

    voice_key = voice_write(voice_name, wav_output_path)

    return voice_key

def voice_write(name, path):

    path_voice = os.path.join(STORAGE, VOICE_FILE_JSON)

    with open(path_voice, 'r', encoding='utf-8') as file:
        data = json.load(file)

    current_keys = [int(key.split('_')[-1]) for key in data.keys() if key.split('_')[-1].isdigit()]
    max_key = max(current_keys, default=0)
    new_key = f"eco_voice_{max_key + 1}"
    data[new_key] = {
        'voice_name' : name,
        'audio_file' : path
    }

    with open(path_voice, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    return new_key