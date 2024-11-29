import json
import os.path

STORAGE = "storage"
VOICE_FILE_JSON = 'voice.json'

def writeVoice(name, path):

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

def readVoice(voice_key):
    path_voice = os.path.join(STORAGE, VOICE_FILE_JSON)
    with open(path_voice, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data[voice_key]


