import hashlib
import os
from datetime import datetime
from pathlib import Path

import ffmpeg
import torch
import torchaudio

from TTS.TTS.tts.configs.xtts_config import XttsConfig
from TTS.TTS.tts.models.xtts import Xtts
from .TextFormatService import normalize_vietnamese_text
from queue import Queue

from ..models.xtts_model import XTTSModel

CHECKPOINT_DIR = 'model/'
REPO_ID = "capleaf/viXTTS",
USE_DEEPSPEED = False
OUTPUT_DIR = 'storage/outputs'
XTTS_MODEL = None
CONDITIONING_LATENTS_CACHE = {}
TOKENIZER_PATH = 'model/vocab.json'
model_queue = Queue()

def xtts_to_speech(text, file_name, speaker_audio_file, speed = 1):
    if not speaker_audio_file:
        return "Bạn cần cung cấp tệp âm thanh tham chiếu!", None

    text = normalize_vietnamese_text(text)
    gpt_cond_latent, speaker_embedding = _extract_latents(speaker_audio_file)
    print("process 1")
    model = model_queue.get()
    try:
        wav_chunks = model.inference(
            text=text,
            language="vi",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.3,
            length_penalty=1.0,
            repetition_penalty=8.0,
            top_k=30,
            top_p=0.85,
            enable_text_splitting=True,
            speed=_adjust_number(speed),
        )
    finally:
        torch.cuda.empty_cache()
        model_queue.put(model)


    out_wav = torch.from_numpy(wav_chunks["wav"]).unsqueeze(0)
    wav_output_path = os.path.join(OUTPUT_DIR, f"{file_name}.wav")
    torchaudio.save(wav_output_path, out_wav, 24000)
    return _convert_wav_to_mp3(wav_output_path)

def _load_model():
    global XTTS_MODEL

    print("Loading model...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xtts_config = os.path.join(CHECKPOINT_DIR, "config.json")
    config = XttsConfig()
    config.load_json(xtts_config)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=CHECKPOINT_DIR, use_deepspeed=USE_DEEPSPEED, eval=True)

    if torch.cuda.is_available():
        model.cuda()

    XTTS_MODEL = model

def _extract_latents(speaker_audio_file):
    global CONDITIONING_LATENTS_CACHE

    cache_key = (
        speaker_audio_file,
        XTTS_MODEL.config.gpt_cond_len,
        XTTS_MODEL.config.max_ref_len,
        XTTS_MODEL.config.sound_norm_refs,
    )

    if cache_key in CONDITIONING_LATENTS_CACHE:
        gpt_cond_latent, speaker_embedding = CONDITIONING_LATENTS_CACHE[cache_key]
    else:
        gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
            audio_path=speaker_audio_file,
            gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
            max_ref_length=XTTS_MODEL.config.max_ref_len,
            sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
        )
        CONDITIONING_LATENTS_CACHE[cache_key] = (gpt_cond_latent, speaker_embedding)

    return gpt_cond_latent, speaker_embedding

def _convert_wav_to_mp3(wav_file_path):
    mp3_file_path = wav_file_path.replace(".wav", ".mp3")

    ffmpeg.input(wav_file_path).output(
        mp3_file_path,
        ar=44100,  # Sampling rate 44.1 kHz
        ac=1,
        ab="128k",
        format="mp3",
        acodec="libmp3lame",
        strict='normal'
    ).run(overwrite_output=True)

    delete_file = Path(wav_file_path)
    if delete_file.exists():
        delete_file.unlink()

    return mp3_file_path

def _adjust_number(num):
    if num is None:
        return 1
    elif num < 0.8:
        return 0.8
    elif num > 1.5:
        return 1.5
    return num

if XTTS_MODEL is None:
    _load_model()

    for i in range(1):
        model_queue.put(XTTSModel(model_id=i).initialization(use_deepspeed=True))
