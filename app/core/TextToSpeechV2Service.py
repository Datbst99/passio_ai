import os
import subprocess
import torch
import ffmpeg
import torchaudio

from TTS.TTS.tts.configs.tortoise_config import TortoiseConfig
from TTS.TTS.tts.layers.tortoise import audio_utils
from TTS.TTS.tts.layers.tortoise.audio_utils import load_voices
from TTS.TTS.tts.models.tortoise import Tortoise
from vinorm import TTSnorm
from huggingface_hub import snapshot_download
from underthesea import sent_tokenize
from datetime import datetime
from pathlib import Path

conditioning_latents_cache = {}

class TextToSpeechService:

    def __init__(self, checkpoint_dir="model/", repo_id="capleaf/viXTTS", use_deepspeed=False):
        self.checkpoint_dir = checkpoint_dir
        self.repo_id = repo_id
        self.use_deepspeed = use_deepspeed
        current_path = Path(os.path.dirname(os.path.abspath(__file__)))
        project_dir = current_path.parent.parent
        self.output_dir = os.path.join(project_dir, "outputs")
        self.model = self._load_model()

    def text_to_speech(self, text, speaker_audio_file, speed):
        if not speaker_audio_file:
            return "Bạn cần cung cấp tệp âm thanh tham chiếu!", None

        text = self._normalize_vietnamese_text(text)
        # gpt_cond_latent, speaker_embedding = self._extract_latents(speaker_audio_file)
        # sentences = sent_tokenize(text)
        voice_samples, conditioning_latents = load_voices(['dat'], ["model"])
        wav_chunks = self.model.inference(
                text=text,
                voice_samples=voice_samples,
                conditioning_latents=conditioning_latents,
                temperature=0.3,
                length_penalty=1.0,
                repetition_penalty=8.0,
                top_k=30,
                top_p=0.85

            )

        out_wav = torch.cat(wav_chunks["wav"], dim=0).unsqueeze(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_output_path = os.path.join(self.output_dir, f"output_{timestamp}.wav")
        torchaudio.save(wav_output_path, out_wav, 24000)

        return wav_output_path


    def predict_speaker(self, filename):
        subprocess.run(
            [
                "deepFilter",
                filename,
            ]
        )
        wav_path = os.path.splitext(filename)[0] + '.wav'

        ffmpeg.input(filename).output(
            wav_path,
            ar=44100,
            ac=1,
            ab="384K",
            format="wav",
            acodec="libmp3lame",
            strict='normal'
        ).run()

        delete_file = Path(filename)
        if delete_file.exists():
            delete_file.unlink()

        return wav_path

    def _load_model(self):
        print("Loading model...")
        self._clear_gpu_cache()
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # required_files = ["model.pth", "config.json", "vocab.json", "speakers_xtts.pth"]
        # files_in_dir = os.listdir(self.checkpoint_dir)
        # if not all(file in files_in_dir for file in required_files):
        #     print(f"Thiếu file mô hình! Đang tải...")
        #     snapshot_download(repo_id=self.repo_id, repo_type="model", local_dir=self.checkpoint_dir)
        #     print("Tải xong mô hình.")

        xtts_config = os.path.join(self.checkpoint_dir, "config.json")
        config = TortoiseConfig()
        config.load_json(xtts_config)
        model = Tortoise.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=self.checkpoint_dir, use_deepspeed=self.use_deepspeed, eval=True)

        if torch.cuda.is_available():
            model.cuda()

        return model

    def _extract_latents(self, speaker_audio_file):
        reference_clips = [audio_utils.load_audio(speaker_audio_file, 24000)]
        print(reference_clips)
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            voice_samples=reference_clips
        )

        return gpt_cond_latent, speaker_embedding

    def _normalize_vietnamese_text(self, text):
        text = text.lower()
        text = text.replace("%,", "% ,")
        return (
            TTSnorm(text, unknown=False, lower=False, rule=True)
            .replace("..", ".")
            .replace("?.", "?")
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace('"', "")
            .replace("'", "")
            .replace("!", "")
            .replace("ai", "Ây Ai")
            .replace("a.i", "Ây Ai")
            .replace("kol", "cây âu eo")
            .replace("cerave", "xê ra vi")
            .replace("₫", "đồng")
        )

    def _calculate_keep_len(self, text):
        word_count = len(text.split())
        num_punct = text.count(".") + text.count("!") + text.count("?") + text.count(",")
        if word_count < 5:
            return 15000 * word_count + 2000 * num_punct
        elif word_count < 10:
            return 13000 * word_count + 2000 * num_punct
        return -1

    def _clear_gpu_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _convert_wav_to_mp3(self, wav_file_path):
        mp3_file_path = wav_file_path.replace(".wav", ".mp3")

        ffmpeg.input(wav_file_path).output(
            mp3_file_path,
            ar=44100,  # Sampling rate 44.1 kHz
            ac=1,
            ab="128k",
            format="mp3",
            acodec="libmp3lame",
            strict='normal'
        ).run()

        # mp3_file_path = wav_file_path.replace(".wav", ".mp3")
        # audio = AudioSegment.from_wav(wav_file_path)
        # audio.export(mp3_file_path, format="mp3", bitrate="128k")

        delete_file = Path(wav_file_path)
        if delete_file.exists():
            delete_file.unlink()

        return mp3_file_path

    def _adjust_number(self, num):
        if num is None:
            return 1.08
        elif num < 0.8:
            return 0.8
        elif num > 1.5:
            return 1.5
        return num

