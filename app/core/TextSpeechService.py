import os
from datetime import datetime
from pathlib import Path

import torch
import torchaudio
import subprocess

from df.enhance import enhance, init_df, load_audio, save_audio

from huggingface_hub import snapshot_download
from TTS.TTS.tts.configs.xtts_config import XttsConfig
from TTS.TTS.tts.models.xtts import Xtts
from underthesea import sent_tokenize
from vinorm import TTSnorm
from pydub import AudioSegment
from df import enhance, init_df
import ffmpeg

conditioning_latents_cache = {}

class TextToSpeechService:
    def __init__(self, checkpoint_dir="model/", repo_id="capleaf/viXTTS", use_deepspeed=False):
        self.checkpoint_dir = checkpoint_dir
        self.repo_id = repo_id
        self.use_deepspeed = use_deepspeed
        current_path = Path(os.path.dirname(os.path.abspath(__file__)))
        project_dir = current_path.parent.parent
        self.output_dir = os.path.join(project_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    def text_to_speech(self, text, speaker_audio_file):
        if not speaker_audio_file:
            return "Bạn cần cung cấp tệp âm thanh tham chiếu!", None

        # Chuẩn hóa văn bản
        text = self._normalize_vietnamese_text(text)

        gpt_cond_latent, speaker_embedding = self._extract_latents(speaker_audio_file)

        sentences = sent_tokenize(text)
        wav_chunks = []
        for sentence in sentences:
            if sentence.strip() == "":
                continue
            wav_chunk = self.model.inference(
                text=sentence,
                language="vi",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.3,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=30,
                top_p=0.85,
                enable_text_splitting=True,
            )

            keep_len = self._calculate_keep_len(sentence)
            wav_chunk["wav"] = wav_chunk["wav"][:keep_len]
            wav_chunks.append(torch.tensor(wav_chunk["wav"]))

        # Kết hợp tất cả các đoạn âm thanh lại
        out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_output_path = os.path.join(self.output_dir, f"output_{timestamp}.wav")
        torchaudio.save(wav_output_path, out_wav, 24000)
        print(f"Đã lưu âm thanh vào {wav_output_path}")
        return wav_output_path

    def convert_mp3_wav(self, mp3_file_path, wav_file_path):
        audio = AudioSegment.from_mp3(mp3_file_path)
        audio.export(wav_file_path, format="wav")


    def process_audio(self, filename):
        subprocess.run(
            [
                "deepFilter",
                filename,
                "-o",
                os.path.dirname(filename),
            ]
        )
        wav_path = os.path.splitext(filename)[0] + '.wav'

        ffmpeg.input(filename).output(
            wav_path,
            ar=24000,
            ac=1,
            ab="384K",
            format="wav",
            acodec="pcm_s16le",
            map_metadata = "-1"
        ).run()

        return wav_path


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

    def _load_model(self):
        self._clear_gpu_cache()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        required_files = ["model.pth", "config.json", "vocab.json"]
        files_in_dir = os.listdir(self.checkpoint_dir)
        if not all(file in files_in_dir for file in required_files):
            print(f"Thiếu file mô hình! Đang tải...")
            snapshot_download(repo_id=self.repo_id, repo_type="model", local_dir=self.checkpoint_dir)
            print("Tải xong mô hình.")

        # Tải cấu hình và khởi tạo mô hình
        xtts_config = os.path.join(self.checkpoint_dir, "config.json")
        config = XttsConfig()
        config.load_json(xtts_config)
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=self.checkpoint_dir, use_deepspeed=self.use_deepspeed)

        if torch.cuda.is_available():
            print(torch.cuda.is_available())
            model.cuda()
        return model

    def _normalize_vietnamese_text(self, text):
        text = (
            TTSnorm(text, unknown=False, lower=False, rule=True)
            .replace("..", ".")
            .replace("!.", "!")
            .replace("?.", "?")
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace('"', "")
            .replace("'", "")
            .replace("AI", "Ây Ai")
            .replace("A.I", "Ây Ai")
            .replace("KOL", "Cây âu eo")
            .replace("kol", "Cây âu eo")
        )
        return text


    def _extract_latents(self, speaker_audio_file):
        global conditioning_latents_cache

        if not hasattr(self, 'model'):
            self.model = self._load_model()

        cache_key = (
            speaker_audio_file,
            self.model.config.gpt_cond_len,
            self.model.config.max_ref_len,
            self.model.config.sound_norm_refs,
        )

        if cache_key in conditioning_latents_cache:
            gpt_cond_latent, speaker_embedding = conditioning_latents_cache[cache_key]
        else:
            print("Computing conditioning latents...")
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=speaker_audio_file,
                gpt_cond_len=self.model.config.gpt_cond_len,
                max_ref_length=self.model.config.max_ref_len,
                sound_norm_refs=self.model.config.sound_norm_refs,
            )
            conditioning_latents_cache[cache_key] = (gpt_cond_latent, speaker_embedding)

        return gpt_cond_latent, speaker_embedding

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

        mp3_file_path = wav_file_path.replace(".wav", ".mp3")
        audio = AudioSegment.from_wav(wav_file_path)
        audio.export(mp3_file_path, format="mp3", bitrate="128k", parameters=["-ar", "44100", "-ac", "1"])

        delete_file = Path(wav_file_path)
        if delete_file.exists():
            delete_file.unlink()

        return mp3_file_path