import os
from datetime import datetime

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


conditioning_latents_cache = {}

class TextToSpeechService:
    def __init__(self, checkpoint_dir="model/", repo_id="capleaf/viXTTS", use_deepspeed=False):
        self.checkpoint_dir = checkpoint_dir
        self.repo_id = repo_id
        self.use_deepspeed = use_deepspeed
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def text_to_speech(self, text, speaker_audio_file):
        if not speaker_audio_file:
            return "Bạn cần cung cấp tệp âm thanh tham chiếu!", None

        # Chuẩn hóa văn bản
        text = self._normalize_vietnamese_text(text)

        audio = AudioSegment.from_file(speaker_audio_file)
        print(audio.frame_rate)

        # Trích xuất gpt_cond_latent và speaker_embedding từ âm thanh tham chiếu
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
            wav_chunks.append(torch.tensor(wav_chunk["wav"]))


            if wav_chunks:
                # Kết hợp tất cả các đoạn âm thanh lại
                # out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
                out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
                # resampler = torchaudio.transforms.Resample(orig_freq=24000 , new_freq=44100)
                # out_wav = resampler(out_wav)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                wav_output_path = os.path.join(self.output_dir, f"output_{timestamp}.wav")
                torchaudio.save(wav_output_path, out_wav, 24000)
                print(f"Đã lưu âm thanh vào {wav_output_path}")
                return self._convert_wav_to_mp3(wav_output_path)

            return "Không có âm thanh được tạo ra!", None

    def convert_mp3_wav(self, mp3_file_path, wav_file_path):
        audio = AudioSegment.from_mp3(mp3_file_path)
        audio.export(wav_file_path, format="wav")

    def upload_and_process_audio(self, filepath, output_dir="/uploads", denoise=True, sample_rate=24000):
        """
        Processes an audio file, applies denoising if specified, converts to WAV format, and resamples it.

        Parameters:
            filepath (str): Path to the input audio file.
            output_dir (str): Directory to save the processed audio file.
            denoise (bool): Whether to apply denoising using DeepFilterNet. Default is True.
            sample_rate (int): Target sample rate for the output WAV file. Default is 22050 Hz.

        Returns:
            str: Path to the processed audio file.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # filename = os.path.basename(filepath)
        output_path = os.path.join(output_dir, "user_sample.wav")

        try:
            # Apply denoising if specified
            if denoise:
                subprocess.run(["deepFilter", filepath], check=True)

                processed_filename = filepath.replace('.wav', '_DeepFilterNet3.wav')
            else:
                processed_filename = filepath

            # Convert and resample the audio
            subprocess.run([
                "ffmpeg", "-i", processed_filename, "-ac", "1",
                "-ar", str(sample_rate), "-vn", output_path,
                "-y", "-hide_banner", "-loglevel", "error"
            ], check=True)

            # Remove the intermediate processed file if denoise was applied
            if denoise:
                os.remove(processed_filename)
            else:
                os.remove(filepath)

            print("> Đã tải và xử lý file âm thanh thành công")
            return output_path

        except subprocess.CalledProcessError as e:
            print("Error processing audio:", e)
            return None

    def process_audio(self, filename, denoise=False):

        if denoise:
            # Run DeepFilterNet denoising process
            model, df_state, _ = init_df()
            audio, _ = load_audio(filename, sr=df_state.sr())
            enhanced = enhance(model, df_state, audio)
            denoised_filename = filename.replace('.wav', '_DeepFilterNet3.wav')
            save_audio(denoised_filename, enhanced, df_state.sr())
            # result = subprocess.run(['deepFilter', filename], check=True)
            # print(filename, result)
            # Convert denoised file to mono and resample to 22050 Hz

            subprocess.run(['ffmpeg', '-i', denoised_filename, '-ac', '1', '-ar', '24000', '-vn', filename, '-y', '-hide_banner',
                 '-loglevel', 'error'])
            print(denoised_filename)
            # Remove intermediate denoised file
            os.remove(denoised_filename)
        else:
            # Convert original file to mono and resample to 22050 Hz
            subprocess.run(['ffmpeg', '-i', filename, '-ac', '1', '-ar', '24000', '-vn', filename, '-y', '-hide_banner',
                            '-loglevel', 'error'])

    def _clear_gpu_cache(self):
        print(torch.cuda.is_available())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self):
        self._clear_gpu_cache()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        required_files = ["model.pth", "config.json", "vocab.json", "speakers_xtts.pth"]
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
            model.cuda()

        print("Mô hình đã được tải.")
        return model

    def _normalize_vietnamese_text(self, text):
        text = (
            TTSnorm(text, unknown=False, lower=False, rule=True)
            .replace("..", ".")
            .replace("!.", "!")
            .replace("?.", "?")
            .replace(" .", ".")
            .replace(". ", ".")
            .replace(" ,", ",")
            .replace('"', "")
            .replace("'", "")
            .replace("AI", "Ây Ai")
            .replace("A.I", "Ây Ai")
            .replace("KOL", "Cây âu eo")
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

        # gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
        #     audio_path=speaker_audio_file,
        #     gpt_cond_len=self.model.config.gpt_cond_len,
        #     max_ref_length=self.model.config.max_ref_len,
        #     sound_norm_refs=self.model.config.sound_norm_refs,
        # )
        return gpt_cond_latent, speaker_embedding

    def _convert_wav_to_mp3(self, wav_file_path):
        mp3_file_path = wav_file_path.replace(".wav", ".mp3")
        audio = AudioSegment.from_wav(wav_file_path)
        audio.export(mp3_file_path, format="mp3", bitrate="128k")
        print(f"Đã chuyển đổi âm thanh sang {mp3_file_path}")
        return mp3_file_path