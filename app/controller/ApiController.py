import os

from flask import request, send_file, jsonify
from ..core.TextToSpeechService import TextToSpeechService
from config.voice import Voice

tts = TextToSpeechService()
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class ApiController:

    @staticmethod
    def text_to_speech():
        data = request.json
        text = data.get('text')
        voice_key = data.get('voice_key')
        speed = data.get('speed')

        if not text or text.strip() == "":
            return {"error": "Text is required and cannot be empty"}, 400

        audio_file = next((item['audio_file'] for item in Voice.SAMPLE if item['key'] == voice_key), None)

        if not text or not audio_file:
            return {"error": "Missing text or invalid voice_key"}, 400

        output_path = tts.text_to_speech(text, audio_file, speed)

        if not os.path.exists(output_path):
            return {"error": "Failed to generate audio"}, 500

        return send_file(output_path, mimetype='audio/mpeg')

    @staticmethod
    def upload_audio():
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and file.filename.endswith('.mp3'):

            mp3_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(mp3_path)
            wav_path = tts.predict_speaker(mp3_path)
            voice_key = Voice.add_sample(wav_path)

            return jsonify({"message": "File converted successfully", "voice_key": voice_key}), 200
        else:
            return jsonify({"error": "Invalid file type. Please upload an MP3 file."}), 400

