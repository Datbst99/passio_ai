import os

from flask import request, send_file, jsonify
from ..core.UploadFileService import upload_voice, get_voice_path
from ..core.XTTSService import xtts_to_speech

class ApiController:

    @staticmethod
    def text_to_speech():
        data = request.json
        text = data.get('text')
        voice_key = data.get('voice_key')
        speed = data.get('speed')

        if not text or text.strip() == "":
            return {"error": "Text is required and cannot be empty"}, 400

        audio_file = get_voice_path(voice_key)

        if not text or not audio_file:
            return {"error": "Invalid voice_key"}, 400

        output_path = xtts_to_speech(text, audio_file, speed)

        if not os.path.exists(output_path):
            return {"error": "Failed to generate audio"}, 500

        output_path = os.path.abspath(output_path)

        return send_file(output_path, mimetype='audio/mpeg')

    @staticmethod
    def upload_audio():
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        name = request.form.get('name')

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and file.filename.endswith('.mp3'):
            voice_key = upload_voice(file, name)
            return jsonify({"message": "File converted successfully", "voice_key": voice_key}), 200
        else:
            return jsonify({"error": "Invalid file type. Please upload an MP3 file."}), 400

