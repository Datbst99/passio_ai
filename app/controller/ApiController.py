import hashlib
import os

from flask import request, send_file, jsonify
from ..core.UploadFileService import upload_voice, get_voice_path, get_voice_storage
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

        file_name = voice_key + text
        file_name = hashlib.md5(file_name.encode()).hexdigest()
        find_file_mp3 = file_name  + ".mp3"
        voice_storage = get_voice_storage(find_file_mp3)
        if voice_storage is not None:
            return send_file(voice_storage, mimetype='audio/mpeg')


        audio_file = get_voice_path(voice_key)

        if not text or not audio_file:
            return {"error": "Invalid voice_key"}, 400

        output_path = xtts_to_speech(text, file_name, audio_file, speed)

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

        if file and (file.filename.endswith('.mp3') or file.filename.endswith('.m4a')):
            voice_key = upload_voice(file, name)
            return jsonify({"message": "File converted successfully", "voice_key": voice_key}), 200
        else:
            return jsonify({"error": "Invalid file type. Please upload an MP3 file."}), 400

