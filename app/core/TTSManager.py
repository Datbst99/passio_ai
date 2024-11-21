import threading

from app.core.TextToSpeechService import TextToSpeechService


class TTSManager:
    _instance = None
    _lock = threading.Lock()

    @staticmethod
    def get_tts_service():
        if TTSManager._instance is None:
            with TTSManager._lock:
                if TTSManager._instance is None:
                    TTSManager._instance = TextToSpeechService()
        return TTSManager._instance
