import json
import os

import pika
from app.config import Config
from app.core.TextSpeechService import TextToSpeechService

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='passio_ai')
channel.queue_declare(queue='passio_backend')

tts = TextToSpeechService()

def process_message(body):
    # Xử lý nội dung tin nhắn ở đây

    try:
        response = body.decode()
        response = json.loads(response)
        voice_key = response.get("voice_key")
        text = response.get("text")
        if voice_key or text:
            audio_file = next((item['audio_file'] for item in Config.VOICE if item['key'] == voice_key), None)
            output_path = tts.text_to_speech(text, audio_file)
            if os.path.exists(output_path):
                with open(output_path, 'rb') as audio_file:
                    audio_data = audio_file.read()

                channel.basic_publish(exchange='', routing_key='passio_ai', body=audio_data)
                print("oke")
    except Exception as e:
        print(f"Lỗi: {e}")


def callback(ch, method, properties, body):
    process_message(body)

channel.basic_consume(queue='passio_backend', on_message_callback=callback, auto_ack=True)
print('Đang lắng nghe tin nhắn.')
channel.start_consuming()
