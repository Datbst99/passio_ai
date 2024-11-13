import json
import threading

import pika

from app.core.S3Service import S3Service
from app.core.TextSpeechService import TextToSpeechService
from config.core import Core
from config.voice import Voice

tts = TextToSpeechService()
s3  = S3Service()

class TextToSpeechCommand:
    def __init__(self):
        credentials = pika.PlainCredentials(Core.RABBITMQ_USER, Core.RABBITMQ_PASS)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=Core.RABBITMQ_HOST,
                port=Core.RABBITMQ_PORT,
                virtual_host=Core.RABBITMQ_VHOST,
                credentials=credentials
            )
        )
        self.channel = connection.channel()
        self.channel.queue_declare(queue='passio_backend', durable=True)

    def handle(self):
        self.channel.basic_consume(queue='passio_backend', on_message_callback=self._callback, auto_ack=True)
        self.channel.start_consuming()

    def _callback(self, ch, method, properties, body):
        try:
            response = json.loads(body.decode())
            voice_key = response['voice_key']
            text = response['text']

            if response['id'] or voice_key or text:
                print(f"Process id: {response['id']} \n")
                audio_sample = next((item['audio_file'] for item in Voice.SAMPLE if item['key'] == voice_key), None)
                output_path = tts.text_to_speech(text, audio_sample)
                url = s3.upload(output_path)
                data = {
                    'id' : response['id'],
                    'url' : url
                }
                self.channel.queue_declare(queue='passio_ai')
                self.channel.basic_publish(exchange='', routing_key='passio_ai', body=json.dumps(data))
        except Exception as e:
            print('Lỗi')
            print(e)

