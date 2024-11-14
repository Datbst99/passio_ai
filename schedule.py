import threading

from app.command.TextToSpeechCommand import TextToSpeechCommand

command = TextToSpeechCommand()
command.handle()

# def start_consumer():
#     command = TextToSpeechCommand()
#     command.handle()
#
# for _ in range(2):
#     threading.Thread(target=start_consumer).start()