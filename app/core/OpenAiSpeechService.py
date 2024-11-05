import cv2
import numpy as np
import mediapipe as mp
import soundfile as sf
from moviepy.editor import *

from app.core.TextSpeechService import TextToSpeechService

# Khởi tạo MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def detect_face(image):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        results = face_detection.process(image)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                        int(bboxC.width * iw), int(bboxC.height * ih))
                mp_drawing.draw_detection(image, detection)
                return bbox
    return None

def detect_hands(image):
    with mp_hands.Hands(min_detection_confidence=0.5) as hands:
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                return hand_landmarks
    return None

def text_to_speech(text):
    tts = TextToSpeechService()
    audio_file = "model/samples/nu-luu-loat.wav"
    output_path = tts.text_to_speech(text, audio_file)
    print(output_path)
    return output_path

def create_video(image_path, audio_path, output_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Chuyển đổi kích thước để có tỷ lệ 16:9
    target_width = 1280
    target_height = 720
    image_resized = cv2.resize(image, (target_width, target_height))

    # Lặp qua từng khung hình
    frames = []
    for _ in range(30):  # Giả sử video dài 1 giây với 30 FPS
        # Phát hiện tay và vẽ lên hình ảnh
        detect_hands(image_resized)
        frames.append(image_resized)

    # Tạo video từ khung hình
    clip = ImageSequenceClip(frames, fps=30)
    clip = clip.set_audio(AudioFileClip(audio_path))
    clip.write_videofile(output_path, codec='libx264', fps=30)

# Sử dụng hàm
text = "Xin chào! Đây là một ví dụ về chuyển đổi văn bản thành giọng nói và video."
# image_path = 'images/img_2.jpg'  # Thay bằng đường dẫn tới ảnh của bạn
image_path = r'C:\Users\Admin\Desktop\Python\images\img_2.jpg'  # Sử dụng đường dẫn tuyệt đối
audio = r'C:\Users\Admin\Desktop\Python\app\core\output\output_20241030_000246.wav'
out = r'C:\Users\Admin\Desktop\Python\video\output_video.mp4'
# audio_path = text_to_speech(text)
create_video(image_path, audio, out)
