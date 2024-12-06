import os

import pandas as pd

# Đọc file parquet
df = pd.read_parquet('/home/datdv/Desktop/Python/passio_ai/data/train-00000-of-00354.parquet')

output_dir = "dataset/wavs"
os.makedirs(output_dir, exist_ok=True)

metadata = []


# Duyệt qua từng dòng dữ liệu
for idx, row in df.iterrows():
    # Lấy dữ liệu nhị phân và tên file từ dict
    audio_data = row['audio']
    if isinstance(audio_data, dict):
        audio_binary = audio_data.get('bytes')  # Lấy dữ liệu nhị phân
        file_name = audio_data.get('path')  # Lấy tên file
        if audio_binary is None or file_name is None:
            raise ValueError(f"Missing data at index {idx}")
    else:
        raise TypeError(f"Expected dict, but got {type(audio_data)} at index {idx}")

    # Lưu file âm thanh
    audio_path = os.path.join(output_dir, file_name)
    with open(audio_path, "wb") as f:
        f.write(audio_binary)

    # Thêm vào metadata
    transcription = row['text']
    audio_path = audio_path.replace('dataset/', '')
    metadata.append(f"{audio_path}|{transcription}")


# Lưu metadata thành file CSV
metadata_file = "dataset/metadata.csv"
with open(metadata_file, "w", encoding="utf-8") as f:
    f.write("\n".join(metadata))

print("Đã tạo xong dữ liệu:")
print(f"- File metadata: {metadata_file}")
print(f"- File âm thanh: {output_dir}")