class Voice:
    SAMPLE = [
        {
            "key": "eco_voice_07",
            "voice": "Giọng 1",
            "audio_file": "model/samples/nam-calm.wav"
        },
        {
            "key": "eco_voice_08",
            "voice": "Giọng 1",
            "audio_file": "model/samples/nam-cham.wav"
        },
        {
            "key": "eco_voice_09",
            "voice": "Giọng 1",
            "audio_file": "model/samples/nam-nhanh.wav"
        },
        {
            "key": "eco_voice_10",
            "voice": "Giọng 1",
            "audio_file": "model/samples/nam-truyen-cam.wav"
        },
        {
            "key": "eco_voice_11",
            "voice": "Giọng 1",
            "audio_file": "model/samples/nu-calm.wav"
        },
        {
            "key": "eco_voice_12",
            "voice": "Giọng 1",
            "audio_file": "model/samples/nu-cham.wav"
        },
        {
            "key": "eco_voice_13",
            "voice": "Giọng 1",
            "audio_file": "model/samples/nu-luu-loat.wav"
        },
        {
            "key": "eco_voice_14",
            "voice": "Giọng 1",
            "audio_file": "model/samples/nu-nhan-nha.wav"
        },
        {
            "key": "eco_voice_15",
            "voice": "Giọng 1",
            "audio_file": "model/samples/nu-nhe-nhang.wav"
        },
        {
            "key": "eco_voice_16",
            "voice": "Giọng 1",
            "audio_file": "uploads/host6.wav"
        },
        {
            "key": "eco_voice_17",
            "voice": "Giọng 1",
            "audio_file": "uploads/host11_v2.wav"
        },
        {
            "key": "eco_voice_18",
            "voice": "Giọng 1",
            "audio_file": "uploads/host21.wav"
        },
        {
            "key": "eco_voice_19",
            "voice": "Giọng 1",
            "audio_file": "uploads/chiLydalie.wav"
        },
        {
            "key": "eco_voice_20",
            "voice": "Giọng 1",
            "audio_file": "uploads/ChiLydia.wav"
        },
        {
            "key": "eco_voice_21",
            "voice": "Giọng 1",
            "audio_file": "uploads/host15.wav"
        },
        {
            "key": "eco_voice_22",
            "voice": "Giọng 1",
            "audio_file": "uploads/host_q_eco.wav"
        },
    ]

    @classmethod
    def add_sample(cls, audio_file):
        max_key = max(
            int(item["key"].split("_")[-1]) for item in cls.SAMPLE
        )
        # Tạo key mới
        new_key = f"eco_voice_{max_key + 1}"
        # Thêm mục mới
        cls.SAMPLE = cls.SAMPLE + [{
            "key": new_key,
            "voice": "voice 1",
            "audio_file": audio_file
        }]

        return new_key
