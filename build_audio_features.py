import os
import json
import subprocess
import shutil
import gdown
from faster_whisper import WhisperModel
import time

CACHE_FILE = "cache/asr.json"
TEMP_DIR = "temp_audio"

# Lấy các GDrive file ID ở đây (có thể lấy từ link chia sẻ)
# Ví dụ link: https://drive.google.com/file/d/1XyZ.../view
# ID là phần: 1XyZ...
GDRIVE_IDS = [
    # "1XyZ...",
]

def init_env():
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs("cache", exist_ok=True)

def extract_audio(video_path, audio_path):
    cmd = [
        "ffmpeg", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
        audio_path, "-y", "-loglevel", "error"
    ]
    subprocess.run(cmd)

def main():
    init_env()
    
    # 1. Load model (Chạy CPU hoặc GPU int8 rất nhẹ)
    print("⏳ Đang tải model Faster-Whisper (lần đầu sẽ tốn chút thời gian tải model)...")
    model = WhisperModel("small", device="cuda", compute_type="int8")
    
    # Load cache cũ nếu có
    asr_cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            asr_cache = json.load(f)

    for file_id in GDRIVE_IDS:
        print(f"\n📥 Đang tải file từ Google Drive: {file_id}")
        video_path = os.path.join(TEMP_DIR, f"{file_id}.mp4")
        audio_path = os.path.join(TEMP_DIR, f"{file_id}.wav")
        
        # Tải từ GDrive
        gdown.download(id=file_id, output=video_path, quiet=False)
        
        if not os.path.exists(video_path):
            print(f"❌ Tải thất bại: {file_id}")
            continue
            
        print(f"🎵 Đang trích xuất âm thanh...")
        extract_audio(video_path, audio_path)
        
        # XÓA VIDEO CHO NHẸ Ổ CỨNG
        os.remove(video_path)
        
        print(f"🧠 Đang nhận diện giọng nói (ASR)...")
        segments, info = model.transcribe(audio_path, beam_size=5, language="vi")
        
        full_text = []
        for segment in segments:
            full_text.append(f"[{segment.start:.1f}s-{segment.end:.1f}s]: {segment.text}")
            
        # Tạm lưu bằng file ID (bạn có thể đổi tên sau)
        asr_cache[file_id] = "\n".join(full_text)
        
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(asr_cache, f, ensure_ascii=False, indent=2)
            
        # Xóa file audio
        os.remove(audio_path)
        print(f"✅ Hoàn thành file {file_id}! Đã lưu vào {CACHE_FILE}")

    # Dọn rác
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("\n🎉 XONG TOÀN BỘ!")

if __name__ == "__main__":
    main()
