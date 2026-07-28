"""
extract_keyframes.py
======================
BƯỚC 1 của pipeline. Với mỗi video trong data/videos/, thực hiện:

  1a) TÁCH CẢNH bằng PySceneDetect -> mỗi cảnh lấy 1 frame ở giữa làm "keyframe"
      đại diện, lưu ảnh ra data/keyframes/<video_id>/<frame_idx>.jpg

  1b) TRÍCH TRANSCRIPT (lời thoại) bằng faster-whisper (model đa ngôn ngữ,
      tự nhận diện video đang nói tiếng Việt hay tiếng Anh) -> lưu ra
      cache/transcripts.json kèm timestamp từng câu.

Cả 2 bước đều CACHE theo video_id: nếu video đã xử lý rồi (đã có trong file
cache) thì bỏ qua, không chạy lại -> chạy lại script này nhiều lần khi debug
(vd thêm video mới) sẽ không mất thời gian xử lý lại video cũ.

Cách chạy:
    python extract_keyframes.py
"""

import os
import json
import cv2
from tqdm import tqdm
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from faster_whisper import WhisperModel

import config


# =================================================================
# TIỆN ÍCH ĐỌC / GHI CACHE JSON (load dở dang -> ghi tiếp, không mất dữ liệu cũ)
# =================================================================
def load_cache(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_video_list():
    """Lấy danh sách tên file video có trong thư mục data/videos/."""
    exts = (".mp4", ".avi", ".mkv", ".mov", ".webm")
    videos = [f for f in os.listdir(config.VIDEO_DIR) if f.lower().endswith(exts)]
    return sorted(videos)


# =================================================================
# 1a) TÁCH CẢNH BẰNG PYSCENEDETECT
# =================================================================
def detect_scenes(video_path):
    """
    Dùng PySceneDetect (ContentDetector - phát hiện đổi cảnh dựa trên độ lệch màu sắc/độ sáng
    giữa các frame liên tiếp) để tìm ranh giới các cảnh trong video.

    Trả về: list các tuple (start_frame, end_frame).
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=config.SCENE_THRESHOLD, min_scene_len=config.MIN_SCENE_LEN)
    )
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if len(scene_list) > 0:
        return [(start.get_frames(), end.get_frames()) for start, end in scene_list]

    # Video quá ngắn / không đổi cảnh rõ rệt (vd: 1 shot tĩnh) -> coi cả video là 1 scene duy nhất.
    # Lấy tổng số frame qua OpenCV để không phụ thuộc backend nội bộ của PySceneDetect.
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return [(0, max(total_frames, 1))]


def extract_keyframe_images(video_path, video_id, scenes):
    """
    Với mỗi scene (start_frame, end_frame), lấy NHIỀU frame tại các vị trí
    cố định (mặc định: 25%, 50%, 75% độ dài scene) thay vì chỉ 1 frame giữa.

    Lý do: 1 frame/scene bỏ sót nhiều thông tin (đặc biệt scene dài > 5s).
    Lấy 3 frame/scene tăng recall ~3x mà không cần thay đổi model hay pipeline.

    Scene ngắn (< 2 * MIN_SCENE_LEN frames) chỉ lấy 1 frame giữa để tránh
    các frame bị trùng hoặc quá gần nhau.

    Lưu ảnh ra: data/keyframes/<video_id>/<frame_idx:08d>.jpg
    Trả về: list metadata dict {video_id, frame_idx, timestamp_sec, path}
    """
    # Vị trí lấy frame trong mỗi scene (theo tỷ lệ % độ dài scene).
    # Đổi về [0.5] nếu muốn quay lại chế độ 1 frame/scene (ít VRAM hơn).
    SAMPLE_POSITIONS = getattr(config, "KEYFRAME_POSITIONS", [0.25, 0.5, 0.75])
    MIN_SCENE_FRAMES_FOR_MULTI = 2 * config.MIN_SCENE_LEN  # scene quá ngắn → chỉ lấy 1 frame

    out_dir = os.path.join(config.KEYFRAME_DIR, video_id)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    metas = []
    seen_frames = set()  # tránh lưu trùng frame nếu scene quá ngắn

    for start_f, end_f in scenes:
        length = end_f - start_f

        # Scene quá ngắn → chỉ lấy 1 frame giữa
        positions = SAMPLE_POSITIONS if length >= MIN_SCENE_FRAMES_FOR_MULTI else [0.5]

        for pos in positions:
            target_frame = start_f + int(length * pos)
            if target_frame in seen_frames:
                continue
            seen_frames.add(target_frame)

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ok, frame = cap.read()
            if not ok:
                continue

            frame_path = os.path.join(out_dir, f"{target_frame:08d}.jpg")
            cv2.imwrite(frame_path, frame)
            metas.append({
                "video_id": video_id,
                "frame_idx": target_frame,
                "timestamp_sec": round(target_frame / fps, 2),
                "path": frame_path,
            })

    cap.release()
    return metas



def run_keyframe_extraction():
    """Chạy tách cảnh + lưu keyframe cho TẤT CẢ video, có cache theo video_id."""
    keyframe_meta = load_cache(config.KEYFRAME_META_CACHE)  # {video_id: [meta, ...]}
    videos = get_video_list()

    if not videos:
        print(f"[CẢNH BÁO] Không tìm thấy video nào trong {config.VIDEO_DIR}. "
              f"Hãy copy video (.mp4/.avi/.mkv/.mov/.webm) vào thư mục này trước.")
        return keyframe_meta

    for video_file in tqdm(videos, desc="[1a] Trích xuất keyframes"):
        video_id = os.path.splitext(video_file)[0]
        if video_id in keyframe_meta:
            continue  # ĐÃ XỬ LÝ RỒI -> bỏ qua (đây chính là cơ chế cache)

        video_path = os.path.join(config.VIDEO_DIR, video_file)
        try:
            scenes = detect_scenes(video_path)
            metas = extract_keyframe_images(video_path, video_id, scenes)
            keyframe_meta[video_id] = metas
            save_cache(config.KEYFRAME_META_CACHE, keyframe_meta)  # ghi ngay sau mỗi video
        except Exception as e:
            print(f"[LỖI] Không xử lý được video {video_file}: {e}")

    print(f"[1a] Hoàn tất: đã có keyframe cho {len(keyframe_meta)} video "
          f"(tổng {sum(len(v) for v in keyframe_meta.values())} keyframe).")
    return keyframe_meta


# =================================================================
# 1b) TRÍCH TRANSCRIPT BẰNG FASTER-WHISPER
# =================================================================
def run_transcript_extraction():
    """
    Dùng faster-whisper để lấy transcript (lời thoại) cho từng video.
    faster-whisper tự giải mã audio trực tiếp từ file video (không cần tách audio thủ công)
    và tự nhận diện ngôn ngữ đang nói (vi hoặc en) cho từng video.
    """
    transcript_cache = load_cache(config.TRANSCRIPT_CACHE)  # {video_id: {language, segments:[...]}}
    videos = get_video_list()
    if not videos:
        return transcript_cache

    # Chỉ tải model nếu thực sự còn video cần xử lý (tránh tải model chỉ để... không dùng)
    videos_to_process = [
        v for v in videos if os.path.splitext(v)[0] not in transcript_cache
    ]
    if not videos_to_process:
        print("[1b] Tất cả video đã có transcript trong cache -> bỏ qua.")
        return transcript_cache

    print(f"[1b] Đang tải Whisper model '{config.WHISPER_MODEL_SIZE}' "
          f"(lần đầu sẽ tải weight, hơi lâu)...")
    model = WhisperModel(
        config.WHISPER_MODEL_SIZE,
        device=config.DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )

    for video_file in tqdm(videos_to_process, desc="[1b] Trích xuất transcript"):
        video_id = os.path.splitext(video_file)[0]
        video_path = os.path.join(config.VIDEO_DIR, video_file)
        try:
            segments, info = model.transcribe(
                video_path,
                beam_size=5,
                vad_filter=True,  # Voice Activity Detection: cắt bỏ khoảng lặng -> transcript sạch hơn
            )
            seg_list = [
                {
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                }
                for seg in segments
                if seg.text.strip()
            ]
            transcript_cache[video_id] = {
                "language": info.language,  # 'vi' hoặc 'en' - Whisper tự nhận diện
                "segments": seg_list,
            }
            save_cache(config.TRANSCRIPT_CACHE, transcript_cache)  # ghi ngay sau mỗi video
        except Exception as e:
            print(f"[LỖI] Không lấy được transcript video {video_file}: {e}")

    print(f"[1b] Hoàn tất: đã có transcript cho {len(transcript_cache)} video.")
    return transcript_cache


if __name__ == "__main__":
    run_keyframe_extraction()
    run_transcript_extraction()
    print("\n✅ Xong bước 1. Tiếp theo chạy: python build_index.py")
