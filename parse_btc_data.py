"""
parse_btc_data.py
==================
BƯỚC 0 của pipeline mới: Đọc + parse dữ liệu gốc từ BTC.

Cấu trúc thực tế sau giải nén:
  data/btc/map_keyframes/map-keyframes/  ← 873 file CSV (L21_V001.csv, ...)
    Columns: n, pts_time, fps, frame_idx
  data/btc/media_info/media-info/        ← 873 file JSON (L21_V001.json, ...)
    Keys: author, title, description, keywords, watch_url, ...
  data/btc/clip_features/clip-features-32/ ← 873 file .npy (L21_V001.npy, ...)
    Shape: (K, 512), dtype: float16

Script sẽ:
  1) Parse keyframe map (CSV) → cache/btc_keyframe_meta.json
  2) Parse media metadata (JSON) → cache/btc_media_info.json
  3) Build Image FAISS index từ CLIP features .npy (dim=512, ViT-B/32)
     → index/btc_image_index.faiss  +  index/btc_image_id_map.json

Cách chạy:
    python parse_btc_data.py
"""

import os, json, csv, glob
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BTC_DIR     = os.path.join(BASE_DIR, "data", "btc")
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
INDEX_DIR   = os.path.join(BASE_DIR, "index")

# Đường dẫn thực tế sau khi giải nén (nested subfolder)
MAP_KF_DIR  = os.path.join(BTC_DIR, "map_keyframes", "map-keyframes")
MEDIA_DIR   = os.path.join(BTC_DIR, "media_info",    "media-info")
CLIP_DIR    = os.path.join(BTC_DIR, "clip_features",  "clip-features-32")

BTC_KF_META_CACHE    = os.path.join(CACHE_DIR, "btc_keyframe_meta.json")
BTC_MEDIA_INFO_CACHE = os.path.join(CACHE_DIR, "btc_media_info.json")
BTC_IMAGE_INDEX      = os.path.join(INDEX_DIR, "btc_image_index.faiss")
BTC_IMAGE_ID_MAP     = os.path.join(INDEX_DIR, "btc_image_id_map.json")


# =================================================================
# 1) PARSE KEYFRAME MAP (CSV format thực tế: n, pts_time, fps, frame_idx)
# =================================================================
def parse_keyframe_map() -> dict:
    """
    Đọc 873 file CSV trong map_keyframes/map-keyframes/.
    Mỗi file = 1 video, tên file = video_id (e.g. L21_V001.csv).
    Columns: n (số thứ tự keyframe), pts_time (giây), fps, frame_idx (frame số thực trong video).

    Trả về: {video_id: [{frame_idx, timestamp_sec}, ...]}
    """
    if not os.path.exists(MAP_KF_DIR):
        print(f"[parse_keyframe_map] ❌ Không tìm thấy {MAP_KF_DIR}")
        return {}

    csv_files = sorted(glob.glob(os.path.join(MAP_KF_DIR, "*.csv")))
    print(f"[parse_keyframe_map] Tìm thấy {len(csv_files)} file CSV")

    result = {}
    for fpath in tqdm(csv_files, desc="Parse CSV keyframe maps"):
        video_id = Path(fpath).stem  # L21_V001
        try:
            with open(fpath, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                frames = []
                for row in reader:
                    # frame_idx là cột số thực trong video (quan trọng nhất cho BTC)
                    frame_idx = int(row.get("frame_idx") or row.get("n") or 0)
                    timestamp = float(row.get("pts_time") or 0.0)
                    frames.append({
                        "video_id":      video_id,
                        "frame_idx":     frame_idx,
                        "timestamp_sec": round(timestamp, 3),
                    })
                result[video_id] = frames
        except Exception as e:
            print(f"  [WARN] Lỗi đọc {fpath}: {e}")

    total_kf = sum(len(v) for v in result.values())
    print(f"[parse_keyframe_map] ✅ {len(result)} video, {total_kf} keyframe tổng")
    return result


# =================================================================
# 2) PARSE MEDIA INFO (YouTube metadata)
# =================================================================
def parse_media_info() -> dict:
    """
    Đọc 873 file JSON trong media_info/media-info/.
    Keys thực tế: author, channel_id, channel_url, description, keywords,
                  length, publish_date, thumbnail_url, title, watch_url.

    Trả về: {video_id: {title, description, keywords, author, watch_url}}
    """
    if not os.path.exists(MEDIA_DIR):
        print(f"[parse_media_info] ❌ Không tìm thấy {MEDIA_DIR}")
        return {}

    json_files = sorted(glob.glob(os.path.join(MEDIA_DIR, "*.json")))
    print(f"[parse_media_info] Tìm thấy {len(json_files)} file JSON")

    result = {}
    for fpath in tqdm(json_files, desc="Parse media info JSON"):
        video_id = Path(fpath).stem
        try:
            with open(fpath, encoding="utf-8") as f:
                meta = json.load(f)
            result[video_id] = {
                "title":       meta.get("title", ""),
                "description": (meta.get("description") or "")[:500],
                "keywords":    meta.get("keywords", []),
                "author":      meta.get("author", ""),
                "watch_url":   meta.get("watch_url", ""),
                "length_sec":  meta.get("length", 0),
            }
        except Exception as e:
            print(f"  [WARN] Lỗi đọc {fpath}: {e}")

    print(f"[parse_media_info] ✅ {len(result)} video có metadata")
    return result


# =================================================================
# 3) BUILD IMAGE FAISS INDEX TỪ CLIP FEATURES BTC
# =================================================================
def build_btc_image_index(keyframe_map: dict):
    """
    Load 873 file .npy (mỗi file shape (K, 512) dtype float16) →
    stack lại → normalize L2 → build FAISS IndexFlatIP (dim=512).

    Thứ tự ghép: sorted theo video_id → keyframe_map[video_id] tương ứng.
    """
    if os.path.exists(BTC_IMAGE_INDEX) and os.path.exists(BTC_IMAGE_ID_MAP):
        print("[BTC CLIP] Index đã tồn tại → bỏ qua.")
        print("  (Xóa index/btc_image_index.faiss để rebuild)")
        return

    if not os.path.exists(CLIP_DIR):
        print(f"[BTC CLIP] ❌ Không tìm thấy {CLIP_DIR}")
        return

    npy_files = sorted(glob.glob(os.path.join(CLIP_DIR, "*.npy")))
    print(f"[BTC CLIP] Tìm thấy {len(npy_files)} file .npy")

    all_vecs = []
    all_meta = []
    skipped  = 0

    for fpath in tqdm(npy_files, desc="Load CLIP features"):
        video_id = Path(fpath).stem

        if video_id not in keyframe_map:
            skipped += 1
            continue

        frames = keyframe_map[video_id]
        vecs = np.load(fpath).astype("float32")   # (K, 512) float16 → float32

        if vecs.ndim != 2 or vecs.shape[1] != 512:
            print(f"  [WARN] {video_id}: shape={vecs.shape} bất thường, bỏ qua")
            skipped += 1
            continue

        # Khớp số lượng (một vài video có thể lệch 1-2 frame)
        if len(frames) != len(vecs):
            min_len = min(len(frames), len(vecs))
            print(f"  [WARN] {video_id}: {len(frames)} frames ≠ {len(vecs)} vecs "
                  f"→ dùng {min_len}")
            frames = frames[:min_len]
            vecs   = vecs[:min_len]

        all_vecs.append(vecs)
        all_meta.extend(frames)

    if not all_vecs:
        print("[BTC CLIP] ❌ Không có vector nào hợp lệ")
        return

    print(f"[BTC CLIP] Stack {len(all_meta)} vectors... (bỏ qua {skipped} video)")
    vectors = np.vstack(all_vecs)          # (N_total, 512)
    faiss.normalize_L2(vectors)            # cosine sim qua inner product

    index = faiss.IndexFlatIP(512)
    index.add(vectors)

    faiss.write_index(index, BTC_IMAGE_INDEX)
    with open(BTC_IMAGE_ID_MAP, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False)

    print(f"[BTC CLIP] ✅ Build xong: {len(all_meta)} vectors, dim=512")
    print(f"  → {BTC_IMAGE_INDEX}  ({os.path.getsize(BTC_IMAGE_INDEX) // (1024*1024)} MB)")
    print(f"  → {BTC_IMAGE_ID_MAP}")


# =================================================================
# MAIN
# =================================================================
if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    # 1) Parse keyframe map
    print("\n" + "="*60)
    print("BƯỚC 1: PARSE KEYFRAME MAP")
    print("="*60)
    kf_map = parse_keyframe_map()
    if kf_map:
        sample_vid = sorted(kf_map.keys())[0]
        print(f"  Mẫu [{sample_vid}]: {kf_map[sample_vid][:2]}")
        with open(BTC_KF_META_CACHE, "w", encoding="utf-8") as f:
            json.dump(kf_map, f, ensure_ascii=False)
        print(f"  ✅ Lưu → {BTC_KF_META_CACHE}")

    # 2) Parse media info
    print("\n" + "="*60)
    print("BƯỚC 2: PARSE MEDIA INFO")
    print("="*60)
    media_info = parse_media_info()
    if media_info:
        sample_vid = sorted(media_info.keys())[0]
        print(f"  Mẫu [{sample_vid}]: title='{media_info[sample_vid]['title'][:60]}'")
        with open(BTC_MEDIA_INFO_CACHE, "w", encoding="utf-8") as f:
            json.dump(media_info, f, ensure_ascii=False)
        print(f"  ✅ Lưu → {BTC_MEDIA_INFO_CACHE}")

    # 3) Build image FAISS index
    print("\n" + "="*60)
    print("BƯỚC 3: BUILD IMAGE FAISS INDEX (CLIP ViT-B/32, dim=512)")
    print("="*60)
    build_btc_image_index(kf_map)

    print("\n✅ Xong! Bước tiếp theo: chạy build_btc_text_index.py")
