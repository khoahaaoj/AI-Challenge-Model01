"""
build_btc_text_index.py
=========================
Build BGE-M3 dense text index và BM25 sparse index cho dữ liệu BTC.

Các nguồn text:
  1. Metadata YouTube (title, description, tags, author)
  2. Objects JSON (phát hiện bằng Faster R-CNN)
  (Captions/OCR/Transcript từ video cá nhân sẽ không được gộp vào đây
   để giữ sạch dữ liệu BTC).

Cách chạy:
    python build_btc_text_index.py
"""

import os, json, glob
import numpy as np
import faiss
import pickle
from tqdm import tqdm

import config
from tokenizer_utils import get_tokenizer

# =================================================================
# BẢNG DỊCH NHÃN TIẾNG ANH → TIẾNG VIỆT (Faster R-CNN / Google Vision)
# Giúp BM25 khớp query tiếng Việt với object labels tiếng Anh trong index.
# Ví dụ: query "người đàn ông" → khớp với record chứa "Person người"
# =================================================================
_ENTITY_VI_MAP = {
    "Person": "người",
    "Man": "đàn ông người đàn ông",
    "Woman": "phụ nữ người phụ nữ",
    "Child": "trẻ em",
    "Boy": "bé trai con trai",
    "Girl": "bé gái con gái",
    "Car": "xe ô tô xe hơi",
    "Motorcycle": "xe máy",
    "Bicycle": "xe đạp",
    "Bus": "xe buýt xe bus",
    "Truck": "xe tải",
    "Van": "xe van",
    "Vehicle": "phương tiện giao thông",
    "Wheel": "bánh xe",
    "Dog": "con chó chó",
    "Cat": "con mèo mèo",
    "Bird": "con chim chim",
    "Horse": "con ngựa ngựa",
    "Cow": "con bò bò",
    "Elephant": "con voi voi",
    "Tree": "cây cây xanh",
    "Building": "tòa nhà công trình",
    "House": "nhà",
    "Road": "đường đường phố",
    "Sky": "bầu trời",
    "Water": "nước",
    "Flag": "cờ",
    "Chair": "ghế",
    "Table": "bàn",
    "Food": "thức ăn đồ ăn",
    "Airplane": "máy bay",
    "Train": "tàu hỏa xe lửa",
    "Boat": "thuyền tàu",
    "Police officer": "cảnh sát công an",
    "Soldier": "binh sĩ lính",
    "Traffic light": "đèn giao thông đèn tín hiệu",
    "Helmet": "mũ bảo hiểm",
    "Glasses": "kính mắt",
    "Clothing": "quần áo",
    "Ball": "bóng",
    "Window": "cửa sổ",
    "Door": "cửa",
    "Crowd": "đám đông",
    "Stage": "sân khấu",
    "Microphone": "micro",
    "Fire": "lửa cháy",
    "Smoke": "khói",
    "Boat": "thuyền",
    "Sports equipment": "dụng cụ thể thao",
    "Signage": "biển hiệu biển báo",
}

_ENTITY_VI_MAP_LOWER = {k.lower(): v for k, v in _ENTITY_VI_MAP.items()}

def _get_vi_label(entity: str) -> str:
    """Tra nhãn tiếng Việt cho một entity class. Trả về chuỗi rỗng nếu không tìm thấy."""
    return _ENTITY_VI_MAP.get(entity) or _ENTITY_VI_MAP_LOWER.get(entity.lower(), "")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_btc_text_records():
    """Gộp metadata YouTube và objects vào danh sách records."""
    print("[1] Xây dựng text records từ dữ liệu BTC...")
    records = []
    
    kf_map = {}
    if os.path.exists(config.BTC_KF_META_CACHE):
        kf_map = load_json(config.BTC_KF_META_CACHE)
    
    # 1. Load Metadata (Video-level)
    if os.path.exists(config.BTC_MEDIA_INFO_CACHE):
        media_info = load_json(config.BTC_MEDIA_INFO_CACHE)
        for video_id, meta in media_info.items():
            parts = [
                meta.get("title", ""),
                meta.get("description", "")[:500],
                " ".join(meta.get("keywords", [])),
                meta.get("author", "")
            ]
            text = " ".join(p for p in parts if p).strip()
            if text:
                records.append({
                    "text": text,
                    "source": "metadata",
                    "video_id": video_id,
                    "frame_idx": None,
                    "timestamp_sec": 0.0
                })
        print(f"  + Đã thêm {len(media_info)} records từ YouTube metadata.")
    else:
        print("  - Không tìm thấy btc_media_info_cache")

    # 2. Load Objects (Frame-level)
    # Cấu trúc: data/btc/objects/objects/<video_id>/001.json, 002.json...
    objects_dir = os.path.join(config.BTC_DIR, "objects", "objects")
    if os.path.exists(objects_dir) and kf_map:
        obj_count = 0
        for video_id, frames in tqdm(kf_map.items(), desc="Đọc object labels"):
            video_obj_dir = os.path.join(objects_dir, video_id)
            if not os.path.exists(video_obj_dir):
                continue
            for i, frame in enumerate(frames):
                frame_idx = frame["frame_idx"]
                json_path = os.path.join(video_obj_dir, f"{i+1:03d}.json")
                if not os.path.exists(json_path):
                    continue
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    scores = data.get("detection_scores", [])
                    entities = data.get("detection_class_entities", [])
                    
                    # [IMPROVED] Hạ threshold xuống 0.2 để bắt được nhiều object hơn
                    # Đếm số lượng từng class để tạo text phông phú hơn
                    # (ví dụ: "Person Person Person Car" thìm BM25 và BGE-M3 hiểu "3 people 1 car")
                    class_counter = {}
                    for score_str, entity in zip(scores, entities):
                        if float(score_str) >= 0.2:
                            class_counter[entity] = class_counter.get(entity, 0) + 1
                    
                    if class_counter:
                        # Tạo text theo format: "Person người Person người Car xe ô tô"
                        # - Tiếng Anh: để BGE-M3 dense search (đã dịch sang Anh) khớp được
                        # - Tiếng Việt: để BM25 exact-match khớp được query tiếng Việt
                        parts = []
                        for entity, count in sorted(class_counter.items(), key=lambda x: -x[1]):
                            vi_label = _get_vi_label(entity)
                            for _ in range(min(count, 3)):  # tối đa 3 lần mỗi class
                                parts.append(entity)
                                if vi_label:
                                    parts.append(vi_label)
                        text = " ".join(parts)
                        records.append({
                            "text": text,
                            "source": "objects",
                            "video_id": video_id,
                            "frame_idx": frame_idx,
                            "timestamp_sec": frame["timestamp_sec"]
                        })
                        obj_count += 1
                except Exception as e:
                    pass
        print(f"  + Đã thêm {obj_count} records từ Object Detection (threshold=0.2, có count).")
    else:
        print("  - Không tìm thấy thư mục objects hoặc btc_keyframe_meta.json")


    # 3. Load OCR (Frame-level)
    # [BUG FIX] Tên file BTC là số thứ tự (001.jpg = ảnh thứ 1, không phải frame_idx thật).
    # Phải tra kf_map để lấy frame_idx thật và timestamp_sec đúng.
    if os.path.exists(config.OCR_CACHE):
        ocr_data = load_json(config.OCR_CACHE)
        ocr_count = 0
        # Build lookup: video_id -> {seq_num (1-based): frame_info}
        kf_seq_lookup = {}
        for vid, frames in kf_map.items():
            kf_seq_lookup[vid] = {i + 1: f for i, f in enumerate(frames)}
        
        for path, text in ocr_data.items():
            text = text.strip()
            if not text:
                continue
            
            # path: .../data/keyframes/L10_V010/001.jpg
            # 001.jpg → seq_num=1 → kf_map[video_id][0] (frame thứ nhất)
            try:
                filename = os.path.basename(path)
                video_id = os.path.basename(os.path.dirname(path))
                seq_num = int(filename.split(".")[0])  # 001 → 1, 002 → 2 ...
                
                # Tra kf_map để lấy frame_idx thật và timestamp_sec
                if video_id in kf_seq_lookup and seq_num in kf_seq_lookup[video_id]:
                    frame_info = kf_seq_lookup[video_id][seq_num]
                    frame_idx = frame_info["frame_idx"]
                    timestamp_sec = frame_info["timestamp_sec"]
                else:
                    # Fallback cho video không thuộc BTC (video01, video02...)
                    frame_idx = seq_num
                    timestamp_sec = 0.0
                            
                records.append({
                    "text": text,
                    "source": "ocr",
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "timestamp_sec": timestamp_sec
                })
                ocr_count += 1
            except Exception:
                pass
        print(f"  + Đã thêm {ocr_count} records từ OCR.")
    else:
        print("  - Không tìm thấy ocr.json")

    return records

def build_dense_text_index(records):
    print("\n[2] Build BGE-M3 Dense Text Index...")
    if not records:
        print("  - Không có text record nào. Bỏ qua.")
        return

    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(config.BGE_MODEL_NAME, use_fp16=(config.DEVICE == "cuda"))

    texts = [r["text"] for r in records]
    print(f"  - Đang encode {len(texts)} đoạn text...")
    
    # Chia batch lớn để chạy nhanh nếu có VRAM trống, BGE-M3 trên CPU cũng khá ổn
    output = model.encode(texts, batch_size=32, max_length=512)
    vectors = np.array(output["dense_vecs"]).astype("float32")

    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, config.TEXT_INDEX_PATH)
    with open(config.TEXT_ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
        
    print(f"  ✅ Đã build text index: dim={vectors.shape[1]}, records={len(records)}")
    print(f"  → {config.TEXT_INDEX_PATH}")


def build_sparse_bm25_index(records):
    print("\n[3] Build BM25 Sparse Index...")
    if not records:
        print("  - Không có text record nào. Bỏ qua.")
        return

    from rank_bm25 import BM25Okapi
    _tokenize, used_underthesea = get_tokenizer(use_underthesea=None)
    
    print(f"  - Tokenizer: {'underthesea' if used_underthesea else 'str.split()'}")
    print(f"  - Đang tokenize {len(records)} records...")
    
    tokenized_corpus = [_tokenize(r["text"]) for r in records]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "records": records,
            "used_underthesea": used_underthesea,
        }, f)

    print(f"  ✅ Đã build BM25 index: records={len(records)}")
    print(f"  → {config.BM25_INDEX_PATH}")


if __name__ == "__main__":
    records = build_btc_text_records()
    build_dense_text_index(records)
    build_sparse_bm25_index(records)
    print("\n🎉 Hoàn tất xây dựng BGE-M3 & BM25 cho BTC data!")
