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

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_btc_text_records():
    """Gộp metadata YouTube và objects vào danh sách records."""
    print("[1] Xây dựng text records từ dữ liệu BTC...")
    records = []
    
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
    if os.path.exists(objects_dir) and os.path.exists(config.BTC_KF_META_CACHE):
        kf_map = load_json(config.BTC_KF_META_CACHE)
        obj_count = 0
        for video_id, frames in tqdm(kf_map.items(), desc="Đọc object labels"):
            video_obj_dir = os.path.join(objects_dir, video_id)
            if not os.path.exists(video_obj_dir):
                continue
            for i, frame in enumerate(frames):
                frame_idx = frame["frame_idx"]
                # BTC lưu file tên theo số thứ tự (1-based) với padding 3 chữ số: 001.json, 002.json
                json_path = os.path.join(video_obj_dir, f"{i+1:03d}.json")
                if not os.path.exists(json_path):
                    continue
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Format thực tế: {"detection_scores": ["0.9",...], "detection_class_entities": ["Person",...]}
                    scores = data.get("detection_scores", [])
                    entities = data.get("detection_class_entities", [])
                    
                    classes = []
                    for score_str, entity in zip(scores, entities):
                        if float(score_str) >= 0.3:
                            classes.append(entity)
                            
                    if classes:
                        # Lọc trùng và nối lại
                        text = " ".join(set(classes))
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
        print(f"  + Đã thêm {obj_count} records từ Object Detection.")
    else:
        print("  - Không tìm thấy thư mục objects hoặc btc_keyframe_meta.json")

    # 3. Load OCR (Frame-level)
    if os.path.exists(config.OCR_CACHE):
        ocr_data = load_json(config.OCR_CACHE)
        ocr_count = 0
        for path, text in ocr_data.items():
            text = text.strip()
            if not text:
                continue
            
            # path: .../data/keyframes/L10_V010/001.jpg
            try:
                filename = os.path.basename(path)
                video_id = os.path.basename(os.path.dirname(path))
                frame_idx_str = filename.split(".")[0]
                frame_idx = int(frame_idx_str)
                
                # Cần timestamp_sec. Lấy từ kf_map nếu có
                timestamp_sec = 0.0
                if "kf_map" in locals() and video_id in kf_map:
                    for f in kf_map[video_id]:
                        if f["frame_idx"] == frame_idx:
                            timestamp_sec = f["timestamp_sec"]
                            break
                            
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
