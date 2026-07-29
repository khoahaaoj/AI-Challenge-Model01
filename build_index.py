"""
build_index.py
================
BƯỚC 2 của pipeline. Từ keyframe + transcript đã trích ở extract_keyframes.py,
xây dựng các FAISS index dùng để tìm kiếm:

  1) IMAGE INDEX (CLIP):
        keyframe (ảnh) -> CLIP image encoder -> vector -> FAISS
        -> cho phép tìm keyframe theo NỘI DUNG HÌNH ẢNH (vd: "người đàn ông áo đỏ")

  2) TEXT INDEX (BGE-M3), gộp CHUNG 3 nguồn text:
        a) Caption tiếng Việt do Qwen2-VL-2B-Instruct (4-bit) sinh ra cho từng keyframe
        b) Text đọc được bằng OCR (vi + en) trong từng keyframe (biển hiệu, phụ đề...)
        c) Transcript (lời thoại) từ Whisper
     -> BGE-M3 là model đa ngôn ngữ, encode tốt CẢ tiếng Việt lẫn tiếng Anh,
        nên nhánh này bù lại điểm yếu "chỉ hiểu tiếng Anh" của CLIP.

Mỗi bước đều CACHE ra file riêng (captions.json, ocr.json, *.faiss...) để khi debug
lỗi 1 bước, không phải chạy lại từ đầu các bước trước.

Cách chạy:
    python build_index.py
"""

import os
import json
import time
import numpy as np
import faiss
import torch
from PIL import Image
from tqdm import tqdm

import config


# =================================================================
# TIỆN ÍCH CHUNG
# =================================================================
def load_cache(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def flatten_keyframes(keyframe_meta):
    """Gộp toàn bộ keyframe của mọi video thành 1 list phẳng, tiện duyệt tuần tự."""
    flat = []
    for metas in keyframe_meta.values():
        flat.extend(metas)
    return flat


# =================================================================
# 1) CLIP IMAGE INDEX
# =================================================================
def build_image_index(keyframe_meta):
    """
    Encode toàn bộ keyframe bằng CLIP image encoder -> FAISS index (BATCHED).
    Dùng IndexFlatIP (Inner Product) trên vector ĐÃ CHUẨN HÓA (L2-normalize)
    -> Inner Product trên vector đơn vị = Cosine Similarity, là cách chuẩn để so khớp embedding CLIP.

    Batch encode (thay vì từng ảnh 1) giúp GPU tận dụng parallelism, nhanh hơn 5-8x.
    CLIP_BATCH_SIZE=16 an toàn với 4GB VRAM. Tăng lên 32-64 nếu VRAM lớn hơn.
    """
    if os.path.exists(config.IMAGE_INDEX_PATH) and os.path.exists(config.IMAGE_ID_MAP_PATH):
        print("[CLIP] Đã có image index -> bỏ qua. (Xóa 2 file trong index/ nếu muốn build lại)")
        return

    import open_clip

    print(f"[CLIP] Đang tải model SigLIP2: {config.CLIP_MODEL_NAME}...")
    # SigLIP2 load qua hf-hub: dùng create_model_from_pretrained (trả về 2-tuple)
    # thay vì create_model_and_transforms (3-tuple) của CLIP cũ.
    model, preprocess = open_clip.create_model_from_pretrained(config.CLIP_MODEL_NAME)
    model = model.to(config.DEVICE).eval()

    flat_keyframes = flatten_keyframes(keyframe_meta)
    if not flat_keyframes:
        print("[CLIP] Không có keyframe nào -> bỏ qua. Hãy chạy extract_keyframes.py trước.")
        return

    # CLIP_BATCH_SIZE: 16 an toàn với 4GB VRAM (ảnh CLIP resize về 224×224, nhỏ).
    # Tăng lên 32 hoặc 64 nếu VRAM lớn hơn để nhanh hơn nữa.
    CLIP_BATCH_SIZE = getattr(config, "CLIP_BATCH_SIZE", 16)
    print(f"[CLIP] Encode {len(flat_keyframes)} keyframes với batch_size={CLIP_BATCH_SIZE}...")

    vectors = []
    id_map = []  # vị trí trong FAISS index -> metadata keyframe tương ứng

    with torch.no_grad():
        for batch_start in tqdm(
            range(0, len(flat_keyframes), CLIP_BATCH_SIZE),
            desc="[CLIP] Encode keyframes (batched)"
        ):
            batch_kfs = flat_keyframes[batch_start: batch_start + CLIP_BATCH_SIZE]

            # Load ảnh — lọc bỏ ảnh lỗi để 1 ảnh hỏng không crash cả batch
            tensors = []
            valid_kfs = []
            for kf in batch_kfs:
                try:
                    image = Image.open(kf["path"]).convert("RGB")
                    tensors.append(preprocess(image))
                    valid_kfs.append(kf)
                except Exception as e:
                    print(f"[LỖI] Đọc ảnh {kf['path']} thất bại: {e}")

            if not tensors:
                continue

            # Stack thành 1 tensor batch -> encode cùng lúc -> GPU tận dụng tối đa
            batch_tensor = torch.stack(tensors).to(config.DEVICE)
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2-normalize (bắt buộc cho cosine sim)

            vectors.extend(feats.cpu().numpy().tolist())
            id_map.extend(valid_kfs)

    vectors = np.array(vectors).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, config.IMAGE_INDEX_PATH)
    save_cache(config.IMAGE_ID_MAP_PATH, id_map)
    print(f"[CLIP] ✅ Đã build image index với {len(id_map)} keyframe "
          f"(dim={vectors.shape[1]}).")


# =================================================================
# 2) GEMINI CAPTION (sinh caption tiếng Anh cho từng keyframe)
# =================================================================
def build_captions(keyframe_meta):
    """
    Sinh caption mô tả nội dung cho từng keyframe bằng Qwen2-VL-2B-Instruct (4-bit quantized).
    Chạy cục bộ trên GPU, không cần API key. Phù hợp với máy có VRAM 4GB.

    Qwen2-VL-2B được nén 4-bit (NF4) qua BitsAndBytes giúp tiết kiệm VRAM ~75%%.
    Caption sinh bằng tiếng Việt (phù hợp với dataset và query tiếng Việt của cuộc thi).
    """
    caption_cache = load_cache(config.CAPTION_CACHE)
    flat_keyframes = flatten_keyframes(keyframe_meta)

    # Lọc các ảnh chưa có caption hoặc bị dính nhãn fallback
    unprocessed_frames = [
        kf for kf in flat_keyframes
        if (kf["path"] not in caption_cache
            or not caption_cache[kf["path"]]
            or caption_cache[kf["path"]] == "Khung hình video")
        and os.path.exists(kf["path"])
    ]

    if not unprocessed_frames:
        print("[Qwen2-VL] Tất cả keyframe đã có caption hợp lệ trong cache -> bỏ qua.")
        return

    print(f"[Qwen2-VL] Tìm thấy {len(unprocessed_frames)} ảnh cần sinh caption.")
    print(f"[Qwen2-VL] Đang tải model Qwen2-VL-2B-Instruct (4-bit) lên {config.DEVICE}...")

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    device = config.DEVICE

    # Cấu hình nén 4-bit tiết kiệm bộ nhớ VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    ) if device == "cuda" else None

    # Tải model 4-bit
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        dtype=torch.float16 if device == "cuda" else torch.float32
    )

    # Giới hạn max_pixels để tiết kiệm bộ nhớ khi encode ảnh
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        max_pixels=384 * 28 * 28  # Resize ảnh nhỏ gọn để không ngốn VRAM
    )

    processed_since_save = 0
    for kf in tqdm(unprocessed_frames, desc="[Qwen2-VL 4-bit] Sinh caption"):
        key = kf["path"]
        try:
            image = Image.open(key).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Mô tả ngắn gọn nội dung bức ảnh này bằng tiếng Việt trong 1 câu:"},
                    ],
                }
            ]

            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text_prompt],
                images=[image],
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                generated_ids = qwen_model.generate(**inputs, max_new_tokens=50)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            cap = output_text[0].strip() if output_text else ""
            caption_cache[key] = cap if cap else "Khung hình video"

        except Exception as e:
            print(f"\n❌ LỖI: {e}")
            caption_cache[key] = "Khung hình video"

        processed_since_save += 1
        if processed_since_save >= 10:
            save_cache(config.CAPTION_CACHE, caption_cache)
            processed_since_save = 0

    save_cache(config.CAPTION_CACHE, caption_cache)

    # Giải phóng hoàn toàn VRAM cho bước EasyOCR và BGE-M3 tiếp theo
    del qwen_model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Qwen2-VL] ✅ Đã hoàn tất caption cho {len(caption_cache)} ảnh.")


# =================================================================
# 3) EASYOCR (đọc chữ trong từng keyframe, hỗ trợ song song vi + en)
# =================================================================
def build_ocr(keyframe_meta):
    """
    Đọc text xuất hiện trong từng keyframe (biển hiệu, phụ đề cứng, banner, chú thích...)
    bằng EasyOCR với 2 ngôn ngữ vi + en cùng lúc.
    """
    import easyocr

    ocr_cache = load_cache(config.OCR_CACHE)

    print(f"[EasyOCR] Đang tải model OCR cho ngôn ngữ {config.OCR_LANGS}...")
    reader = easyocr.Reader(config.OCR_LANGS, gpu=(config.DEVICE == "cuda"))

    flat_keyframes = flatten_keyframes(keyframe_meta)
    processed_since_save = 0
    for kf in tqdm(flat_keyframes, desc="[EasyOCR] Đọc text trong ảnh"):
        key = kf["path"]
        if key in ocr_cache:
            continue
        try:
            # detail=0 -> chỉ trả về list chuỗi text (bỏ bounding box + confidence cho gọn)
            texts = reader.readtext(key, detail=0)
            ocr_cache[key] = " ".join(texts).strip()
        except Exception as e:
            print(f"[LỖI] OCR lỗi ở {key}: {e}")
            ocr_cache[key] = ""

        processed_since_save += 1
        if processed_since_save >= 50:
            save_cache(config.OCR_CACHE, ocr_cache)
            processed_since_save = 0

    save_cache(config.OCR_CACHE, ocr_cache)
    print(f"[EasyOCR] ✅ Đã OCR {len(ocr_cache)} ảnh.")


# =================================================================
# 4) GỘP TEXT RECORDS (caption + ocr + transcript) -> BGE-M3 TEXT INDEX
# =================================================================
def build_text_records(keyframe_meta, transcript_cache):
    """
    Gộp 3 nguồn text thành 1 danh sách record THỐNG NHẤT.
    Mỗi record: {text, source, video_id, frame_idx, timestamp_sec}
      - caption/ocr: frame_idx là số cụ thể (gắn trực tiếp với 1 keyframe)
      - transcript : frame_idx = None (chỉ có timestamp, sẽ suy ra keyframe gần nhất lúc SEARCH)
    Record có text rỗng bị loại để tránh làm nhiễu index.
    """
    caption_cache = load_cache(config.CAPTION_CACHE)
    ocr_cache = load_cache(config.OCR_CACHE)

    records = []

    # -- caption + OCR: gắn trực tiếp với từng keyframe --
    for video_id, metas in keyframe_meta.items():
        for kf in metas:
            path = kf["path"]

            caption = caption_cache.get(path, "").strip()
            if caption:
                records.append({
                    "text": caption, "source": "caption",
                    "video_id": video_id, "frame_idx": kf["frame_idx"],
                    "timestamp_sec": kf["timestamp_sec"],
                })

            ocr_text = ocr_cache.get(path, "").strip()
            if ocr_text:
                records.append({
                    "text": ocr_text, "source": "ocr",
                    "video_id": video_id, "frame_idx": kf["frame_idx"],
                    "timestamp_sec": kf["timestamp_sec"],
                })

    # -- transcript: gắn theo timestamp, KHÔNG gắn keyframe cụ thể ở bước build --
    for video_id, info in transcript_cache.items():
        for seg in info.get("segments", []):
            text = seg["text"].strip()
            if text:
                records.append({
                    "text": text, "source": "transcript",
                    "video_id": video_id, "frame_idx": None,
                    "timestamp_sec": seg["start"],
                })

    return records


def build_text_index(keyframe_meta, transcript_cache):
    """
    Encode toàn bộ text record bằng BGE-M3 -> FAISS index (cosine similarity qua Inner Product
    trên vector đã chuẩn hóa, giống nhánh CLIP).
    """
    if os.path.exists(config.TEXT_INDEX_PATH) and os.path.exists(config.TEXT_ID_MAP_PATH):
        print("[BGE-M3] Đã có text index -> bỏ qua. (Xóa 2 file trong index/ nếu muốn build lại)")
        return

    from FlagEmbedding import BGEM3FlagModel

    records = build_text_records(keyframe_meta, transcript_cache)
    if not records:
        print("[BGE-M3] Không có text record nào (caption/ocr/transcript đều rỗng) -> bỏ qua.")
        return

    print(f"[BGE-M3] Đang tải model {config.BGE_MODEL_NAME}...")
    model = BGEM3FlagModel(config.BGE_MODEL_NAME, use_fp16=(config.DEVICE == "cuda"))

    texts = [r["text"] for r in records]
    print(f"[BGE-M3] Encode {len(texts)} đoạn text (caption + ocr + transcript)...")
    output = model.encode(texts, batch_size=32, max_length=512)
    vectors = np.array(output["dense_vecs"]).astype("float32")

    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, config.TEXT_INDEX_PATH)
    save_cache(config.TEXT_ID_MAP_PATH, records)
    print(f"[BGE-M3] ✅ Đã build text index với {len(records)} đoạn text "
          f"(dim={vectors.shape[1]}).")


# =================================================================
# 5) BM25 SPARSE INDEX (keyword exact match)
# =================================================================
def build_bm25_index(records):
    """
    Build BM25 sparse index từ cùng text records (caption + ocr + transcript).

    BM25 bổ sung điểm mạnh mà dense vector (BGE-M3) yếu:
    exact keyword match — tên người, địa danh, thương hiệu, số liệu.
    Không cần GPU, không tốn VRAM, build cực nhanh (<5s).

    Tokenizer:
      - Nếu đã cài underthesea: tách từ tiếng Việt chính xác hơn
        (pip install underthesea). Tốt hơn với query ghép từ như "thủ tướng chính phủ".
      - Fallback về str.split(): vẫn tốt với tên riêng và keyword ngắn.
    """
    import pickle
    from rank_bm25 import BM25Okapi

    if os.path.exists(config.BM25_INDEX_PATH):
        print("[BM25] Đã có index → bỏ qua. (Xóa index/bm25_index.pkl để build lại)")
        return

    if not records:
        print("[BM25] Không có records → bỏ qua.")
        return

    # Chọn tokenizer: underthesea nếu có, fallback simple split
    try:
        from underthesea import word_tokenize
        def _tokenize(text):
            return word_tokenize(text.lower(), format="text").split()
        print("[BM25] Dùng underthesea word_tokenize (tiếng Việt tốt hơn).")
    except ImportError:
        def _tokenize(text):
            return text.lower().split()
        print("[BM25] Dùng str.split() (cài underthesea để cải thiện tiếng Việt).")

    print(f"[BM25] Tokenize và build index từ {len(records)} records...")
    tokenized_corpus = [_tokenize(r["text"]) for r in records]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "records": records}, f)

    print(f"[BM25] ✅ Đã build BM25 index với {len(records)} records.")


# =================================================================
# MAIN
# =================================================================
if __name__ == "__main__":
    keyframe_meta = load_cache(config.KEYFRAME_META_CACHE)
    transcript_cache = load_cache(config.TRANSCRIPT_CACHE)

    if not keyframe_meta:
        raise RuntimeError(
            "Chưa có cache/keyframe_meta.json. Hãy chạy `python extract_keyframes.py` trước!"
        )

    build_image_index(keyframe_meta)                      # 1. SigLIP2 -> FAISS image index
    build_captions(keyframe_meta)                          # 2. Qwen2-VL caption
    build_ocr(keyframe_meta)                               # 3. EasyOCR
    build_text_index(keyframe_meta, transcript_cache)      # 4. BGE-M3 -> FAISS text index
    records = build_text_records(keyframe_meta, transcript_cache)  # tái dùng records đã có
    build_bm25_index(records)                              # 5. BM25 sparse index

    print("\n✅ Hoàn tất bước 2. Chạy `streamlit run app.py` để bắt đầu tìm kiếm.")
