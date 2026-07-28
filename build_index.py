"""
build_index.py
================
BƯỚC 2 của pipeline. Từ keyframe + transcript đã trích ở extract_keyframes.py,
xây dựng các FAISS index dùng để tìm kiếm:

  1) IMAGE INDEX (CLIP):
        keyframe (ảnh) -> CLIP image encoder -> vector -> FAISS
        -> cho phép tìm keyframe theo NỘI DUNG HÌNH ẢNH (vd: "người đàn ông áo đỏ")

  2) TEXT INDEX (BGE-M3), gộp CHUNG 3 nguồn text:
        a) Caption tiếng Anh do Gemini sinh ra cho từng keyframe (mô tả ảnh)
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
    Encode toàn bộ keyframe bằng CLIP image encoder -> FAISS index.
    Dùng IndexFlatIP (Inner Product) trên vector ĐÃ CHUẨN HÓA (L2-normalize)
    -> Inner Product trên vector đơn vị = Cosine Similarity, là cách chuẩn để so khớp embedding CLIP.
    """
    if os.path.exists(config.IMAGE_INDEX_PATH) and os.path.exists(config.IMAGE_ID_MAP_PATH):
        print("[CLIP] Đã có image index -> bỏ qua. (Xóa 2 file trong index/ nếu muốn build lại)")
        return

    import open_clip

    print(f"[CLIP] Đang tải model {config.CLIP_MODEL_NAME} ({config.CLIP_PRETRAINED})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED
    )
    model = model.to(config.DEVICE).eval()

    flat_keyframes = flatten_keyframes(keyframe_meta)
    if not flat_keyframes:
        print("[CLIP] Không có keyframe nào -> bỏ qua. Hãy chạy extract_keyframes.py trước.")
        return

    vectors = []
    id_map = []  # vị trí trong FAISS index -> metadata keyframe tương ứng

    with torch.no_grad():
        for kf in tqdm(flat_keyframes, desc="[CLIP] Encode keyframes"):
            try:
                image = Image.open(kf["path"]).convert("RGB")
                tensor = preprocess(image).unsqueeze(0).to(config.DEVICE)
                feat = model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)  # chuẩn hóa vector (bắt buộc cho cosine sim)
                vectors.append(feat.cpu().numpy()[0])
                id_map.append(kf)
            except Exception as e:
                print(f"[LỖI] Encode ảnh {kf['path']} thất bại: {e}")

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
    Gọi Gemini API để sinh caption mô tả nội dung từng keyframe.
    Cache theo KEY = đường dẫn ảnh -> nếu bị đứt giữa chừng (rate limit, mất mạng),
    chạy lại script sẽ chỉ caption tiếp phần ảnh còn thiếu, không làm lại từ đầu.
    """
    if not config.GEMINI_API_KEY:
        print("[Gemini] CHƯA cấu hình biến môi trường GEMINI_API_KEY -> bỏ qua bước caption.\n"
              "         (Pipeline vẫn chạy được, chỉ mất phần tín hiệu caption trong nhánh text)")
        return

    import google.generativeai as genai
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)

    caption_cache = load_cache(config.CAPTION_CACHE)  # {đường_dẫn_ảnh: caption}
    flat_keyframes = flatten_keyframes(keyframe_meta)

    prompt = (
        "Describe this video keyframe in 1-2 concise English sentences. "
        "Focus on: main objects, people, actions, setting/location, and any visible text or signage. "
        "Output ONLY the description, no preamble, no markdown."
    )

    processed_since_save = 0
    for kf in tqdm(flat_keyframes, desc="[Gemini] Sinh caption"):
        key = kf["path"]
        if key in caption_cache:
            continue
        try:
            image = Image.open(key).convert("RGB")
            response = model.generate_content([prompt, image])
            caption_cache[key] = (response.text or "").strip()
        except Exception as e:
            print(f"[LỖI] Gemini caption lỗi ở {key}: {e}")
            caption_cache[key] = ""  # đánh dấu đã thử (rỗng) để không loop vô hạn khi rerun
            time.sleep(2)  # nghỉ 1 chút nếu bị rate limit

        processed_since_save += 1
        if processed_since_save >= 20:  # lưu cache định kỳ, tránh mất tiến độ nếu bị ngắt giữa chừng
            save_cache(config.CAPTION_CACHE, caption_cache)
            processed_since_save = 0

    save_cache(config.CAPTION_CACHE, caption_cache)
    print(f"[Gemini] ✅ Đã caption {len(caption_cache)} ảnh.")


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
# MAIN
# =================================================================
if __name__ == "__main__":
    keyframe_meta = load_cache(config.KEYFRAME_META_CACHE)
    transcript_cache = load_cache(config.TRANSCRIPT_CACHE)

    if not keyframe_meta:
        raise RuntimeError(
            "Chưa có cache/keyframe_meta.json. Hãy chạy `python extract_keyframes.py` trước!"
        )

    build_image_index(keyframe_meta)                    # 1. CLIP -> FAISS image index
    build_captions(keyframe_meta)                        # 2. Gemini caption (cần GEMINI_API_KEY)
    build_ocr(keyframe_meta)                              # 3. EasyOCR
    build_text_index(keyframe_meta, transcript_cache)     # 4. BGE-M3 -> FAISS text index

    print("\n✅ Hoàn tất bước 2. Chạy `streamlit run app.py` để bắt đầu tìm kiếm.")
