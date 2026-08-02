"""
search_utils.py
=================
BƯỚC 3 của pipeline: TÌM KIẾM. File này CHỈ chứa logic xử lý (không có UI),
UI nằm ở app.py (Streamlit) và gọi vào các hàm ở đây.

Luồng xử lý đầy đủ cho 1 câu query (tiếng Việt hoặc tiếng Anh) - xem full_search():

    query
      ├─► [CLIP] langid detect → dịch Vi→En (Gemini → NLLB fallback) → SigLIP2 → Image FAISS → top 50
      ├─► [BGE-M3] dùng CÙNG bản đã dịch → Text FAISS → top 50
      └─► [BM25] query gốc (exact keyword) → BM25 sparse → top 50
                    │
                    ▼
        Hybrid Fusion (RRF hoặc weighted sum) → top 50 hợp nhất (loại trùng theo keyframe)
                    │
                    ▼
        CrossEncoder rerank (bge-reranker-v2-m3) → top 10
          (image-only candidates dùng caption cache làm fallback passage)
                    │
                    ▼
        (tuỳ chọn) Gemini verify lại bằng ẢNH THẬT → sắp xếp lại top 10
"""
import os
import json
import numpy as np
import faiss
import torch
from PIL import Image

# Fix AttributeError: type object 'tqdm' has no attribute '_lock' khi HuggingFace snapshot_download chạy trong Streamlit thread
try:
    import tqdm
    import tqdm.contrib.concurrent as _tqdm_concurrent
    from contextlib import contextmanager

    _orig_ensure_lock = _tqdm_concurrent.ensure_lock
    @contextmanager
    def _safe_ensure_lock(*args, **kwargs):
        try:
            with _orig_ensure_lock(*args, **kwargs) as lk:
                yield lk
        except AttributeError:
            yield None
    _tqdm_concurrent.ensure_lock = _safe_ensure_lock
except Exception:
    pass

import config
from tokenizer_utils import detect_language, get_tokenizer



# =================================================================
# BIẾN GLOBAL (cache model + index trong bộ nhớ, load 1 lần)
# =================================================================
_clip_model = None
_clip_tokenizer = None
_clip_preprocess = None
_bge_model = None
_reranker_model = None
_gemini_model = None
_bm25_data = None
_nllb_pipeline = None

_image_index = None
_image_id_map = None
_text_index = None
_text_id_map = None
_keyframe_meta = None
_caption_cache = None


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_indices():
    """Load FAISS index + id_map + keyframe_meta vào bộ nhớ (chỉ chạy thật sự 1 lần)."""
    global _image_index, _image_id_map, _text_index, _text_id_map, _keyframe_meta

    if _image_index is None:
        if not os.path.exists(config.IMAGE_INDEX_PATH):
            raise RuntimeError(
                "Chưa có index/image_index.faiss. Hãy chạy `python build_index.py` trước!"
            )
        _image_index = faiss.read_index(config.IMAGE_INDEX_PATH)
        _image_id_map = _load_json(config.IMAGE_ID_MAP_PATH)

    if _text_index is None and os.path.exists(config.TEXT_INDEX_PATH):
        _text_index = faiss.read_index(config.TEXT_INDEX_PATH)
        _text_id_map = _load_json(config.TEXT_ID_MAP_PATH)

    meta_cache_path = config.BTC_KF_META_CACHE if getattr(config, "USE_BTC_DATA", False) else config.KEYFRAME_META_CACHE
    if _keyframe_meta is None and os.path.exists(meta_cache_path):
        _keyframe_meta = _load_json(meta_cache_path)


def get_clip_model():
    """
    Load CLIP model phù hợp với index đã build:
    - USE_BTC_DATA=True  → ViT-B/32 OpenAI (dim=512, khớp với BTC CLIP features)
    - USE_BTC_DATA=False → SigLIP2 ViT-B-16 (dim=768, chất lượng cao hơn nhưng cần self-encode)
    """
    global _clip_model, _clip_tokenizer, _clip_preprocess
    if _clip_model is None:
        import open_clip
        if getattr(config, "USE_BTC_DATA", True):
            # ViT-B/32 OpenAI — cùng model mà BTC đã dùng để trích CLIP features
            model, _, preprocess = open_clip.create_model_and_transforms(
                config.CLIP_MODEL_NAME,
                pretrained=config.CLIP_PRETRAINED,
            )
        else:
            # SigLIP2: load qua hf-hub (create_model_from_pretrained trả về 2-tuple)
            model, preprocess = open_clip.create_model_from_pretrained(config.CLIP_MODEL_NAME)
        _clip_model = model.to(config.DEVICE).eval()
        _clip_preprocess = preprocess
        _clip_tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
    return _clip_model, _clip_tokenizer


def get_bge_model():
    global _bge_model
    if _bge_model is None:
        from FlagEmbedding import BGEM3FlagModel
        _bge_model = BGEM3FlagModel(config.BGE_MODEL_NAME, use_fp16=False, devices="cpu")
    return _bge_model


def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        from FlagEmbedding import FlagReranker
        _reranker_model = FlagReranker(
            config.RERANKER_MODEL_NAME, use_fp16=(config.DEVICE == "cuda")
        )
    return _reranker_model


def get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        import google.generativeai as genai
        genai.configure(api_key=config.GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(config.GEMINI_MODEL)
    return _gemini_model


def _get_nllb_pipeline():
    """Singleton: load NLLB-200-distilled-600M 1 lần, tái sử dụng cho mọi query."""
    global _nllb_pipeline
    if _nllb_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            print("[CLIP][NLLB] Đang load NLLB-200-distilled-600M...")
            _nllb_pipeline = hf_pipeline(
                "translation",
                model="facebook/nllb-200-distilled-600M",
                src_lang="vie_Latn",
                tgt_lang="eng_Latn",
                max_length=128,
                device=-1,
            )
            print("[CLIP][NLLB] NLLB-200 sẵn sàng (fallback dịch local khi mất mạng).")
        except Exception as e:
            print(f"[CLIP][NLLB] Không load được NLLB ({e}). Fallback về query gốc.")
            _nllb_pipeline = None
    return _nllb_pipeline


def _load_caption_cache():
    """Load caption cache vào bộ nhớ (lazy, chỉ load 1 lần)."""
    global _caption_cache
    if _caption_cache is None:
        if os.path.exists(config.CAPTION_CACHE):
            _caption_cache = _load_json(config.CAPTION_CACHE)
        else:
            _caption_cache = {}
    return _caption_cache


def _get_passage_for_candidate(c):
    """
    Lấy passage tốt nhất cho 1 candidate để đưa vào CrossEncoder.
    Ưu tiên: matched_text > caption cache > chuỗi rỗng.
    """
    text = (c.get("matched_text") or "").strip()
    if text and text != "Khung hình video":
        return text

    caption_cache = _load_caption_cache()
    path = c.get("path", "")
    if path:
        caption = (caption_cache.get(path) or "").strip()
        if caption and caption != "Khung hình video":
            return caption

    return ""


def translate_query_for_clip(query):
    """
    Dịch query tiếng Việt sang tiếng Anh.
    Chiến lược: langid detect → Gemini API → NLLB-200 local fallback → query gốc.

    FIX: thay isascii() bằng langid để xử lý đúng tiếng Việt không dấu.
    FIX: thêm NLLB local fallback khi Gemini lỗi/mất mạng lúc thi đấu.
    """
    lang = detect_language(query)
    if lang == "en":
        return query

    # Thử Gemini trước (nhanh + chất lượng cao)
    if config.GEMINI_API_KEY:
        try:
            model = get_gemini_model()
            prompt = (
                f'Translate the following Vietnamese search query to English. '
                f'Output ONLY the translated English text, nothing else.\n'
                f'Query: "{query}"'
            )
            response = model.generate_content(prompt)
            translated = (response.text or "").strip()
            if translated:
                print(f"[CLIP][Gemini] Dịch: '{query}' → '{translated}'")
                return translated
        except Exception as e:
            print(f"[CLIP][Gemini] Thất bại ({e}). Thử NLLB local...")

    # Fallback: NLLB-200 local (không cần internet)
    nllb = _get_nllb_pipeline()
    if nllb is not None:
        try:
            result = nllb(query)
            translated = result[0]["translation_text"].strip() if result else ""
            if translated:
                print(f"[CLIP][NLLB] Dịch: '{query}' → '{translated}'")
                return translated
        except Exception as e:
            print(f"[CLIP][NLLB] Lỗi ({e}). Dùng query gốc.")

    print(f"[CLIP] Không thể dịch '{query}'. Nhánh ảnh dùng query gốc — recall có thể giảm.")
    return query


# =================================================================
# 1) SEARCH ẢNH: SigLIP2 text encoder -> FAISS Image Index
# =================================================================
def search_image(query, top_k=None, _pre_translated=None):
    """
    Encode câu query bằng SigLIP2 text encoder rồi search Image FAISS Index.
    _pre_translated: nếu full_search() đã dịch rồi, truyền vào để tránh gọi API lần 2.
    """
    load_indices()
    top_k = top_k or config.TOP_K_RETRIEVE
    model, tokenizer = get_clip_model()

    clip_query = _pre_translated if _pre_translated is not None else translate_query_for_clip(query)

    with torch.no_grad():
        tokens = tokenizer([clip_query]).to(config.DEVICE)
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        vec = feat.cpu().numpy().astype("float32")

    scores, idxs = _image_index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        meta = _image_id_map[idx]
        
        # Xử lý đường dẫn ảnh (đặc biệt cho dữ liệu BTC)
        img_path = meta.get("path", "")
        if not img_path:
            img_path = _get_btc_image_path(meta["video_id"], meta["frame_idx"])
                
        results.append({
            "video_id": meta["video_id"],
            "frame_idx": meta["frame_idx"],
            "timestamp_sec": meta["timestamp_sec"],
            "path": img_path,
            "score": float(score),
        })
    return results


# =================================================================
# 2) SEARCH TEXT: BGE-M3 -> FAISS Text Index
# =================================================================
def _get_btc_image_path(video_id, frame_idx):
    """Suy luận đường dẫn ảnh BTC dựa vào thứ tự frame (1-based)."""
    base_dir = os.path.join(config.KEYFRAME_DIR, video_id)
    # Tìm thứ tự của frame này trong mảng
    if _keyframe_meta and video_id in _keyframe_meta:
        frames = _keyframe_meta[video_id]
        for i, f in enumerate(frames):
            if f["frame_idx"] == frame_idx:
                seq_idx = i + 1
                # BTC thường đặt tên 001.jpg, 002.jpg...
                p = os.path.join(base_dir, f"{seq_idx:03d}.jpg")
                if os.path.exists(p):
                    return p
                p = os.path.join(base_dir, f"{seq_idx:04d}.jpg")
                if os.path.exists(p):
                    return p
    # Fallback lại frame_idx nếu không tìm thấy
    for fmt in [f"{frame_idx:04d}.jpg", f"{frame_idx:08d}.jpg", f"{frame_idx:06d}.jpg", f"{frame_idx}.jpg"]:
        p = os.path.join(base_dir, fmt)
        if os.path.exists(p):
            return p
    return os.path.join(base_dir, f"{frame_idx:03d}.jpg")


def _nearest_keyframe(video_id, timestamp_sec):
    """Tìm keyframe gần nhất về mặt thời gian cho record transcript."""
    if _keyframe_meta is None or video_id not in _keyframe_meta:
        return None
    metas = _keyframe_meta[video_id]
    if not metas:
        return None
    return min(metas, key=lambda m: abs(m["timestamp_sec"] - timestamp_sec))


def search_text(query, top_k=None):
    """
    Encode query bằng BGE-M3 (đa ngôn ngữ) rồi search Text FAISS Index.
    Khi được gọi từ full_search(), query đã là tiếng Anh (đã dịch trước).
    """
    load_indices()
    if _text_index is None:
        return []

    top_k = top_k or config.TOP_K_RETRIEVE
    model = get_bge_model()

    output = model.encode([query], max_length=512)
    vec = np.array(output["dense_vecs"]).astype("float32")
    faiss.normalize_L2(vec)

    scores, idxs = _text_index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        record = _text_id_map[idx]
        frame_idx = record["frame_idx"]

        if frame_idx is None:
            nearest = _nearest_keyframe(record["video_id"], record["timestamp_sec"])
            if nearest is None:
                continue
            frame_idx = nearest["frame_idx"]
            path = nearest.get("path", "")
        else:
            path = record.get("path", "")
            
        if not path:
            path = _get_btc_image_path(record["video_id"], frame_idx)

        results.append({
            "video_id": record["video_id"],
            "frame_idx": frame_idx,
            "timestamp_sec": record["timestamp_sec"],
            "path": path,
            "score": float(score),
            "matched_text": record["text"],
            "text_source": record["source"],
        })
    return results


# =================================================================
# 3) SEARCH BM25: sparse keyword retrieval
# =================================================================
def get_bm25():
    """Lazy-load BM25 index từ pickle (chỉ load 1 lần, CPU-only)."""
    global _bm25_data
    if _bm25_data is None:
        import pickle
        if os.path.exists(config.BM25_INDEX_PATH):
            with open(config.BM25_INDEX_PATH, "rb") as f:
                _bm25_data = pickle.load(f)
    return _bm25_data


def search_bm25(query, top_k=None):
    """
    Tìm kiếm BM25 Okapi trên corpus caption + OCR + transcript.
    FIX: đọc flag used_underthesea từ pickle để tokenize đúng theo cách đã build.
    """
    top_k = top_k or config.TOP_K_RETRIEVE
    data = get_bm25()
    if data is None:
        return []

    bm25, records = data["bm25"], data["records"]

    # FIX: dùng đúng tokenizer theo cách đã build (backward-compat với pickle cũ)
    used_underthesea = data.get("used_underthesea", False)
    _tokenize, _ = get_tokenizer(use_underthesea=used_underthesea)
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    top_idxs = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_idxs:
        if scores[idx] <= 0:
            break
        r = records[int(idx)]
        frame_idx = r["frame_idx"]

        if frame_idx is None:
            nearest = _nearest_keyframe(r["video_id"], r["timestamp_sec"])
            if nearest is None:
                continue
            frame_idx = nearest["frame_idx"]
            path = nearest.get("path", "")
        else:
            path = r.get("path", "")
            
        if not path:
            path = _get_btc_image_path(r["video_id"], frame_idx)

        results.append({
            "video_id": r["video_id"],
            "frame_idx": frame_idx,
            "timestamp_sec": r["timestamp_sec"],
            "path": path,
            "score": float(scores[idx]),
            "matched_text": r["text"],
            "text_source": r["source"],
        })
    return results


# =================================================================
# 4) HYBRID FUSION
# =================================================================
def _key(item):
    return (item["video_id"], item["frame_idx"])


def fuse_rrf(image_results, text_results, bm25_results=None, k=None, top_k=None):
    """Reciprocal Rank Fusion (RRF) cho 3 nhánh: CLIP + BGE-M3 + BM25."""
    k = k or config.RRF_K
    top_k = top_k or config.TOP_K_RETRIEVE

    rrf_scores = {}
    item_cache = {}

    for rank, item in enumerate(image_results):
        key = _key(item)
        # Trọng số x3 cho Image để tránh bị OCR đè bẹp
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 3.0 / (k + rank + 1)
        item_cache.setdefault(key, item)

    for rank, item in enumerate(text_results):
        key = _key(item)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in item_cache or "matched_text" not in item_cache[key]:
            item_cache[key] = item

    for rank, item in enumerate(bm25_results or []):
        key = _key(item)
        # Trọng số x0.5 cho BM25 vì hay bị dính từ khóa nhiễu trong OCR/News Ticker
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 0.5 / (k + rank + 1)
        if key not in item_cache or "matched_text" not in item_cache[key]:
            item_cache[key] = item

    ranked_keys = sorted(rrf_scores.keys(), key=lambda kk: rrf_scores[kk], reverse=True)
    fused = []
    for key in ranked_keys[:top_k]:
        item = dict(item_cache[key])
        item["fused_score"] = rrf_scores[key]
        fused.append(item)
    return fused


def fuse_weighted(image_results, text_results, image_weight=None, text_weight=None, top_k=None):
    """Weighted sum fusion (không hỗ trợ BM25 — dùng RRF nếu cần BM25)."""
    image_weight = image_weight if image_weight is not None else config.IMAGE_WEIGHT
    text_weight = text_weight if text_weight is not None else config.TEXT_WEIGHT
    top_k = top_k or config.TOP_K_RETRIEVE

    def normalize(items):
        if not items:
            return {}
        s = [it["score"] for it in items]
        lo, hi = min(s), max(s)
        rng = (hi - lo) or 1e-6
        return {_key(it): (it["score"] - lo) / rng for it in items}

    img_norm = normalize(image_results)
    txt_norm = normalize(text_results)

    item_cache = {_key(it): it for it in image_results}
    for it in text_results:
        item_cache.setdefault(_key(it), it)

    all_keys = set(img_norm) | set(txt_norm)
    combined = {
        key: image_weight * img_norm.get(key, 0.0) + text_weight * txt_norm.get(key, 0.0)
        for key in all_keys
    }
    ranked_keys = sorted(combined.keys(), key=lambda kk: combined[kk], reverse=True)

    fused = []
    for key in ranked_keys[:top_k]:
        item = dict(item_cache[key])
        item["fused_score"] = combined[key]
        fused.append(item)
    return fused


# =================================================================
# 5) CROSS-ENCODER RERANK
# =================================================================
def rerank(query, candidates, top_k=None):
    """CrossEncoder bge-reranker-v2-m3 rerank top candidates."""
    top_k = top_k or config.TOP_K_RERANK
    if not candidates:
        return []

    reranker = get_reranker_model()
    pairs = [[query, _get_passage_for_candidate(c)] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)

    if isinstance(scores, float):
        scores = [scores]

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
        # Kết hợp điểm RRF (visual) và Reranker (text) để không loại bỏ ảnh thuần túy
        c["final_score"] = c.get("fused_score", 0.0) + float(s) * 0.5

    return sorted(candidates, key=lambda c: c["final_score"], reverse=True)[:top_k]


# =================================================================
# 6) VERIFY BẰNG GEMINI (VLM)
# =================================================================
def verify_with_gemini(query, candidates):
    """Gửi ảnh thật cho Gemini chấm điểm 0-10. Bước lọc cuối cùng."""
    if not config.GEMINI_API_KEY:
        for c in candidates:
            c["verify_score"] = None
        return candidates

    model = get_gemini_model()
    prompt_template = (
        'You are scoring video keyframe retrieval results.\n'
        'Query (Vietnamese or English): "{query}"\n'
        'Rate how well this image matches the query using this rubric:\n'
        '  9-10: Perfect match (correct objects AND correct action/scene)\n'
        '  6-8:  Partial match (correct objects, wrong action OR context)\n'
        '  3-5:  Weak match (related topic but different content)\n'
        '  0-2:  No match (completely unrelated)\n'
        'Respond with ONLY valid JSON, no markdown: {{"score": <0-10>}}'
    )

    for c in candidates:
        try:
            img_path = c.get("path", "")
            if not img_path or not os.path.exists(img_path):
                c["verify_score"] = 0
                continue
            image = Image.open(img_path).convert("RGB")
            response = model.generate_content([prompt_template.format(query=query), image])
            raw = (response.text or "").strip()
            try:
                import json as _json
                score = int(_json.loads(raw).get("score", 0))
            except Exception:
                digits = "".join(ch for ch in raw if ch.isdigit())
                score = int(digits[:2]) if digits else 0
            c["verify_score"] = max(0, min(10, score))
        except Exception as e:
            print(f"[LỖI] Gemini verify lỗi ở {c['path']}: {e}")
            c["verify_score"] = None

    def sort_key(c):
        has_verify = c.get("verify_score") is not None
        return (has_verify, c.get("verify_score") if has_verify else 0, c.get("final_score", c.get("fused_score", 0)))

    return sorted(candidates, key=sort_key, reverse=True)


# =================================================================
# HÀM TỔNG HỢP
# =================================================================
def full_search(query, fusion_method=None, do_verify=False):
    """
    Chạy toàn bộ pipeline: search_image + search_text + search_bm25
    → fusion → rerank → (tuỳ chọn) verify.

    FIX query tiếng Việt không dấu: dịch 1 lần, dùng chung cho SigLIP2 và BGE-M3.
    BM25 giữ query gốc (exact keyword match).
    CrossEncoder dùng query gốc (multilingual).

    Returns: (fused, reranked, verified)
    """
    fusion_method = fusion_method or config.FUSION_METHOD

    # Dịch 1 lần duy nhất, dùng chung cho cả SigLIP2 và BGE-M3
    lang = detect_language(query)
    if lang != "en":
        translated_query = translate_query_for_clip(query)
        print(f"[full_search] '{query}' → '{translated_query}' (SigLIP2 + BGE-M3)")
    else:
        translated_query = query

    # Nhánh ảnh: dùng bản đã dịch (tránh gọi API lần 2)
    image_results = search_image(query, top_k=config.TOP_K_RETRIEVE, _pre_translated=translated_query)
    # Nhánh text dense: dùng bản đã dịch (BGE-M3 hiểu tiếng Anh tốt hơn Việt không dấu)
    text_results = search_text(translated_query, top_k=config.TOP_K_RETRIEVE)
    # Nhánh BM25: giữ query gốc (exact-match tên riêng, số liệu)
    bm25_results = search_bm25(query, top_k=config.TOP_K_RETRIEVE)

    if fusion_method == "rrf":
        fused = fuse_rrf(image_results, text_results, bm25_results=bm25_results,
                         top_k=config.TOP_K_RETRIEVE)
    else:
        fused = fuse_weighted(image_results, text_results, top_k=config.TOP_K_RETRIEVE)

    # CrossEncoder: query gốc (multilingual, hiểu tiếng Việt có dấu + tiếng Anh)
    reranked = rerank(query, fused, top_k=config.TOP_K_RERANK)
    verified = verify_with_gemini(query, reranked) if do_verify else None

    return fused, reranked, verified
