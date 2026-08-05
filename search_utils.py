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
_siglip_model = None
_siglip_preprocess = None
_siglip_tokenizer = None
_bge_model = None
_reranker_model = None
_gemini_model = None
_bm25_data = None
_nllb_pipeline = None

_image_index = None
_image_id_map = None
_siglip_index = None
_siglip_id_map = None
_text_index = None
_text_id_map = None
_keyframe_meta = None
_caption_cache = None

# Paths SigLIP2 index (defined here to avoid circular import)
_SIGLIP_INDEX_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index", "siglip_image_index.faiss")
_SIGLIP_ID_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index", "siglip_image_id_map.json")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_indices():
    """Load FAISS index + id_map + keyframe_meta vào bộ nhớ (chỉ chạy thật sự 1 lần)."""
    global _image_index, _image_id_map, _siglip_index, _siglip_id_map, _text_index, _text_id_map, _keyframe_meta

    if _image_index is None:
        if not os.path.exists(config.IMAGE_INDEX_PATH):
            raise RuntimeError(
                "Chưa có index/image_index.faiss. Hãy chạy `python build_index.py` trước!"
            )
        _image_index = faiss.read_index(config.IMAGE_INDEX_PATH)
        _image_id_map = _load_json(config.IMAGE_ID_MAP_PATH)

    # SigLIP2 index — tự động load nếu đã build
    if _siglip_index is None and os.path.exists(_SIGLIP_INDEX_PATH):
        print("[Search] SigLIP2 index detected — loading...")
        _siglip_index = faiss.read_index(_SIGLIP_INDEX_PATH)
        _siglip_id_map = _load_json(_SIGLIP_ID_MAP_PATH)
        print(f"[Search] SigLIP2 index: {_siglip_index.ntotal:,} vectors (dim={_siglip_index.d})")

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


def get_siglip_model():
    """Load SigLIP2-SO400M model (chỉ khi index đã tồn tại)."""
    global _siglip_model, _siglip_preprocess, _siglip_tokenizer
    if _siglip_model is None:
        import open_clip
        MODEL_NAME = "hf-hub:timm/ViT-SO400M-14-SigLIP2-384"
        print(f"[SigLIP2] Load model cho query encoding...")
        _siglip_model, _siglip_preprocess = open_clip.create_model_from_pretrained(MODEL_NAME)
        _siglip_model = _siglip_model.to(config.DEVICE).eval()
        _siglip_tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        print("[SigLIP2] Model ready.")
    return _siglip_model, _siglip_tokenizer


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
        from google import genai
        _gemini_model = genai.Client(api_key=config.GEMINI_API_KEY)
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
    Ưu tiên: matched_text > caption cache > metadata tổng hợp > chuỗi rỗng.
    [P2 Fix] Không bao giờ trả về chuỗi rỗng nếu còn thông tin nào đó.
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

    # [P2] Fallback tổng hợp từ metadata: video_id + timestamp + text_source
    parts = []
    vid = c.get("video_id", "")
    if vid:
        parts.append(f"Video {vid}")
    ts = c.get("timestamp_sec")
    if ts is not None:
        parts.append(f"at {ts:.1f}s")
    src = c.get("text_source", "")
    if src:
        parts.append(f"source: {src}")
    if parts:
        return " | ".join(parts)

    return "video keyframe"


_query_expansion_cache: dict = {}

def expand_query_with_gemini(query: str) -> list[str]:
    """
    [P9] Query Expansion: dùng Gemini sinh 2-3 cách diễn đạt khác nhau
    cùng ý nghĩa với query gốc để tăng recall cho BM25 và BGE-M3.
    Trả về list gồm query gốc + các bản mở rộng.
    Có cache để không gọi API nhiều lần cho cùng 1 query.
    """
    if not config.GEMINI_API_KEY:
        return [query]
    if query in _query_expansion_cache:
        return _query_expansion_cache[query]

    try:
        client = get_gemini_model()
        prompt = (
            f'You are a search query expansion expert for a Vietnamese video retrieval system.\n'
            f'Given the search query below, generate 2 alternative Vietnamese phrasings that '
            f'mean the same thing but use different words/synonyms.\n'
            f'Query: "{query}"\n\n'
            f'Output ONLY a valid JSON array of 2 strings (no markdown, no explanation).\n'
            f'Example: ["alt phrasing 1", "alt phrasing 2"]'
        )
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt
        )
        raw = (response.text or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        import json as _json
        expansions = _json.loads(raw)
        if isinstance(expansions, list):
            result = [query] + [e for e in expansions if isinstance(e, str) and e != query]
            _query_expansion_cache[query] = result
            print(f"[QueryExpansion] '{query}' → {result}")
            return result
    except Exception as e:
        print(f"[QueryExpansion] Lỗi ({e}). Dùng query gốc.")

    _query_expansion_cache[query] = [query]
    return [query]


def translate_query_for_clip(query):
    """
    Dịch query tiếng Việt sang tiếng Anh.
    Chiến lược: langid detect → Gemini API → NLLB-200 local fallback → query gốc.
    """
    lang = detect_language(query)
    if lang == "en":
        return query

    if config.GEMINI_API_KEY:
        try:
            client = get_gemini_model()
            from google import genai
            prompt = (
                f'Translate the following Vietnamese search query to English. '
                f'Output ONLY the translated English text, nothing else.\n'
                f'Query: "{query}"'
            )
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt
            )
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


def search_siglip(query, top_k=None, _pre_translated=None):
    """
    [SigLIP2] Encode query bằng SigLIP2-SO400M text encoder → search SigLIP2 FAISS Index.
    Chỉ chạy nếu index đã được build (build_siglip_index.py).
    """
    load_indices()
    if _siglip_index is None:
        return []  # Index chưa build → bỏ qua nhánh này

    top_k = top_k or config.TOP_K_RETRIEVE
    model, tokenizer = get_siglip_model()

    clip_query = _pre_translated if _pre_translated is not None else translate_query_for_clip(query)

    with torch.no_grad():
        tokens = tokenizer([clip_query]).to(config.DEVICE)
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        vec = feat.cpu().numpy().astype("float32")

    scores, idxs = _siglip_index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        meta = _siglip_id_map[idx]
        img_path = meta.get("path", "")
        if not img_path:
            img_path = _get_btc_image_path(meta["video_id"], meta["frame_idx"])
        results.append({
            "video_id":      meta["video_id"],
            "frame_idx":     meta["frame_idx"],
            "timestamp_sec": meta["timestamp_sec"],
            "path":          img_path,
            "score":         float(score),
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


def fuse_rrf(image_results, text_results, bm25_results=None, siglip_results=None, k=None, top_k=None):
    """
    Reciprocal Rank Fusion (RRF) cho 4 nhánh: CLIP + SigLIP2 + BGE-M3 + BM25.
    SigLIP2 chỉ được dùng nếu siglip_results không rỗng (index đã build).
    """
    k = k or config.RRF_K
    top_k = top_k or config.TOP_K_RETRIEVE

    rrf_scores = {}
    item_cache = {}

    # CLIP ViT-B/32 (BTC) — weight 3.0
    for rank, item in enumerate(image_results):
        key = _key(item)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 3.0 / (k + rank + 1)
        item_cache.setdefault(key, item)

    # SigLIP2-SO400M (self-encoded) — weight 2.5 (chất lượng cao hơn CLIP ViT-B/32)
    for rank, item in enumerate(siglip_results or []):
        key = _key(item)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 2.5 / (k + rank + 1)
        item_cache.setdefault(key, item)

    # BGE-M3 text — weight 1.0
    for rank, item in enumerate(text_results):
        key = _key(item)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in item_cache or "matched_text" not in item_cache[key]:
            item_cache[key] = item

    # BM25 sparse — weight 0.5
    for rank, item in enumerate(bm25_results or []):
        key = _key(item)
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

    fused_vals = [c.get("fused_score", 0.0) for c in candidates]
    lo, hi = (min(fused_vals), max(fused_vals)) if fused_vals else (0, 1)
    rng = (hi - lo) or 1e-6

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
        norm_fused = (c.get("fused_score", 0.0) - lo) / rng
        c["final_score"] = 0.5 * norm_fused + 0.5 * float(s)

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

    client = get_gemini_model()
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

    def _score_single(c):
        try:
            img_path = c.get("path", "")
            if not img_path or not os.path.exists(img_path):
                c["verify_score"] = 0
                return
            from google import genai as _genai
            from google.genai import types as _gtypes
            image_bytes = open(img_path, "rb").read()
            ext = os.path.splitext(img_path)[1].lower()
            mime = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png'}.get(ext, 'image/jpeg')
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[
                    _gtypes.Part.from_bytes(data=image_bytes, mime_type=mime),
                    prompt_template.format(query=query)
                ]
            )
            raw = (response.text or "").strip()
            try:
                import json as _json
                score = int(_json.loads(raw).get("score", 0))
            except Exception:
                digits = "".join(ch for ch in raw if ch.isdigit())
                score = int(digits[:2]) if digits else 0
            c["verify_score"] = max(0, min(10, score))
        except Exception as e:
            print(f"[LỖI] Gemini verify lỗi ở {c.get('path', 'unknown')}: {e}")
            c["verify_score"] = None

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(_score_single, candidates)

    def sort_key(c):
        has_verify = c.get("verify_score") is not None
        return (has_verify, c.get("verify_score") if has_verify else 0, c.get("final_score", c.get("fused_score", 0)))

    return sorted(candidates, key=sort_key, reverse=True)


# =================================================================
# HÀM TỔNG HỢP
# =================================================================
def full_search(query, fusion_method=None, do_verify=False, top_k=None, skip_rerank=False, use_expansion=False):
    """
    Chạy toàn bộ pipeline: search_image + search_text + search_bm25
    → fusion → rerank → (tuỳ chọn) verify.
    
    use_expansion=True: Bật Query Expansion (P9) cho BM25 để tăng Recall.
    """
    fusion_method = fusion_method or config.FUSION_METHOD
    top_k = top_k or config.TOP_K_RETRIEVE

    # Dịch 1 lần duy nhất, dùng chung cho cả SigLIP2 và BGE-M3
    lang = detect_language(query)
    if lang != "en":
        translated_query = translate_query_for_clip(query)
        print(f"[full_search] '{query}' → '{translated_query}' (SigLIP2 + BGE-M3)")
    else:
        translated_query = query

    # Nhánh ảnh: dùng bản đã dịch (tránh gọi API lần 2)
    image_results = search_image(query, top_k=top_k, _pre_translated=translated_query)
    # Nhánh text dense: dùng bản đã dịch (BGE-M3 hiểu tiếng Anh tốt hơn Việt không dấu)
    text_results = search_text(translated_query, top_k=top_k)
    
    # [P9] Query Expansion cho BM25 (exact keyword matching hưởng lợi nhiều nhất)
    if use_expansion:
        expanded_queries = expand_query_with_gemini(query)
    else:
        expanded_queries = [query]
    
    all_bm25 = []
    for eq in expanded_queries:
        partial = search_bm25(eq, top_k=top_k // len(expanded_queries) + 10)
        all_bm25.extend(partial)
    # Dedup theo (video_id, frame_idx), giữ điểm cao nhất
    bm25_seen = {}
    for r in all_bm25:
        k = (r["video_id"], r["frame_idx"])
        if k not in bm25_seen or r["score"] > bm25_seen[k]["score"]:
            bm25_seen[k] = r
    bm25_results = sorted(bm25_seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    # [SigLIP2] Nhánh thứ 4 — tự động bỏ qua nếu chưa build index
    siglip_results = search_siglip(query, top_k=top_k, _pre_translated=translated_query)
    if siglip_results:
        print(f"[full_search] SigLIP2: {len(siglip_results)} kết quả")

    if fusion_method == "rrf":
        fused = fuse_rrf(image_results, text_results, bm25_results=bm25_results,
                         siglip_results=siglip_results, top_k=top_k)
    else:
        fused = fuse_weighted(image_results, text_results, top_k=top_k)

    if skip_rerank:
        return fused, fused, None

    # CrossEncoder: query gốc (multilingual, hiểu tiếng Việt có dấu + tiếng Anh)
    reranked = rerank(query, fused, top_k=config.TOP_K_RERANK)
    verified = verify_with_gemini(query, reranked) if do_verify else None

    return fused, reranked, verified
