"""
search_utils.py
=================
BƯỚC 3 của pipeline: TÌM KIẾM. File này CHỈ chứa logic xử lý (không có UI),
UI nằm ở app.py (Streamlit) và gọi vào các hàm ở đây.

Luồng xử lý đầy đủ cho 1 câu query (tiếng Việt hoặc tiếng Anh) - xem full_search():

    query
      ├─► CLIP text encoder  ─► search Image FAISS Index  ─► top 50 (theo NỘI DUNG ẢNH)
      └─► BGE-M3 encoder     ─► search Text  FAISS Index  ─► top 50 (theo CAPTION/OCR/TRANSCRIPT)
                    │
                    ▼
        Hybrid Fusion (RRF hoặc weighted sum) ─► top 50 hợp nhất (loại trùng theo keyframe)
                    │
                    ▼
        CrossEncoder rerank (bge-reranker-v2-m3) ─► top 10 chính xác hơn
                    │
                    ▼
        (tuỳ chọn) Gemini/VLM verify lại bằng ẢNH THẬT ─► sắp xếp lại top 10 theo độ tin cậy

Các model được load 1 LẦN DUY NHẤT (lazy singleton qua biến global) và tái sử dụng
cho mọi query -> tránh phải load lại model (rất chậm) mỗi lần người dùng tìm kiếm.
"""
import transformers.tokenization_utils_base

if not hasattr(transformers.tokenization_utils_base.PreTrainedTokenizerBase, "prepare_for_model"):
    def prepare_for_model(self, *args, **kwargs):
        return self._encode_plus(*args, **kwargs)
    transformers.tokenization_utils_base.PreTrainedTokenizerBase.prepare_for_model = prepare_for_model

import os
import json
import numpy as np
import faiss
import torch
from PIL import Image

import config


# =================================================================
# BIẾN GLOBAL (cache model + index trong bộ nhớ, load 1 lần)
# =================================================================
_clip_model = None
_clip_tokenizer = None
_bge_model = None
_reranker_model = None
_gemini_model = None

_image_index = None
_image_id_map = None
_text_index = None
_text_id_map = None
_keyframe_meta = None


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

    if _keyframe_meta is None and os.path.exists(config.KEYFRAME_META_CACHE):
        _keyframe_meta = _load_json(config.KEYFRAME_META_CACHE)


def get_clip_model():
    """Load CLIP model + tokenizer (dùng để encode cả ảnh lúc build lẫn text query lúc search)."""
    global _clip_model, _clip_tokenizer
    if _clip_model is None:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED
        )
        _clip_model = model.to(config.DEVICE).eval()
        _clip_tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
    return _clip_model, _clip_tokenizer


def get_bge_model():
    global _bge_model
    if _bge_model is None:
        from FlagEmbedding import BGEM3FlagModel
        _bge_model = BGEM3FlagModel(config.BGE_MODEL_NAME, use_fp16=(config.DEVICE == "cuda"))
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


# =================================================================
# 1) SEARCH ẢNH: CLIP text encoder -> FAISS Image Index
# =================================================================
def search_image(query, top_k=None):
    """
    Encode câu query (vi/en) bằng CLIP text encoder rồi search Image FAISS Index.

    LƯU Ý QUAN TRỌNG: CLIP gốc (OpenAI) được train chủ yếu bằng dữ liệu tiếng Anh,
    nên với query tiếng Việt, nhánh này thường kém chính xác hơn nhánh BGE-M3.
    Đây chính là lý do pipeline luôn kết hợp (fusion) cả 2 nhánh thay vì chỉ dùng CLIP.
    """
    load_indices()
    top_k = top_k or config.TOP_K_RETRIEVE
    model, tokenizer = get_clip_model()

    with torch.no_grad():
        tokens = tokenizer([query]).to(config.DEVICE)
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        vec = feat.cpu().numpy().astype("float32")

    scores, idxs = _image_index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        meta = _image_id_map[idx]
        results.append({
            "video_id": meta["video_id"],
            "frame_idx": meta["frame_idx"],
            "timestamp_sec": meta["timestamp_sec"],
            "path": meta["path"],
            "score": float(score),
        })
    return results


# =================================================================
# 2) SEARCH TEXT: BGE-M3 -> FAISS Text Index
# =================================================================
def _nearest_keyframe(video_id, timestamp_sec):
    """
    Record transcript chỉ có timestamp (không gắn sẵn 1 keyframe cụ thể lúc build).
    Hàm này tìm keyframe GẦN NHẤT về mặt thời gian trong cùng video, để khi hiển thị
    kết quả transcript, người dùng vẫn thấy được 1 ảnh đại diện cụ thể.
    """
    if _keyframe_meta is None or video_id not in _keyframe_meta:
        return None
    metas = _keyframe_meta[video_id]
    if not metas:
        return None
    return min(metas, key=lambda m: abs(m["timestamp_sec"] - timestamp_sec))


def search_text(query, top_k=None):
    """
    Encode query bằng BGE-M3 (đa ngôn ngữ, hỗ trợ tốt cả vi và en) rồi search Text FAISS Index
    (index này gồm cả caption + ocr + transcript đã gộp chung lúc build_index.py).

    Mỗi kết quả trả về luôn được gắn về 1 keyframe CỤ THỂ:
      - record nguồn caption/ocr: dùng thẳng frame_idx đã lưu
      - record nguồn transcript : suy ra keyframe gần nhất theo thời gian (_nearest_keyframe)
    """
    load_indices()
    if _text_index is None:
        return []  # chưa build được text index (vd: chưa cấu hình GEMINI_API_KEY) -> trả rỗng, không lỗi

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
            path = nearest["path"]
        else:
            # dựng lại path ảnh từ video_id + frame_idx, đúng quy tắc đặt tên ở extract_keyframes.py
            path = os.path.join(config.KEYFRAME_DIR, record["video_id"], f"{frame_idx:08d}.jpg")

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
# 3) HYBRID FUSION: gộp 2 danh sách (image + text) thành 1 danh sách duy nhất
# =================================================================
def _key(item):
    """Khóa định danh 1 keyframe duy nhất, dùng để gộp kết quả trùng nhau giữa 2 nhánh."""
    return (item["video_id"], item["frame_idx"])


def fuse_rrf(image_results, text_results, k=None, top_k=None):
    """
    Reciprocal Rank Fusion (RRF): score(item) = Σ 1 / (k + rank_trong_từng_danh_sách).
    Ưu điểm: chỉ dựa vào THỨ HẠNG (rank), không quan tâm thang điểm gốc -> rất phù hợp
    để kết hợp 2 nguồn điểm số vốn không cùng bản chất như cosine-CLIP và cosine-BGE-M3
    (2 model khác nhau, phân bố điểm số khác nhau, so trực tiếp giá trị số là không hợp lý).
    """
    k = k or config.RRF_K
    top_k = top_k or config.TOP_K_RETRIEVE

    rrf_scores = {}
    item_cache = {}

    for rank, item in enumerate(image_results):
        key = _key(item)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        item_cache.setdefault(key, item)

    for rank, item in enumerate(text_results):
        key = _key(item)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        # Ưu tiên giữ bản ghi có "matched_text" (cần cho bước rerank sau) khi item đã tồn tại
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
    """
    Weighted sum: chuẩn hóa min-max điểm mỗi nhánh về [0,1] rồi cộng có trọng số.
    Dễ hiểu/dễ chỉnh hơn RRF (đổi trọng số trực quan), nhưng nhạy cảm với outlier điểm số
    hơn RRF. Dùng làm phương án thay thế để so sánh, không nhất thiết phải tốt hơn RRF.
    """
    image_weight = image_weight if image_weight is not None else config.IMAGE_WEIGHT
    text_weight = text_weight if text_weight is not None else config.TEXT_WEIGHT
    top_k = top_k or config.TOP_K_RETRIEVE

    def normalize(items):
        if not items:
            return {}
        scores = [it["score"] for it in items]
        lo, hi = min(scores), max(scores)
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
# 4) CROSS-ENCODER RERANK (bge-reranker-v2-m3, đa ngôn ngữ)
# =================================================================
def rerank(query, candidates, top_k=None):
    """
    Bi-encoder (CLIP/BGE-M3 ở bước fusion) encode query và passage RIÊNG BIỆT rồi so cosine
    -> nhanh nhưng kém chính xác. CrossEncoder encode CẢ CẶP (query, passage) CÙNG LÚC
    -> chậm hơn nhiều nhưng chính xác hơn hẳn -> chỉ áp dụng cho top 50 sau fusion (không thể
    áp dụng cho toàn bộ dữ liệu vì quá chậm).

    passage đại diện cho mỗi candidate = "matched_text" (caption/ocr/transcript khớp nhất).
    Nếu candidate chỉ đến từ nhánh ảnh (không có matched_text) thì để chuỗi rỗng.
    """
    top_k = top_k or config.TOP_K_RERANK
    if not candidates:
        return []

    reranker = get_reranker_model()
    pairs = [[query, c.get("matched_text", "") or ""] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)  # sigmoid-normalize về khoảng 0..1

    if isinstance(scores, float):  # chỉ có 1 candidate -> compute_score trả về số đơn thay vì list
        scores = [scores]

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


# =================================================================
# 5) VERIFY BẰNG GEMINI (VLM) - bước lọc cuối cùng, nhìn ảnh THẬT thay vì chỉ vector
# =================================================================
def verify_with_gemini(query, candidates):
    """
    Với mỗi candidate đã qua rerank, gửi ẢNH KEYFRAME THẬT + câu query cho Gemini,
    yêu cầu chấm điểm mức độ khớp (0-10). Bước này giúp bắt các trường hợp "giống nhau
    về embedding nhưng sai về nội dung" mà CLIP/BGE-M3/CrossEncoder không phân biệt được,
    vì đây là model duy nhất trong pipeline thực sự "NHÌN" lại ảnh gốc kết hợp suy luận ngôn ngữ.

    Nếu chưa cấu hình GEMINI_API_KEY, hàm này bỏ qua (giữ nguyên thứ tự rerank).
    """
    if not config.GEMINI_API_KEY:
        for c in candidates:
            c["verify_score"] = None
        return candidates

    model = get_gemini_model()
    prompt_template = (
        "You are verifying video retrieval results for a video search engine.\n"
        "Query (may be in Vietnamese or English): \"{query}\"\n"
        "Does this image match the query? "
        "Answer with ONLY a single integer from 0 to 10 (10 = perfect match, 0 = completely unrelated). "
        "No explanation, just the number."
    )

    for c in candidates:
        try:
            image = Image.open(c["path"]).convert("RGB")
            response = model.generate_content([prompt_template.format(query=query), image])
            digits = "".join(ch for ch in (response.text or "") if ch.isdigit())
            score = int(digits) if digits else 0
            c["verify_score"] = max(0, min(10, score))
        except Exception as e:
            print(f"[LỖI] Gemini verify lỗi ở {c['path']}: {e}")
            c["verify_score"] = None

    def sort_key(c):
        # Ưu tiên verify_score (nếu Gemini chấm được), fallback rerank_score khi Gemini lỗi
        has_verify = c["verify_score"] is not None
        return (has_verify, c["verify_score"] if has_verify else 0, c["rerank_score"])

    return sorted(candidates, key=sort_key, reverse=True)


# =================================================================
# HÀM TỔNG HỢP: chạy toàn bộ pipeline tìm kiếm cho 1 query
# =================================================================
def full_search(query, fusion_method=None, do_verify=False):
    """
    Chạy toàn bộ luồng: search_image + search_text -> fusion -> rerank -> (tuỳ chọn) verify.

    Trả về tuple:
        fused     : top 50 sau khi hợp nhất 2 nhánh (chưa rerank)
        reranked  : top 10 sau CrossEncoder rerank
        verified  : top 10 sau Gemini verify (None nếu do_verify=False)
    """
    fusion_method = fusion_method or config.FUSION_METHOD

    image_results = search_image(query, top_k=config.TOP_K_RETRIEVE)
    text_results = search_text(query, top_k=config.TOP_K_RETRIEVE)

    if fusion_method == "rrf":
        fused = fuse_rrf(image_results, text_results, top_k=config.TOP_K_RETRIEVE)
    else:
        fused = fuse_weighted(image_results, text_results, top_k=config.TOP_K_RETRIEVE)

    reranked = rerank(query, fused, top_k=config.TOP_K_RERANK)

    verified = verify_with_gemini(query, reranked) if do_verify else None

    return fused, reranked, verified
