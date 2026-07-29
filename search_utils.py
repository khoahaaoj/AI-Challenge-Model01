"""
search_utils.py
=================
BƯỚC 3 của pipeline: TÌM KIẾM. File này CHỈ chứa logic xử lý (không có UI),
UI nằm ở app.py (Streamlit) và gọi vào các hàm ở đây.

Luồng xử lý đầy đủ cho 1 câu query (tiếng Việt hoặc tiếng Anh) - xem full_search():

    query
      ├─► [CLIP] Dịch Vi->En (Gemini) -> text encoder -> Image FAISS -> top 50 (nội dung ảnh)
      └─► [BGE-M3] encoder -> Text FAISS -> top 50 (caption Qwen2-VL / OCR / transcript)
                    │
                    ▼
        Hybrid Fusion (RRF hoặc weighted sum) ─► top 50 hợp nhất (loại trùng theo keyframe)
                    │
                    ▼
        CrossEncoder rerank (bge-reranker-v2-m3) ─► top 10
          (image-only candidates dùng caption cache làm fallback passage)
                    │
                    ▼
        (tuỳ chọn) Gemini verify lại bằng ẢNH THẬT ─► sắp xếp lại top 10

Các model được load 1 LẦN DUY NHẤT (lazy singleton qua biến global) và tái sử dụng
cho mọi query -> tránh phải load lại model (rất chậm) mỗi lần người dùng tìm kiếm.

Ghi chú version (kiểm tra ngày 2026-07-28):
  FlagEmbedding==1.4.0 + transformers==4.57.6 tương thích tốt, không cần patch.
"""
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
_clip_preprocess = None  # SigLIP2: lưu preprocess để dùng lại khi encode ảnh (verify)
_bge_model = None
_reranker_model = None
_gemini_model = None
_bm25_data = None        # BM25 sparse index (lazy load từ pickle)

_image_index = None
_image_id_map = None
_text_index = None
_text_id_map = None
_keyframe_meta = None
_caption_cache = None   # cache caption dùng làm fallback passage cho CrossEncoder


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
    """Load SigLIP2 model + tokenizer qua hf-hub (open_clip >= 2.31.0)."""
    global _clip_model, _clip_tokenizer, _clip_preprocess
    if _clip_model is None:
        import open_clip
        # SigLIP2 dùng create_model_from_pretrained (trả về 2-tuple: model, preprocess)
        # thay vì create_model_and_transforms (3-tuple) của CLIP OpenAI cũ.
        model, preprocess = open_clip.create_model_from_pretrained(config.CLIP_MODEL_NAME)
        _clip_model = model.to(config.DEVICE).eval()
        _clip_preprocess = preprocess
        _clip_tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
    return _clip_model, _clip_tokenizer


def get_bge_model():
    global _bge_model
    if _bge_model is None:
        from FlagEmbedding import BGEM3FlagModel
        # Chạy BGE-M3 trên CPU để tiết kiệm ~2.2GB VRAM cho GPU 4GB.
        # Vì chỉ encode 1 câu query ngắn lúc search nên CPU xử lý cực nhanh (0.1s).
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
        try:
            # SDK mới (khuyến dùng từ 2025+)
            from google import genai
            _gemini_model = genai.Client(api_key=config.GEMINI_API_KEY)
        except ImportError:
            # Fallback SDK cũ nếu chưa được upgrade
            import google.generativeai as genai  # noqa: deprecated
            genai.configure(api_key=config.GEMINI_API_KEY)
            _gemini_model = genai.GenerativeModel(config.GEMINI_MODEL)
    return _gemini_model


def _load_caption_cache():
    """Load caption cache vào bộ nhớ (lazy, chỉ load 1 lần) để dùng làm fallback passage cho CrossEncoder."""
    global _caption_cache
    if _caption_cache is None:
        if os.path.exists(config.CAPTION_CACHE):
            _caption_cache = _load_json(config.CAPTION_CACHE)
        else:
            _caption_cache = {}
    return _caption_cache


def _get_passage_for_candidate(c):
    """
    Lấy passage (đoạn text đại diện) tốt nhất cho 1 candidate để đưa vào CrossEncoder.

    Thứ tự ưu tiên:
      1. matched_text (caption/ocr/transcript đã khớp từ nhánh BGE-M3) — tốt nhất
      2. Caption của keyframe từ cache (fallback cho candidate image-only từ CLIP)
      3. Chuỗi rỗng (CrossEncoder sẽ cho điểm thấp, nhưng không bị crash)

    Lý do cần fallback: candidate chỉ đến từ nhánh CLIP (search_image) không có
    matched_text. Nếu để chuỗi rỗng, CrossEncoder luôn cho điểm ~0 → candidate ảnh
    luôn bị đẩy xuống cuối dù ảnh rất khớp với query → bug logic nghiêm trọng.
    """
    # Ưu tiên 1: matched_text từ nhánh text
    text = (c.get("matched_text") or "").strip()
    if text and text != "Khung hình video":
        return text

    # Ưu tiên 2: caption từ cache (fallback cho CLIP-only candidates)
    caption_cache = _load_caption_cache()
    path = c.get("path", "")
    if path:
        caption = (caption_cache.get(path) or "").strip()
        if caption and caption != "Khung hình video":
            return caption

    # Fallback cuối: chuỗi rỗng
    return ""


def translate_query_for_clip(query):
    """
    Dịch query tiếng Việt sang tiếng Anh trước khi đưa vào CLIP text encoder.

    CLIP ViT-B-32 (OpenAI) được train hoàn toàn bằng tiếng Anh. Nếu người dùng
    gõ query tiếng Việt ("người đàn ông áo đỏ"), CLIP text encoder sẽ gần như
    không hiểu → embedding sai → kết quả image search kém.

    Giải pháp: dịch sang tiếng Anh trước ("a man in a red shirt") → CLIP hiểu tốt.
    Dùng Gemini API (rất rẻ, ~0.00001$/call) hoặc bỏ qua nếu chưa cấu hình key.

    Nếu query đã là tiếng Anh hoặc Gemini lỗi → trả về query gốc (graceful fallback).
    """
    if not config.GEMINI_API_KEY:
        return query  # Chưa có key → dùng query gốc

    # Heuristic: nếu query toàn ASCII thì có thể đã là tiếng Anh → không cần dịch
    try:
        if query.isascii():
            return query
    except Exception:
        pass

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
            print(f"[CLIP] Dịch query: '{query}' → '{translated}'")
            return translated
    except Exception as e:
        print(f"[CLIP] Dịch query thất bại ({e}), dùng query gốc.")

    return query


# =================================================================
# 1) SEARCH ẢNH: CLIP text encoder -> FAISS Image Index
# =================================================================
def search_image(query, top_k=None):
    """
    Encode câu query bằng CLIP text encoder rồi search Image FAISS Index.

    QUAN TRỌNG: CLIP ViT-B-32 chỉ hiểu tiếng Anh. Với query tiếng Việt,
    hàm này sẽ tự động dịch sang tiếng Anh (nếu có GEMINI_API_KEY) trước
    khi encode, giúp tăng độ chính xác nhánh image search đáng kể.
    """
    load_indices()
    top_k = top_k or config.TOP_K_RETRIEVE
    model, tokenizer = get_clip_model()

    # Dịch query tiếng Việt sang tiếng Anh để CLIP text encoder hiểu đúng ngữ nghĩa.
    # CLIP ViT-B-32 chỉ hiểu tiếng Anh → query tiếng Việt gốc sẽ cho embedding sai.
    clip_query = translate_query_for_clip(query)

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
# 3) SEARCH BM25: sparse keyword retrieval (bổ sung exact match)
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
    Tìm kiếm bằng BM25 Okapi trên corpus caption + OCR + transcript.
    Chạy hoàn toàn trên CPU, không cần VRAM.

    Mạnh hơn BGE-M3 với: tên riêng, tên địa danh, tên thương hiệu,
    số hiệu văn bản, thưật ngữ chuyên ngành xuất hiện nguyên vẹn.
    Bổ sung vào RRF fusion như nhánh thứ 3 độc lập.
    """
    top_k = top_k or config.TOP_K_RETRIEVE
    data = get_bm25()
    if data is None:
        return []  # Chưa build BM25 index -> bỏ qua, không lỗi

    bm25, records = data["bm25"], data["records"]

    # Dùng cùng tokenizer logic với lúc build (simple split đủ cho query ngắn)
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idxs = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_idxs:
        if scores[idx] <= 0:
            break  # Chỉ giữ kết quả có score thực sự (>0)
        r = records[int(idx)]
        frame_idx = r["frame_idx"]

        if frame_idx is None:
            nearest = _nearest_keyframe(r["video_id"], r["timestamp_sec"])
            if nearest is None:
                continue
            frame_idx = nearest["frame_idx"]
            path = nearest["path"]
        else:
            path = os.path.join(config.KEYFRAME_DIR, r["video_id"], f"{frame_idx:08d}.jpg")

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
# 4) HYBRID FUSION: gộp 3 danh sách (image + dense text + BM25) thành 1
# =================================================================
def _key(item):
    """Khóa định danh 1 keyframe duy nhất, dùng để gộp kết quả trùng nhau giữa 2 nhánh."""
    return (item["video_id"], item["frame_idx"])


def fuse_rrf(image_results, text_results, bm25_results=None, k=None, top_k=None):
    """
    Reciprocal Rank Fusion (RRF) cho 3 nhánh: CLIP (image) + BGE-M3 (dense) + BM25 (sparse).

    score(item) = Σ w_source / (k + rank_trong_source)

    RRF chỉ dựa vào THỨ HẠNG, không quan tâm thầng điểm gốc → phù hợp
    để kết hợp các nguồn điểm không cùng bản chất (cosine-CLIP vs cosine-BGE vs BM25).
    bm25_results=None: tương thích ngược (chưa build BM25 index -> vẫn chạy 2 nhánh cũ).
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
        # Ưu tiên giữ bản ghi có "matched_text" (cần cho bước rerank sau)
        if key not in item_cache or "matched_text" not in item_cache[key]:
            item_cache[key] = item

    # Nhánh BM25 (sparse) — boost kết quả khớp keyword chính xác
    for rank, item in enumerate(bm25_results or []):
        key = _key(item)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
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
    # FIX BUG: dùng _get_passage_for_candidate() thay vì c.get("matched_text", "") trực tiếp.
    # Lý do: candidate chỉ từ nhánh CLIP (image-only) không có matched_text → nếu để ""
    # CrossEncoder luôn cho điểm ~0 → kết quả CLIP bị đẩy xuống cuối dù ảnh rất khớp.
    # _get_passage_for_candidate() tra caption cache làm fallback để CrossEncoder có
    # đủ ngữ cảnh đánh giá candidate từ cả 2 nhánh một cách công bằng.
    pairs = [[query, _get_passage_for_candidate(c)] for c in candidates]
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
    # Prompt có rubric rõ ràng + yêu cầu JSON → Gemini chấm điểm nhất quán hơn,
    # dễ parse hơn cách extract digit thô bạo trước.
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
            image = Image.open(c["path"]).convert("RGB")
            response = model.generate_content([prompt_template.format(query=query), image])
            raw = (response.text or "").strip()
            # Ưu tiên parse JSON {"score": X}; fallback extract digit nếu format lệch
            try:
                import json as _json
                score = int(_json.loads(raw).get("score", 0))
            except Exception:
                digits = "".join(ch for ch in raw if ch.isdigit())
                score = int(digits[:2]) if digits else 0  # lấy tối đa 2 digit (0-10)
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
    Chạy toàn bộ luồng: search_image + search_text + search_bm25
    -> fusion -> rerank -> (tuỳ chọn) verify.

    Trả về tuple:
        fused     : top 50 sau khi hợp nhất 3 nhánh (chưa rerank)
        reranked  : top 10 sau CrossEncoder rerank
        verified  : top 10 sau Gemini verify (None nếu do_verify=False)
    """
    fusion_method = fusion_method or config.FUSION_METHOD

    image_results = search_image(query, top_k=config.TOP_K_RETRIEVE)
    text_results  = search_text(query,  top_k=config.TOP_K_RETRIEVE)
    bm25_results  = search_bm25(query,  top_k=config.TOP_K_RETRIEVE)  # [] nếu chưa build

    if fusion_method == "rrf":
        fused = fuse_rrf(
            image_results, text_results,
            bm25_results=bm25_results,
            top_k=config.TOP_K_RETRIEVE,
        )
    else:
        # weighted fusion không hỗ trợ BM25 (thước đo BM25 khác bản chất); dùng RRF nếu cần BM25
        fused = fuse_weighted(image_results, text_results, top_k=config.TOP_K_RETRIEVE)

    reranked = rerank(query, fused, top_k=config.TOP_K_RERANK)

    verified = verify_with_gemini(query, reranked) if do_verify else None

    return fused, reranked, verified
