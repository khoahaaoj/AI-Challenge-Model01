# AIC 2026 (Bảng A) — Multimedia Video Retrieval System

> Hệ thống tìm kiếm keyframe/video từ truy vấn văn bản tiếng Việt hoặc tiếng Anh,
> chạy **hoàn toàn local** trên GPU 4GB VRAM.

---

## Kiến Trúc Pipeline

### Giai đoạn 1 — Lập Chỉ Mục (Indexing, chạy offline 1 lần)

```
VIDEO ─┬─ PySceneDetect ─────────────────────────────► Keyframes (3 frame/scene: 25%, 50%, 75%)
       └─ Faster-Whisper medium (vi+en, VAD) ─────────► Transcript + timestamp

Keyframes ─┬─ SigLIP2 ViT-B-16 (image encoder) ──────────────────────────► FAISS Image Index (dim=768)
           │
           ├─ Qwen2-VL-2B-Instruct 4-bit NF4 (VLM caption, tiếng Việt) ──┐
           ├─ EasyOCR (vi + en) ──────────────────────────────────────────┼─ BGE-M3 ──► FAISS Text Index (dim=1024)
           └─ Transcript (Whisper) ───────────────────────────────────────┘              └─ BM25 Okapi ──► bm25_index.pkl
```

### Giai đoạn 2 — Tìm Kiếm (Search, real-time trên Streamlit)

```
QUERY (vi/en)
  │
  ├─[1]─ langid detect ngôn ngữ (local) → nếu không phải tiếng Anh:
  │         ├── Gemini API dịch Vi→En (ưu tiên, nhanh)
  │         └── NLLB-200 local fallback (khi mất mạng/Gemini lỗi)  ← MỚI
  │       └─ SigLIP2 text encoder (GPU) ──► FAISS Image Index ──► Top-50
  │
  ├─[2]─ BGE-M3 (CPU, tiết kiệm VRAM) ──► FAISS Text Index ──► Top-50
  │
  └─[3]─ BM25 Okapi (CPU, zero VRAM) ──► bm25_index.pkl ──► Top-50
                │
                ▼
    RRF Fusion k=60 (3 nhánh, dedup theo keyframe) ──► Top-50 hợp nhất
                │
                ▼
    CrossEncoder bge-reranker-v2-m3 ──► Top-10
                │
                ▼ (tuỳ chọn, checkbox trong UI)
    Gemini VLM Verify (ảnh thật + rubric JSON 0-10) ──► Kết quả cuối
```

---

## Model & Thông Số Kỹ Thuật

| Thành phần | Model | Chạy trên | VRAM/RAM |
|---|---|---|---|
| Scene detection | PySceneDetect ContentDetector (threshold=27.0) | CPU | — |
| Speech-to-text | faster-whisper `medium`, float16, VAD | GPU | ~1.5GB VRAM |
| Image encoder | **SigLIP2 ViT-B-16** (open_clip hf-hub) | GPU | ~400MB VRAM |
| VLM caption | Qwen2-VL-2B-Instruct **4-bit NF4** (max_new_tokens=100) | GPU | ~2.0GB VRAM |
| OCR | EasyOCR vi+en | CPU/GPU | — |
| Text embedding | BGE-M3 dense (dim=1024) | **CPU** (tiết kiệm VRAM) | ~4GB RAM |
| Sparse retrieval | **BM25 Okapi** (rank-bm25) | CPU | <100MB RAM |
| Reranker | bge-reranker-v2-m3, fp16, normalize=True | CPU | ~2GB RAM |
| Language detect | **langid** (local, không cần API) | CPU | ~15MB RAM |
| Translation fallback | **NLLB-200-distilled-600M** (local) | CPU | ~2.4GB RAM |
| Translation/Verify | Gemini API (gemini-flash) | Cloud API | — |

> **Tổng VRAM khi search:** SigLIP2 ~400MB + CUDA overhead ~300MB ≈ **< 1GB** (an toàn với 4GB)

---

## Cài Đặt

```bash
# 1. Tạo virtualenv
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. Cài torch đúng bản CUDA (kiểm tra CUDA version bằng `nvidia-smi`)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Cài toàn bộ package
pip install -r requirements.txt

# 4. (Tuỳ chọn) Cải thiện BM25 với từ ghép tiếng Việt
# pip install underthesea
# ⚠️  Nếu cài underthesea sau khi đã build: xóa index/bm25_index.pkl → build lại

# 5. Cấu hình Gemini API key (tùy chọn nhưng khuyến nghị)
#    Lấy key miễn phí tại https://aistudio.google.com/apikey
echo 'GEMINI_API_KEY=your-key-here' > .env
```

> **Không có Gemini key?** Pipeline vẫn chạy đầy đủ 3 nhánh.
> Bước dịch query sẽ dùng **NLLB-200 local** (tự download ~2.4GB lần đầu).
> Verify cuối bị tắt.

---

## Chạy Pipeline

```bash
# ── Bước 0 ──────────────────────────────────────────────────────
# Copy video vào data/videos/  (hỗ trợ .mp4, .avi, .mkv, .mov, .webm)

# ── Bước 1 ──────────────────────────────────────────────────────
# Tách keyframe (3 frame/scene @ 25/50/75%) + transcript Whisper → cache JSON
python3 extract_keyframes.py

# ── Bước 2 ──────────────────────────────────────────────────────
# Build toàn bộ index (mỗi bước cache riêng, tự resume nếu bị gián đoạn):
#   (a) SigLIP2 → FAISS Image Index (dim=768)
#   (b) Qwen2-VL 4-bit → captions.json  (max_new_tokens=100)
#   (c) EasyOCR → ocr.json
#   (d) BGE-M3 → FAISS Text Index (dim=1024)
#   (e) BM25 Okapi → bm25_index.pkl  (lưu flag tokenizer để search khớp build)
python3 build_index.py

# ── Bước 3 ──────────────────────────────────────────────────────
# Khởi động giao diện tìm kiếm Streamlit
streamlit run app.py
```

Mỗi bước **cache riêng theo video/ảnh** — nếu bị gián đoạn, chạy lại đúng lệnh trên sẽ tự tiếp tục từ chỗ dang dở.

---

## Cấu Trúc Thư Mục

```
aic_retrieval/
├── config.py              # ⚙️  Cấu hình trung tâm (path, model name, tham số)
├── extract_keyframes.py   # 📹  Bước 1: PySceneDetect + Faster-Whisper
├── build_index.py         # 🔨  Bước 2: SigLIP2 / Qwen2-VL / EasyOCR / BGE-M3 / BM25
├── search_utils.py        # 🔍  Bước 3: search + RRF fusion + rerank + verify
├── tokenizer_utils.py     # 🔤  Module dùng chung: language detect + BM25 tokenizer
├── app.py                 # 🖥️  UI Streamlit (search, video player, export nộp bài)
├── requirements.txt
├── .env                   # GEMINI_API_KEY (không commit lên git)
│
├── data/
│   ├── videos/            # ← bỏ video gốc vào đây
│   └── keyframes/         # ảnh keyframe tự động sinh ra (video_id/XXXXXXXX.jpg)
│
├── cache/
│   ├── keyframe_meta.json # metadata keyframe (video_id, frame_idx, timestamp, path)
│   ├── transcripts.json   # transcript Whisper theo video
│   ├── captions.json      # caption Qwen2-VL theo keyframe (key = path ảnh)
│   └── ocr.json           # text OCR theo keyframe (key = path ảnh)
│
└── index/
    ├── image_index.faiss  # FAISS Image Index (SigLIP2, dim=768, cosine via IP)
    ├── image_id_map.json  # vị trí FAISS → metadata keyframe
    ├── text_index.faiss   # FAISS Text Index (BGE-M3, dim=1024, cosine via IP)
    ├── text_id_map.json   # vị trí FAISS → record text (caption/ocr/transcript)
    └── bm25_index.pkl     # BM25 Sparse Index + flag "used_underthesea"
```

---

## Tính Năng Giao Diện (app.py)

| Tính năng | Mô tả |
|---|---|
| **Search bar** | Gõ query tiếng Việt hoặc tiếng Anh |
| **Hybrid Fusion** | RRF (khuyến nghị) hoặc Weighted — chọn trong sidebar |
| **Gemini VLM Verify** | Checkbox tuỳ chọn — gửi ảnh thật cho Gemini chấm 0-10 |
| **Hiển thị kết quả** | Grid 2-5 cột, xem điểm RRF / Rerank / Gemini từng kết quả |
| **Text khớp** | Expand xem caption / OCR / transcript đã khớp query |
| **Video player** | Phát video gốc tại timestamp keyframe ±2s ngay trong UI |
| **Chọn & Export** | Tick keyframe đúng → tải JSON / CSV theo format nộp bài AIC |

---

## Chiến Lược Dịch Query (search_utils.py)

Thứ tự ưu tiên khi query không phải tiếng Anh:

```
1. langid.classify(query) → phát hiện ngôn ngữ local (không cần API)
   - Nếu là tiếng Anh → KHÔNG dịch
   - Mọi ngôn ngữ khác (vi, pt*, unknown...) → bước 2

2. Gemini API (nhanh, chất lượng cao)
   - Thành công → dùng kết quả dịch

3. NLLB-200-distilled-600M local (fallback khi Gemini lỗi/mất mạng)
   - Chạy CPU, không cần internet
   - Lần đầu tự download model ~2.4GB

4. Dùng query gốc + cảnh báo rõ ràng trong console
```

> \* langid đôi khi nhận nhầm tiếng Việt không dấu (`"nguoi dan ong ao do"`) là `pt` — đây là hành vi đã biết.
> Code xử lý an toàn: chỉ bỏ qua dịch khi langid **chắc chắn là `en`**, mọi ngôn ngữ khác đều đi qua bước dịch.

---

## Rebuild Một Phần

| Trường hợp | Cần làm |
|---|---|
| Thêm video mới | `extract_keyframes.py` → `build_index.py` (tự bỏ qua video/ảnh đã cache) |
| Đổi CLIP model | Xóa `index/image_index.faiss` + `image_id_map.json` → `build_index.py` |
| Rebuild BM25 (sau thêm video) | Xóa `index/bm25_index.pkl` → `build_index.py` |
| **Cài/gỡ underthesea** | Xóa `index/bm25_index.pkl` → `build_index.py` (**bắt buộc** — flag tokenizer thay đổi) |
| Rebuild text index (BGE-M3) | Xóa `index/text_index.faiss` + `text_id_map.json` → `build_index.py` |
| Redo caption (Qwen2-VL) | Xóa `cache/captions.json` → `build_index.py` |
| Redo transcript (Whisper) | Xóa `cache/transcripts.json` → `extract_keyframes.py` → `build_index.py` |

---

## Tham Số Tìm Kiếm (config.py)

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `TOP_K_RETRIEVE` | 50 | Số kết quả lấy từ mỗi nhánh trước fusion |
| `TOP_K_RERANK` | 10 | Số kết quả cuối sau CrossEncoder |
| `FUSION_METHOD` | `"rrf"` | `"rrf"` hoặc `"weighted"` |
| `RRF_K` | 60 | Hằng số k của RRF (chuẩn literature) |
| `KEYFRAME_POSITIONS` | `[0.25, 0.5, 0.75]` | Vị trí lấy frame trong mỗi scene (3 frame/scene) |
| `SCENE_THRESHOLD` | 27.0 | Ngưỡng ContentDetector (PySceneDetect) |
| `WHISPER_MODEL_SIZE` | `"medium"` | Đổi sang `"large-v3"` để tăng chất lượng tiếng Việt |
| `CLIP_BATCH_SIZE` | 16 | Batch size encode ảnh (tăng lên 32 nếu VRAM > 4GB) |
| `GEMINI_MODEL` | `"gemini-3.5-flash-lite"` | Model Gemini dùng cho dịch và verify |

---

## Troubleshooting

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `RuntimeError: Chưa có index/image_index.faiss` | Chưa chạy `build_index.py` | `python3 build_index.py` |
| `AssertionError` dim mismatch khi search | Đổi CLIP model nhưng chưa xóa index cũ | Xóa `index/image_index.faiss` + `image_id_map.json` → rebuild |
| BM25 recall thấp sau cài underthesea | Tokenizer build/search không khớp | Xóa `index/bm25_index.pkl` → rebuild (flag tự lưu vào pickle) |
| `ModuleNotFoundError: rank_bm25` | Chưa cài package | `pip install rank-bm25` |
| `ModuleNotFoundError: langid` | Chưa cài package | `pip install langid` |
| `FlagEmbedding` cài lỗi | Conflict version | `pip install -U FlagEmbedding transformers accelerate` |
| OOM khi build caption (Qwen2-VL) | Ảnh đầu vào quá lớn | Giảm `max_pixels` trong `build_captions()` ở `build_index.py` |
| NLLB download chậm lần đầu | Model ~2.4GB cần tải về | Chờ lần đầu; sau đó cache tại `~/.cache/huggingface/` |
| EasyOCR chậm lần đầu | Download model OCR ~200MB | Chờ lần đầu; cache tại `~/.EasyOCR/model/` |
| Gemini rate limit (HTTP 429) | Quá nhiều request | Giảm tần suất search, hoặc tắt Gemini verify |
| `[CLIP] ⚠️ Không thể dịch query` | Cả Gemini lẫn NLLB đều lỗi | Kiểm tra kết nối mạng hoặc cài lại `sentencepiece` |
