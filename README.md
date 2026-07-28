# AIC 2026 (Bảng A) — Multimedia Video Retrieval System

> Hệ thống tìm kiếm keyframe/video từ truy vấn văn bản tiếng Việt hoặc tiếng Anh,
> chạy hoàn toàn local trên GPU 4GB VRAM.

---

## Kiến Trúc Pipeline

### Giai đoạn 1 — Lập Chỉ Mục (Indexing, chạy offline 1 lần)

```
VIDEO ─┬─ PySceneDetect ──────────────────────────────► Keyframes (3 frame/scene: 25%, 50%, 75%)
       └─ Faster-Whisper medium (vi+en, VAD) ─────────► Transcript + timestamp

Keyframes ─┬─ SigLIP2 ViT-B-16 (image encoder) ──────────────────► FAISS Image Index (dim=768)
           │
           ├─ Qwen2-VL-2B-Instruct 4-bit (VLM caption, tiếng Việt) ─┐
           ├─ EasyOCR (vi + en) ─────────────────────────────────────┼─ BGE-M3 ──► FAISS Text Index (dim=1024)
           └─ Transcript (Whisper) ──────────────────────────────────┘
                                                                         └─ BM25 Okapi ──► BM25 Sparse Index
```

### Giai đoạn 2 — Tìm Kiếm (Search, real-time trên Streamlit)

```
QUERY (vi/en)
  │
  ├─[1]─ Gemini API: dịch Vi → En (nếu query tiếng Việt)
  │       └─ SigLIP2 text encoder (GPU) ──► FAISS Image Index ──► Top-50 (nhánh ảnh)
  │
  ├─[2]─ BGE-M3 (CPU, tiết kiệm VRAM) ──► FAISS Text Index ──► Top-50 (nhánh text dense)
  │
  └─[3]─ BM25 Okapi (CPU, zero VRAM) ──► BM25 Sparse Index ──► Top-50 (nhánh keyword)
                │
                ▼
    RRF Fusion (3 nhánh) ──► Top-50 hợp nhất
                │
                ▼
    CrossEncoder bge-reranker-v2-m3 ──► Top-10
                │
                ▼ (tuỳ chọn)
    Gemini VLM Verify (nhìn ảnh thật + rubric JSON) ──► Kết quả cuối
```

---

## Model & Thông Số Kỹ Thuật

| Thành phần | Model | Chạy trên | VRAM/RAM |
|---|---|---|---|
| Scene detection | PySceneDetect ContentDetector | CPU | — |
| Speech-to-text | faster-whisper `medium`, float16 | GPU | ~1.5GB VRAM |
| Image encoder | **SigLIP2 ViT-B-16** (open_clip) | GPU | ~400MB VRAM |
| VLM caption | Qwen2-VL-2B-Instruct **4-bit NF4** | GPU | ~2.0GB VRAM |
| OCR | EasyOCR vi+en | CPU/GPU | — |
| Text embedding | BGE-M3 (dense, dim=1024) | **CPU** | ~4GB RAM |
| Sparse retrieval | **BM25 Okapi** (rank-bm25) | CPU | <100MB RAM |
| Reranker | bge-reranker-v2-m3, fp16 | CPU | ~2GB RAM |
| Translation/Verify | Gemini API (gemini-flash) | Cloud API | — |

> **Tổng VRAM khi search:** CLIP ~400MB + CUDA overhead ~300MB ≈ **< 1GB** (an toàn với 4GB)

---

## Cài Đặt

```bash
# 1. Tạo virtualenv
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. Cài torch đúng bản CUDA (kiểm tra CUDA version bằng `nvidia-smi`)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Cài các package còn lại
pip install -r requirements.txt

# 4. Cấu hình Gemini API key (tùy chọn, nhưng nên có)
#    Lấy key miễn phí tại https://aistudio.google.com/apikey
#    Dùng cho: dịch query Vi→En cho SigLIP2 + verify kết quả bằng VLM
export GEMINI_API_KEY="your-key-here"   # Linux/Mac
# setx GEMINI_API_KEY "your-key-here"   # Windows (mở lại terminal sau)

# Hoặc tạo file .env trong thư mục gốc:
echo 'GEMINI_API_KEY=your-key-here' > .env
```

> Nếu **không cấu hình** `GEMINI_API_KEY`: pipeline vẫn chạy đầy đủ với 3 nhánh (SigLIP2 + BGE-M3 + BM25), chỉ mất bước dịch query và verify cuối.

---

## Chạy Pipeline

```bash
# ── Bước 0 ──────────────────────────────────────────────────────────────────
# Copy video vào thư mục data/videos/ (.mp4, .avi, .mkv, .mov, .webm)

# ── Bước 1 ──────────────────────────────────────────────────────────────────
# Tách keyframe (3 frame/scene) + transcript Whisper → cache JSON
python extract_keyframes.py

# ── Bước 2 ──────────────────────────────────────────────────────────────────
# Build toàn bộ index:
#   (a) SigLIP2 image encoder → FAISS Image Index (dim=768)
#   (b) Qwen2-VL 4-bit → captions.json
#   (c) EasyOCR → ocr.json
#   (d) BGE-M3 → FAISS Text Index (dim=1024)
#   (e) BM25 Okapi → bm25_index.pkl  ← mới
python build_index.py

# ── Bước 3 ──────────────────────────────────────────────────────────────────
# Khởi động giao diện tìm kiếm Streamlit
streamlit run app.py
```

Mỗi bước đều **cache theo video/ảnh cụ thể** — nếu bị gián đoạn, chạy lại đúng lệnh trên sẽ tự tiếp tục từ chỗ dang dở.

---

## Cấu Trúc Thư Mục

```
aic_retrieval/
├── config.py              # ⚙️  Cấu hình trung tâm (path, model name, tham số)
├── extract_keyframes.py   # 📹  Bước 1: PySceneDetect + Faster-Whisper
├── build_index.py         # 🔨  Bước 2: SigLIP2 / Qwen2-VL / EasyOCR / BGE-M3 / BM25
├── search_utils.py        # 🔍  Bước 3: search + RRF fusion + rerank + verify
├── app.py                 # 🖥️  UI Streamlit
├── requirements.txt
├── .env                   # GEMINI_API_KEY (không commit lên git)
│
├── data/
│   ├── videos/            # ← bỏ video gốc vào đây
│   └── keyframes/         # ảnh keyframe tự động sinh ra
│
├── cache/
│   ├── keyframe_meta.json # metadata keyframe (video_id, frame_idx, timestamp, path)
│   ├── transcripts.json   # transcript Whisper theo video
│   ├── captions.json      # caption Qwen2-VL theo keyframe
│   └── ocr.json           # text OCR theo keyframe
│
└── index/
    ├── image_index.faiss  # FAISS Image Index (SigLIP2, dim=768)
    ├── image_id_map.json
    ├── text_index.faiss   # FAISS Text Index (BGE-M3, dim=1024)
    ├── text_id_map.json
    └── bm25_index.pkl     # BM25 Sparse Index  ← mới
```

---

## Rebuild Một Phần (khi đổi model hoặc thêm video mới)

| Trường hợp | Cần làm |
|---|---|
| Thêm video mới | Chạy lại `extract_keyframes.py` → `build_index.py` (tự bỏ qua video cũ) |
| Đổi CLIP model (ví dụ: thay SigLIP2 variant) | Xóa `index/image_index.faiss` + `image_id_map.json` → `build_index.py` |
| Rebuild BM25 (sau khi thêm video/caption mới) | Xóa `index/bm25_index.pkl` → `build_index.py` |
| Rebuild text index (BGE-M3) | Xóa `index/text_index.faiss` + `text_id_map.json` → `build_index.py` |
| Rebuild caption (Qwen2-VL) | Xóa `cache/captions.json` → `build_index.py` |

---

## Tham Số Tìm Kiếm (config.py)

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `TOP_K_RETRIEVE` | 50 | Số kết quả lấy từ mỗi nhánh trước fusion |
| `TOP_K_RERANK` | 10 | Số kết quả cuối sau CrossEncoder |
| `FUSION_METHOD` | `"rrf"` | `"rrf"` hoặc `"weighted"` |
| `RRF_K` | 60 | Hằng số k của RRF (giá trị chuẩn trong literature) |
| `KEYFRAME_POSITIONS` | `[0.25, 0.5, 0.75]` | Vị trí lấy frame trong mỗi scene |
| `WHISPER_MODEL_SIZE` | `"medium"` | Đổi sang `"large-v3"` để tăng chất lượng tiếng Việt |

---

## Troubleshooting

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `RuntimeError: Chưa có index/image_index.faiss` | Chưa chạy `build_index.py` | Chạy `python build_index.py` |
| `AssertionError` khi search (dim mismatch) | Đổi CLIP model nhưng chưa xóa index cũ | Xóa `index/image_index.faiss` + `image_id_map.json`, rebuild |
| `ModuleNotFoundError: rank_bm25` | Chưa cài package | `pip install rank-bm25` |
| `FlagEmbedding` cài lỗi | Conflict version | `pip install -U FlagEmbedding transformers accelerate` |
| OOM khi build caption (Qwen2-VL) | Ảnh đầu vào quá lớn | Giảm `max_pixels` trong `build_captions()` ở `build_index.py` |
| Gemini rate limit (HTTP 429) | Quá nhiều request | `build_index.py` tự sleep+retry; chạy lại sau vài phút |
| EasyOCR chậm lần đầu | Download model OCR (~200MB) | Chờ lần đầu; các lần sau đọc từ `~/.EasyOCR/model/` |
| BM25 không trả kết quả | Chưa build BM25 index | Chạy `build_index.py` (hoặc xóa `index/bm25_index.pkl` để rebuild) |
