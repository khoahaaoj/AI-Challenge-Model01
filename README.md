# 🔍 AIC 2026 (Bảng A) — Multimedia Video Retrieval System

> Hệ thống truy vấn video/keyframe từ mô tả tiếng Việt hoặc tiếng Anh cho **AI Challenge HCMC 2026**.
> Hỗ trợ đầy đủ 3 dạng truy vấn của vòng sơ tuyển: **Textual KIS**, **Q&A**, **TRAKE**.
> Chạy **hoàn toàn local**, tối ưu cho GPU **4GB VRAM** (RTX 3050 Laptop trở lên).

---

## Mục Lục

- [Kiến trúc pipeline](#kiến-trúc-pipeline)
- [Model & tài nguyên phần cứng](#model--tài-nguyên-phần-cứng)
- [Cài đặt](#cài-đặt)
- [Chạy pipeline (dữ liệu chính thức BTC)](#chạy-pipeline-dữ-liệu-chính-thức-btc)
- [3 dạng truy vấn & format nộp bài](#3-dạng-truy-vấn--format-nộp-bài)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Tính năng giao diện](#tính-năng-giao-diện-apppy)
- [Chiến lược dịch query](#chiến-lược-dịch-query)
- [Rebuild một phần](#rebuild-một-phần)
- [Tham số tìm kiếm](#tham-số-tìm-kiếm-configpy)
- [Troubleshooting](#troubleshooting)

---

## Kiến Trúc Pipeline

Hệ thống dùng **dữ liệu chính thức do BTC cung cấp** (video + keyframe map + CLIP features
ViT-B/32 + metadata YouTube + object detection) làm nguồn chính — không tự trích/encode lại
ảnh từ đầu, để đảm bảo `frame_idx` khớp tuyệt đối với ground truth khi chấm điểm.

### Giai đoạn 1 — Lập chỉ mục (Indexing, chạy offline 1 lần)

```
data/btc/ (BTC cung cấp)
  ├─ map_keyframes/   (CSV: n, pts_time, fps, frame_idx)
  ├─ media_info/      (JSON: title, description, keywords, author...)
  ├─ clip_features/   (.npy: CLIP ViT-B/32, dim=512 — ĐÃ trích sẵn)
  └─ objects/         (JSON: Faster R-CNN detection, tuỳ chọn tải thêm)
        │
        ▼
┌───────────────────────── parse_btc_data.py ─────────────────────────┐
│  CSV keyframe map  ──► cache/btc_keyframe_meta.json                 │
│  JSON media info   ──► cache/btc_media_info.json                    │
│  .npy CLIP features ──► index/btc_image_index.faiss  (dim=512)      │
│                          (KHÔNG re-encode ảnh — dùng thẳng vector    │
│                           BTC đã trích, tiết kiệm toàn bộ thời gian  │
│                           GPU và đảm bảo khớp ground truth)          │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────── build_ocr_features.py (tuỳ chọn) ────────────────┐
│  EasyOCR (vi+en, CPU) quét từng keyframe ──► cache/ocr.json           │
│  (chữ trên biển hiệu, banner, phụ đề cứng, news ticker...)            │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────── build_btc_text_index.py ─────────────────────┐
│  Gộp 3 nguồn text:                                                    │
│    1) Metadata YouTube (title + description + keywords + author)      │
│    2) Object detection (Faster R-CNN, BTC cung cấp — nếu đã tải)      │
│    3) OCR (cache/ocr.json — nếu đã chạy build_ocr_features.py)        │
│        │                                                               │
│        ├─ BGE-M3 (đa ngôn ngữ, dense) ──► index/btc_text_index.faiss  │
│        └─ BM25 Okapi (sparse, exact keyword) ──► index/btc_bm25_index.pkl │
└─────────────────────────────────────────────────────────────────────┘
```

> **Luồng thay thế** (`USE_BTC_DATA = False` trong `config.py`): dùng cho video tự có,
> ngoài phạm vi dữ liệu BTC — chạy `extract_keyframes.py` (PySceneDetect + Whisper) rồi
> `build_index.py` (tự encode ảnh bằng SigLIP2 + caption bằng Qwen2-VL-2B 4-bit). Xem
> mục [Luồng thay thế](#luồng-thay-thế-video-tự-có) bên dưới.

### Giai đoạn 2 — Tìm kiếm (Search, real-time trên Streamlit)

```
QUERY (vi/en)
  │
  ├─[1] langid.classify() phát hiện ngôn ngữ (local, không cần API)
  │       └─ nếu không phải "en" → dịch 1 lần, dùng chung cho cả 2 nhánh dưới:
  │            Gemini API (ưu tiên) → NLLB-200-distilled-600M local (fallback offline)
  │
  ├─[nhánh ẢNH]  CLIP ViT-B/32 text encoder (GPU) ──► FAISS Image Index ──► Top-100
  ├─[nhánh TEXT] BGE-M3 (CPU, tiết kiệm VRAM) ──► FAISS Text Index ──► Top-100
  └─[nhánh BM25] BM25 Okapi (query gốc, exact keyword) ──► Top-100
                    │
                    ▼
      RRF Fusion (k=60, trọng số Ảnh×3 / Text×1 / BM25×0.5) ──► hợp nhất, khử trùng theo keyframe
                    │
                    ▼
      CrossEncoder bge-reranker-v2-m3 (GPU, fp16) ──► rerank Top-100
        (candidate không có text khớp → fallback caption cache làm passage)
                    │
                    ▼ (tuỳ chọn, checkbox trong UI — chậm hơn, gọi API mỗi ảnh)
      Gemini VLM Verify (ảnh thật + rubric JSON 0-10) ──► sắp xếp lại kết quả cuối
```

**Truy vấn TRAKE** (chuỗi sự kiện có thứ tự) đi theo nhánh riêng ở `trake_module.py`:
tách câu truy vấn thành N sự kiện con (Gemini hoặc heuristic cắt chuỗi) → search từng sự
kiện độc lập → giao `video_id` chung → ghép chuỗi theo ràng buộc thời gian tăng dần.

---

## Model & Tài Nguyên Phần Cứng

| Thành phần | Model | Chạy trên | VRAM/RAM ước tính |
|---|---|---|---|
| Image encoder | **CLIP ViT-B/32** (OpenAI, khớp CLIP features BTC) | GPU | ~600MB |
| Text embedding | **BGE-M3** dense, dim=1024 | **CPU** (ép buộc, tiết kiệm VRAM) | ~4GB RAM |
| Sparse retrieval | **BM25 Okapi** (rank-bm25) | CPU | <100MB RAM |
| Reranker | **bge-reranker-v2-m3**, fp16 | GPU | ~1.1GB |
| OCR | **EasyOCR** (vi+en) | CPU (build-time) | — |
| Ngôn ngữ | **langid** (local) | CPU | ~15MB |
| Dịch fallback | **NLLB-200-distilled-600M** (local) | CPU | ~2.4GB RAM |
| Dịch / Verify / QA | **Gemini API** (`gemini-3.5-flash-lite`) | Cloud API | — |
| *(luồng thay thế)* Speech-to-text | faster-whisper `medium`, float16, VAD | GPU | ~1.5GB |
| *(luồng thay thế)* VLM caption | Qwen2-VL-2B-Instruct 4-bit NF4 | GPU | ~2.0GB |

> **Tổng VRAM khi search (luồng BTC chính thức):** CLIP ViT-B/32 (~600MB) + Reranker fp16
> (~1.1GB) + CUDA overhead (~300MB) ≈ **~2GB — an toàn với card 4GB VRAM**, còn dư chỗ cho
> các tác vụ khác chạy song song trên máy.

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

# 4. (Khuyến nghị) Cải thiện BM25 với từ ghép tiếng Việt
pip install underthesea
# ⚠️ Nếu cài SAU KHI đã build BM25: xóa index/btc_bm25_index.pkl → build lại

# 5. Cấu hình Gemini API key (khuyến nghị — dùng cho dịch query, Q&A, TRAKE parsing, verify)
#    Lấy key miễn phí tại https://aistudio.google.com/apikey
echo 'GEMINI_API_KEY=your-key-here' > .env
```

> **Không có Gemini key?** Pipeline vẫn chạy đầy đủ 3 nhánh tìm kiếm.
> Bước dịch query tự chuyển sang **NLLB-200 local** (tự tải ~2.4GB lần đầu).
> Các tính năng cần Gemini (sinh câu trả lời Q&A, tách sự kiện TRAKE bằng LLM, verify cuối)
> sẽ tắt hoặc dùng fallback heuristic đơn giản hơn.

---

## Chạy Pipeline (Dữ Liệu Chính Thức BTC)

```bash
# ── Bước 0 ──────────────────────────────────────────────────────
# Tải dữ liệu BTC (theo link trong đề bài vòng sơ tuyển) và giải nén vào:
#   data/btc/map_keyframes/map-keyframes/*.csv
#   data/btc/media_info/media-info/*.json
#   data/btc/clip_features/clip-features-32/*.npy
#   (tuỳ chọn) data/btc/objects/objects/<video_id>/*.json
# Nếu muốn hiển thị ảnh + chạy OCR: giải nén thêm Keyframes vào data/keyframes/<video_id>/

# ── Bước 1 ──────────────────────────────────────────────────────
# Parse CSV/JSON/npy của BTC + build Image FAISS Index (dùng thẳng CLIP features có sẵn)
python3 parse_btc_data.py

# ── Bước 2 (tuỳ chọn nhưng khuyến nghị) ─────────────────────────
# Chạy OCR trên toàn bộ keyframe đã tải để tăng recall nhánh text
python3 build_ocr_features.py

# ── Bước 3 ──────────────────────────────────────────────────────
# Build Text Index (BGE-M3) + BM25 Index từ metadata + objects + OCR
python3 build_btc_text_index.py

# ── Bước 4 ──────────────────────────────────────────────────────
# Khởi động giao diện tìm kiếm Streamlit
streamlit run app.py
```

Mỗi bước **cache riêng** (`cache/*.json`, `index/*.faiss`, `index/*.pkl`) — nếu bị gián
đoạn giữa chừng, chạy lại đúng lệnh trên sẽ tự bỏ qua phần đã xong.

### Luồng thay thế (video tự có)

Chỉ dùng khi cần thử nghiệm với video ngoài phạm vi BTC (đặt `USE_BTC_DATA = False`
trong `config.py`):

```bash
# Copy video vào data/videos/  (hỗ trợ .mp4, .avi, .mkv, .mov, .webm)
python3 extract_keyframes.py   # PySceneDetect (3 frame/scene) + Faster-Whisper transcript
python3 build_index.py         # SigLIP2 encode ảnh + Qwen2-VL caption + EasyOCR + BGE-M3 + BM25
streamlit run app.py
```

---

## 3 Dạng Truy Vấn & Format Nộp Bài

Giao diện tự phân loại truy vấn (`qa_module.detect_query_type()`) dựa trên từ khoá, có thể
sai với câu phức tạp — **luôn kiểm tra badge loại truy vấn hiển thị phía trên kết quả** trước
khi tick chọn để nộp.

| Loại | Khi nào | Format nộp bài (JSON/CSV) |
|---|---|---|
| **Textual KIS** | Mô tả 1 sự kiện, tìm đúng video + 1 frame | `video_id, frame_idx, timestamp_sec` |
| **Q&A** | Có câu hỏi kèm mô tả sự kiện | `video_id, frame_idx, timestamp_sec, answer` |
| **TRAKE** | Chuỗi nhiều sự kiện theo thứ tự thời gian | `video_id, frame_ids: [id_1, id_2, ..., id_n]` |

- **Q&A**: nút `🤖 AI` bên cạnh ô đáp án dùng Gemini sinh câu trả lời ngắn gọn (1–5 từ)
  từ ảnh keyframe thật — vẫn có thể sửa tay trước khi nộp.
- **TRAKE**: mỗi kết quả có khung "Xem và chỉnh sửa chuỗi sự kiện" — cho phép **sửa tay**
  từng `frame_id` của từng sự kiện trong chuỗi trước khi tick chọn, vì mô hình tự động
  chưa chắc chọn đúng ngay từ đầu.
- Nút **⬇️ Tải JSON** và **⬇️ Tải CSV** ở sidebar xuất đúng định dạng theo loại truy vấn
  đang chọn, giới hạn tối đa 100 kết quả theo quy chế BTC.

---

## Cấu Trúc Thư Mục

```
aic_retrieval/
├── config.py                 # ⚙️  Cấu hình trung tâm (path, model, tham số, USE_BTC_DATA)
│
├── parse_btc_data.py         # 📦  Bước 1: parse CSV/JSON/npy BTC → cache + Image FAISS Index
├── build_ocr_features.py     # 🔤  Bước 2 (tuỳ chọn): EasyOCR trên toàn bộ keyframe BTC
├── build_btc_text_index.py   # 🔨  Bước 3: gộp metadata + objects + OCR → BGE-M3 + BM25
├── search_utils.py           # 🔍  Bước 4: search 3 nhánh + RRF fusion + rerank + verify
├── qa_module.py               # ❓  Phân loại truy vấn (KIS/QA/TRAKE) + sinh câu trả lời Gemini
├── trake_module.py           # ⛓️  Tách + ghép chuỗi sự kiện cho truy vấn TRAKE
├── tokenizer_utils.py        # 🔤  Dùng chung: language detect + BM25 tokenizer
├── app.py                    # 🖥️  UI Streamlit (search, video player, export nộp bài)
├── test_search.py            # 🧪  Script test nhanh 1 query ngoài UI
│
├── extract_keyframes.py      # 📹  (luồng thay thế) PySceneDetect + Faster-Whisper
├── build_index.py            # 🔨  (luồng thay thế) SigLIP2 + Qwen2-VL + EasyOCR + BGE-M3 + BM25
│
├── requirements.txt
├── .env                       # GEMINI_API_KEY (không commit lên git)
│
├── data/
│   ├── btc/                   # dữ liệu gốc BTC (map_keyframes, media_info, clip_features, objects)
│   ├── keyframes/              # ảnh keyframe (BTC hoặc tự trích, tuỳ USE_BTC_DATA)
│   └── videos/                 # (luồng thay thế) video gốc tự có
│
├── cache/
│   ├── btc_keyframe_meta.json  # metadata keyframe BTC (video_id, frame_idx, timestamp)
│   ├── btc_media_info.json     # metadata YouTube BTC
│   ├── ocr.json                # text OCR theo đường dẫn ảnh
│   ├── captions.json           # (luồng thay thế) caption Qwen2-VL
│   └── transcripts.json        # (luồng thay thế) transcript Whisper
│
└── index/
    ├── btc_image_index.faiss   # FAISS Image Index (CLIP ViT-B/32, dim=512)
    ├── btc_image_id_map.json
    ├── btc_text_index.faiss    # FAISS Text Index (BGE-M3, dim=1024)
    ├── btc_text_id_map.json
    └── btc_bm25_index.pkl      # BM25 Sparse Index + flag "used_underthesea"
```

---

## Tính Năng Giao Diện (app.py)

| Tính năng | Mô tả |
|---|---|
| **Search bar** | Gõ query tiếng Việt hoặc tiếng Anh, tự phân loại KIS / Q&A / TRAKE |
| **Hybrid Fusion** | RRF (khuyến nghị) hoặc Weighted — chọn trong sidebar |
| **Gemini VLM Verify** | Checkbox tuỳ chọn — gửi ảnh thật cho Gemini chấm 0–10 |
| **Hiển thị kết quả** | Grid 2–5 cột, xem điểm Fused (RRF) / Rerank / Gemini từng kết quả |
| **Text khớp** | Expand xem metadata / objects / OCR đã khớp query |
| **Video player** | Phát video gốc tại timestamp keyframe ±2s ngay trong UI |
| **Sửa chuỗi TRAKE** | Sửa tay `frame_id` từng sự kiện trước khi nộp |
| **Sinh đáp án Q&A** | Nút AI dùng Gemini trả lời câu hỏi dựa trên ảnh keyframe |
| **Chọn & Export** | Tick keyframe đúng → tải JSON / CSV theo đúng format nộp bài AIC |

---

## Chiến Lược Dịch Query

Thứ tự ưu tiên khi query không phải tiếng Anh (dùng chung cho nhánh CLIP và BGE-M3):

```
1. langid.classify(query) → phát hiện ngôn ngữ local (không cần API)
   - Nếu là tiếng Anh → KHÔNG dịch
   - Mọi ngôn ngữ khác (vi, unknown...) → bước 2

2. Gemini API (nhanh, chất lượng cao)
   - Thành công → dùng kết quả dịch

3. NLLB-200-distilled-600M local (fallback khi Gemini lỗi/mất mạng)
   - Chạy CPU, không cần internet, lần đầu tự tải model ~2.4GB

4. Dùng query gốc + cảnh báo rõ ràng trong console (recall có thể giảm)
```

> BM25 luôn giữ nguyên query gốc (không dịch) để bắt exact-match tên riêng, số liệu.
> CrossEncoder cũng dùng query gốc vì là model đa ngôn ngữ, hiểu trực tiếp tiếng Việt.

---

## Rebuild Một Phần

| Trường hợp | Cần làm |
|---|---|
| Có thêm batch dữ liệu BTC mới | Giải nén đè vào `data/btc/` → xóa `index/btc_image_index.faiss` → `parse_btc_data.py` |
| Mới chạy xong OCR | `build_ocr_features.py` → xóa `index/btc_text_index.faiss` + `btc_bm25_index.pkl` → `build_btc_text_index.py` |
| Cài/gỡ underthesea | Xóa `index/btc_bm25_index.pkl` → build lại (**bắt buộc** — flag tokenizer thay đổi) |
| Đổi model reranker/BGE-M3 | Sửa `config.py` → xóa `index/btc_text_index.faiss` + `btc_text_id_map.json` → build lại |
| (luồng thay thế) Redo caption | Xóa `cache/captions.json` → `build_index.py` |

---

## Tham Số Tìm Kiếm (config.py)

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `USE_BTC_DATA` | `True` | Dùng dữ liệu chính thức BTC (khuyến nghị) hay tự trích |
| `TOP_K_RETRIEVE` | 100 | Số kết quả lấy từ mỗi nhánh trước fusion |
| `TOP_K_RERANK` | 100 | Số kết quả sau CrossEncoder (BTC cho nộp tối đa 100) |
| `FUSION_METHOD` | `"rrf"` | `"rrf"` (khuyến nghị) hoặc `"weighted"` |
| `RRF_K` | 60 | Hằng số k của công thức RRF |
| `CLIP_MODEL_NAME` | `"ViT-B-32"` (khi `USE_BTC_DATA=True`) | Phải khớp model BTC đã dùng để trích CLIP features |
| `OCR_LANGS` | `["vi", "en"]` | Ngôn ngữ EasyOCR |
| `GEMINI_MODEL` | `"gemini-3.5-flash-lite"` | Model Gemini dùng cho dịch, Q&A, TRAKE parsing, verify |

---

## Troubleshooting

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `RuntimeError: Chưa có index/btc_image_index.faiss` | Chưa chạy `parse_btc_data.py` | `python3 parse_btc_data.py` |
| Không có ảnh hiển thị trong kết quả | Chưa tải Keyframes ZIP về `data/keyframes/` | Tải + giải nén Keyframes theo hướng dẫn BTC |
| BM25 recall thấp sau cài underthesea | Tokenizer build/search không khớp | Xóa `index/btc_bm25_index.pkl` → build lại (flag tự lưu vào pickle) |
| `ModuleNotFoundError: rank_bm25` / `langid` | Chưa cài package | `pip install rank-bm25 langid` |
| `FlagEmbedding` cài lỗi | Conflict version | `pip install -U FlagEmbedding transformers accelerate` |
| NLLB download chậm lần đầu | Model ~2.4GB cần tải về | Chờ lần đầu; sau đó cache tại `~/.cache/huggingface/` |
| EasyOCR chậm lần đầu | Download model OCR ~200MB | Chờ lần đầu; cache tại `~/.EasyOCR/model/` |
| Gemini rate limit (HTTP 429) | Quá nhiều request liên tiếp | Giảm tần suất search, hoặc tắt Gemini verify |
| `[CLIP] Không thể dịch query` | Cả Gemini lẫn NLLB đều lỗi | Kiểm tra kết nối mạng hoặc cài lại `sentencepiece` |
| TRAKE không tìm ra chuỗi nào | Không có video chứa đủ TẤT CẢ sự kiện trong top-100 mỗi nhánh | Thử tách câu truy vấn thủ công thành từng phần đơn giản hơn, tìm riêng từng sự kiện |