# 🔍 AIC 2026 (Bảng A) — Multimedia Video Retrieval System

> Hệ thống truy vấn video/keyframe từ mô tả tiếng Việt hoặc tiếng Anh cho **AI Challenge HCMC 2026**.  
> Hỗ trợ đầy đủ 3 dạng truy vấn: **Textual KIS**, **Q&A**, **TRAKE**.  
> Chạy **hoàn toàn local**, tối ưu cho GPU **4GB VRAM** (RTX 3050 Laptop trở lên).

---

## Mục Lục

- [Kiến trúc pipeline](#kiến-trúc-pipeline)
- [Model & tài nguyên phần cứng](#model--tài-nguyên-phần-cứng)
- [Cài đặt](#cài-đặt)
- [Chạy pipeline](#chạy-pipeline-dữ-liệu-chính-thức-btc)
- [3 dạng truy vấn & format nộp bài](#3-dạng-truy-vấn--format-nộp-bài)
- [Tính năng giao diện](#tính-năng-giao-diện-apppy)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Tham số tìm kiếm](#tham-số-tìm-kiếm-configpy)
- [Đánh giá hiệu năng](#đánh-giá-hiệu-năng-eval_retrievalpy)
- [Rebuild một phần](#rebuild-một-phần)
- [Troubleshooting](#troubleshooting)

---

## Kiến Trúc Pipeline

Hệ thống dùng **dữ liệu chính thức do BTC cung cấp** (CLIP features ViT-B/32 + metadata YouTube + object detection) làm nguồn chính — không tự encode lại ảnh từ đầu, để đảm bảo `frame_idx` khớp tuyệt đối với ground truth khi chấm điểm.

### Giai đoạn 1 — Lập chỉ mục (chạy offline 1 lần)

```
data/btc/ (BTC cung cấp)
  ├─ map_keyframes/   (CSV: frame_idx, pts_time, fps)
  ├─ media_info/      (JSON: title, description, keywords, author)
  ├─ clip_features/   (.npy: CLIP ViT-B/32, dim=512 — đã trích sẵn)
  └─ objects/         (JSON: Faster R-CNN detection — tuỳ chọn tải)
        │
        ▼
  parse_btc_data.py
  → cache/btc_keyframe_meta.json
  → cache/btc_media_info.json
  → index/btc_image_index.faiss  (KHÔNG re-encode ảnh)
        │
        ▼
  build_ocr_features.py  (tuỳ chọn)
  → cache/ocr.json  (EasyOCR vi+en quét từng keyframe)
        │
        ▼
  build_btc_text_index.py
  → Gộp: metadata YouTube + object detection + OCR
  → index/btc_text_index.faiss   (BGE-M3, dim=1024)
  → index/btc_bm25_index.pkl     (BM25 Okapi sparse)
```

### Giai đoạn 2 — Tìm kiếm (real-time trên Streamlit)

```
QUERY (vi/en)
  │
  ├─[1] langid.classify() — phát hiện ngôn ngữ (local)
  │       └─ Nếu không phải "en" → dịch 1 lần, dùng chung 2 nhánh dưới:
  │            Gemini API (ưu tiên) → NLLB-200-600M local (fallback offline)
  │
  ├─[Ảnh]  CLIP ViT-B/32 text encoder (GPU) → FAISS Image Index → Top-100
  ├─[Text] BGE-M3 dense (CPU)              → FAISS Text Index  → Top-100
  └─[BM25] Query Expansion (Gemini, tuỳ chọn) → BM25 Okapi     → Top-100
                │
                ▼
    RRF Fusion (k=60, trọng số Ảnh×3 / Text×1 / BM25×0.5)
                │
                ▼
    CrossEncoder bge-reranker-v2-m3 (GPU fp16)
    final_score = 0.5×norm_fused + 0.5×rerank_score
                │
                ▼ (tuỳ chọn)
    Gemini VLM Verify — chấm điểm 0–10 theo rubric, 10 luồng song song
```

**TRAKE** (chuỗi sự kiện): tách query thành N sự kiện con (Gemini/heuristic) → search từng sự kiện độc lập (`skip_rerank=True`, `top_k=300`) → ghép chuỗi theo ràng buộc thời gian tăng dần → soft-constraint (chấp nhận miss 1 sự kiện).

---

## Model & Tài Nguyên Phần Cứng

| Thành phần | Model | Chạy trên | VRAM/RAM |
|---|---|---|---|
| Image encoder | **CLIP ViT-B/32** (OpenAI) | GPU | ~600MB VRAM |
| Text embedding | **BGE-M3** (dense, dim=1024) | **CPU** | ~4GB RAM |
| Sparse retrieval | **BM25 Okapi** | CPU | <100MB RAM |
| Reranker | **bge-reranker-v2-m3** fp16 | GPU | ~1.1GB VRAM |
| OCR | **EasyOCR** (vi+en) | CPU (build-time) | — |
| Ngôn ngữ detect | **langid** | CPU | ~15MB |
| Dịch fallback | **NLLB-200-distilled-600M** | CPU | ~2.4GB RAM |
| Dịch / Q&A / Verify | **Gemini API** (`google-genai` SDK) | Cloud | — |

> **Tổng VRAM khi search:** CLIP (~600MB) + Reranker fp16 (~1.1GB) + overhead (~300MB) ≈ **~2GB** — an toàn với card 4GB VRAM.

---

## Cài Đặt

```bash
# 1. Tạo virtualenv
python3 -m venv venv
source venv/bin/activate

# 2. Cài torch đúng bản CUDA (kiểm tra với nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Cài toàn bộ package
pip install -r requirements.txt

# 4. (Khuyến nghị) Cải thiện BM25 với từ ghép tiếng Việt
pip install underthesea
# ⚠️ Nếu cài SAU KHI đã build BM25: xóa index/btc_bm25_index.pkl → build lại

# 5. Cấu hình Gemini API key
#    Lấy key tại https://aistudio.google.com/apikey
echo 'GEMINI_API_KEY=your-key-here' > .env
```

> **Không có Gemini key?** Pipeline vẫn chạy đầy đủ 3 nhánh tìm kiếm. Bước dịch query tự chuyển sang **NLLB-200 local** (tự tải ~2.4GB lần đầu). Các tính năng Q&A, TRAKE parsing, verify sẽ tắt hoặc dùng fallback heuristic.

---

## Chạy Pipeline (Dữ Liệu Chính Thức BTC)

```bash
# Bước 0 — Tải dữ liệu BTC vào đúng thư mục:
#   data/btc/map_keyframes/map-keyframes/*.csv
#   data/btc/media_info/media-info/*.json
#   data/btc/clip_features/clip-features-32/*.npy
#   data/btc/objects/objects/<video_id>/*.json  (tuỳ chọn)
#   data/keyframes/<video_id>/*.jpg             (nếu muốn hiển thị ảnh + OCR)

# Bước 1 — Parse BTC data + build Image FAISS Index
python3 parse_btc_data.py

# Bước 2 — OCR toàn bộ keyframe (tuỳ chọn, ~5-10 tiếng với GPU, để máy chạy qua đêm)
python3 build_ocr_features.py

# Bước 3 — Build Text Index (BGE-M3) + BM25 Index
python3 build_btc_text_index.py

# Bước 4 — Khởi động giao diện Streamlit
streamlit run app.py
```

Mỗi bước **checkpoint riêng** — nếu bị gián đoạn, chạy lại sẽ tự bỏ qua phần đã xong.

---

## 3 Dạng Truy Vấn & Format Nộp Bài

Giao diện tự phân loại (`qa_module.detect_query_type()`). Luôn kiểm tra **badge loại truy vấn** hiển thị phía trên kết quả trước khi tick chọn.

| Loại | Khi nào | Format nộp bài |
|---|---|---|
| **Textual KIS** | Mô tả 1 sự kiện, tìm video + frame | `video_id, frame_idx, timestamp_sec` |
| **Q&A** | Có câu hỏi kèm mô tả sự kiện | `video_id, frame_idx, timestamp_sec, answer` |
| **TRAKE** | Chuỗi nhiều sự kiện theo thứ tự thời gian | `video_id, frame_ids: [id_1, id_2, ..., id_n]` |

- **Q&A**: nút `🤖 AI` dùng Gemini sinh đáp án ngắn gọn (1–5 từ) từ ảnh — sửa tay được trước khi nộp.
- **TRAKE**: khung "Xem và chỉnh sửa chuỗi sự kiện" cho phép sửa tay từng `frame_id` trước khi chọn.
- Nút **⬇️ Tải JSON / CSV** xuất đúng format, giới hạn tối đa 100 kết quả theo quy chế BTC.

---

## Tính Năng Giao Diện (app.py)

| Tính năng | Mô tả |
|---|---|
| **Search bar** | Gõ query tiếng Việt/Anh, tự phân loại KIS / Q&A / TRAKE |
| **Query Expansion** | Checkbox — Gemini sinh thêm 2 cách diễn đạt đồng nghĩa cho BM25 |
| **Hybrid Fusion** | RRF (khuyến nghị) hoặc Weighted — chọn trong sidebar |
| **Gemini VLM Verify** | Checkbox — Gemini chấm điểm 0–10 từng ảnh, 10 luồng song song |
| **Quick Select** | Nút ⚡ Top 1 / ⚡ Top 5 — chọn nhanh kết quả hàng đầu |
| **Xóa từng ảnh** | Nút ❌ xóa riêng lẻ từng item khỏi danh sách nộp bài |
| **Text khớp** | Expand xem metadata / objects / OCR đã match với query |
| **Video player** | Phát video tại timestamp keyframe ±2s ngay trong UI |
| **Sửa chuỗi TRAKE** | Sửa tay `frame_id` từng sự kiện trong chuỗi |
| **Export nộp bài** | Tải JSON / CSV theo đúng format AIC, tối đa 100 kết quả |

---

## Cấu Trúc Thư Mục

```
aic_retrieval/
├── config.py                 # ⚙️  Cấu hình trung tâm (path, model, tham số)
├── parse_btc_data.py         # 📦  Bước 1: parse CSV/JSON/npy BTC → Image FAISS Index
├── build_ocr_features.py     # 🔤  Bước 2: EasyOCR toàn bộ keyframe → cache/ocr.json
├── build_btc_text_index.py   # 🔨  Bước 3: metadata + OCR + objects → BGE-M3 + BM25
├── search_utils.py           # 🔍  Core: search + RRF fusion + rerank + verify + query expansion
├── qa_module.py              # ❓  Phân loại truy vấn + sinh đáp án Gemini
├── trake_module.py           # ⛓️  Tách + ghép chuỗi sự kiện TRAKE
├── tokenizer_utils.py        # 🔤  Language detect + BM25 tokenizer dùng chung
├── eval_retrieval.py         # 📊  Đo R@1/5/20/50/100 tự động (cần eval_queries.json)
├── app.py                    # 🖥️  UI Streamlit (search, export nộp bài)
├── requirements.txt
├── .env                      # GEMINI_API_KEY (không commit lên git)
│
├── data/
│   ├── btc/                  # Dữ liệu gốc BTC (map_keyframes, media_info, clip_features, objects)
│   └── keyframes/            # Ảnh keyframe BTC (giải nén để hiển thị ảnh + chạy OCR)
│
├── cache/
│   ├── btc_keyframe_meta.json
│   ├── btc_media_info.json
│   └── ocr.json
│
└── index/
    ├── btc_image_index.faiss
    ├── btc_image_id_map.json
    ├── btc_text_index.faiss
    ├── btc_text_id_map.json
    └── btc_bm25_index.pkl
```

---

## Tham Số Tìm Kiếm (config.py)

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `TOP_K_RETRIEVE` | `100` | Số kết quả từ mỗi nhánh trước fusion |
| `TOP_K_RERANK` | `100` | Số kết quả sau CrossEncoder (BTC cho nộp tối đa 100) |
| `FUSION_METHOD` | `"rrf"` | `"rrf"` (khuyến nghị) hoặc `"weighted"` |
| `RRF_K` | `60` | Hằng số k của công thức RRF |
| `GEMINI_MODEL` | `"gemini-3.5-flash-lite"` | Model Gemini dùng cho dịch, Q&A, TRAKE, verify |
| `OCR_LANGS` | `["vi", "en"]` | Ngôn ngữ EasyOCR |

---

## Đánh Giá Hiệu Năng (eval_retrieval.py)

Tạo file `eval_queries.json` với các query có ground truth từ AIC 2025:

```json
[
    {
        "query": "Xe tải lật trên đường cao tốc",
        "gt_video_id": "L22_V011",
        "gt_frame_idx": 5007
    }
]
```

Chạy đo điểm:

```bash
# Đo R@1/5/20/50/100 — Final Score = trung bình 5 chỉ số
python3 eval_retrieval.py

# Thêm Query Expansion để so sánh A/B
python3 eval_retrieval.py --expansion
```

---

## Rebuild Một Phần

| Trường hợp | Cần làm |
|---|---|
| Có thêm batch BTC mới | Giải nén đè vào `data/btc/` → xóa `index/btc_image_index.faiss` → `parse_btc_data.py` |
| Mới xong OCR | Xóa `index/btc_text_index.faiss` + `btc_bm25_index.pkl` → `build_btc_text_index.py` |
| Cài/gỡ underthesea | Xóa `index/btc_bm25_index.pkl` → build lại (**bắt buộc**) |
| Đổi model reranker/BGE-M3 | Sửa `config.py` → xóa text index + id_map → build lại |

---

## Troubleshooting

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `RuntimeError: Chưa có index/btc_image_index.faiss` | Chưa chạy `parse_btc_data.py` | `python3 parse_btc_data.py` |
| Không có ảnh hiển thị | Chưa tải Keyframes ZIP | Giải nén vào `data/keyframes/` |
| BM25 recall thấp sau cài underthesea | Tokenizer build/search không khớp | Xóa `index/btc_bm25_index.pkl` → build lại |
| `Cannot copy out of meta tensor` | Model cũ bị cache trong bộ nhớ Streamlit | Restart Streamlit (`Ctrl+C` → `streamlit run app.py`) |
| Gemini rate limit (HTTP 429) | Quá nhiều request liên tiếp | Giảm tần suất search / tắt Verify |
| TRAKE không tìm ra chuỗi nào | Query TRAKE quá phức tạp | Thử chia thủ công thành từng câu query KIS đơn giản |
| `[CLIP] Không thể dịch query` | Gemini + NLLB đều lỗi | Kiểm tra mạng hoặc `pip install sentencepiece` |
| `ModuleNotFoundError` | Thiếu package | `pip install -r requirements.txt` |