# 🔍 AIC 2026 (Bảng A) — Multimedia Video Retrieval System

> Hệ thống truy vấn video/keyframe từ mô tả tiếng Việt hoặc tiếng Anh cho **AI Challenge HCMC 2026**.  
> Hỗ trợ đầy đủ 3 dạng truy vấn: **Textual KIS**, **Q&A**, **TRAKE**.  
> Chạy **hoàn toàn local**, tối ưu cho GPU **4GB VRAM** (RTX 3050 Laptop trở lên).

---

## 🌟 Cập Nhật Mới Nhất

*   **Tích hợp Nhánh Âm thanh (ASR)**: Bóc tách hơn 126k câu thoại từ `asr_results.json` vào nhánh Text (BGE-M3 & BM25), cải thiện đáng kể khả năng tìm kiếm video theo lời thoại nhân vật.
*   **Sửa lỗi chuỗi TRAKE (Chronological Order)**: Sửa logic fallback đảm bảo chuỗi các keyframe kiện E1, E2, E3 luôn tuân thủ nghiêm ngặt thứ tự thời gian ($F_{E1} \le F_{E2} \le F_{E3}$).
*   **Frame Browser UI**: Di chuyển và làm lại giao diện duyệt frame theo video_id ra màn hình chính, cho phép hiển thị ảnh to, dễ dàng chọn frame và lưu trực tiếp vào kết quả (Rất hữu ích để sửa lỗi tìm kiếm TRAKE thủ công).
*   **Đóng gói tự động (fix_and_zip.py)**: Tự động format, loại bỏ các ký tự BOM ẩn, sửa lỗi dấu phẩy dư ở cuối, và nén các file CSV theo đúng chuẩn submission của BTC.
*   **SigLIP2-SO400M hoàn chỉnh**: Build xong toàn bộ 177,321 keyframe BTC với SigLIP2-SO400M (dim=1152). Pipeline giờ chạy **4 nhánh song song** (CLIP + SigLIP2 + BGE-M3 + BM25) → RRF Fusion → CrossEncoder.
*   **Tối ưu VRAM (OOM Catcher)**: Fallback tự động sang CPU khi `torch.cuda.OutOfMemoryError` cho cả SigLIP2 và Reranker.

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

Hệ thống dùng **dữ liệu chính thức BTC** làm nguồn chính — không tự encode lại ảnh để đảm bảo `frame_idx` khớp tuyệt đối khi chấm điểm.

---

### Giai đoạn 1 — Lập chỉ mục (chạy offline 1 lần)

```
┌──────────────────── INPUT: data/btc/ (BTC cung cấp) ────────────────────┐
│  map_keyframes/*.csv  → frame_idx, pts_time, fps cho từng video          │
│  media_info/*.json    → title, description, keywords, author (YouTube)  │
│  clip_features/*.npy  → vector CLIP ViT-B/32 dim=512, 1 file/video      │
│  objects/*/*.json     → Faster R-CNN: class_entities + scores/frame     │
│  keyframes/<vid>/*.jpg → ảnh .jpg từng keyframe (để hiển thị + OCR)    │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼  parse_btc_data.py
┌──────────────────── NHÁNH ẢNH ──────────────────────────────────────────┐
│  .npy CLIP features (dim=512) của BTC                                   │
│    → L2-normalize → FAISS IndexFlatIP (cosine sim = inner product)      │
│    → index/btc_image_index.faiss  +  btc_image_id_map.json             │
│                                                                          │
│  ⚠️  KHÔNG re-encode lại ảnh — dùng thẳng vector BTC đã trích sẵn      │
│     → frame_idx khớp tuyệt đối ground truth, không bị lệch 1 frame     │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼  build_ocr_features.py  (tuỳ chọn, để máy qua đêm)
┌──────────────────── OCR ────────────────────────────────────────────────┐
│  EasyOCR (vi + en) quét từng ảnh keyframe                               │
│    → đọc: biển hiệu, news ticker, phụ đề cứng, banner, số liệu         │
│    → giữ text có confidence > 0.3 → nối thành chuỗi                   │
│    → cache/ocr.json  {"/path/to/frame.jpg": "VTV1 18:30 HÀ NỘI"}      │
│    → Checkpoint mỗi 500 ảnh (tiếp tục được nếu bị ngắt)               │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼  build_btc_text_index.py
┌──────────────────── NHÁNH TEXT ─────────────────────────────────────────┐
│  Gộp 4 nguồn text thành danh sách records:                              │
│                                                                          │
│  [1] YouTube Metadata (video-level, 873 records):                       │
│      "title + description[:500] + keywords + author"                    │
│      → đại diện CHỦ ĐỀ, nhân vật, sự kiện chính của video              │
│                                                                          │
│  [2] Object Detection (frame-level, ~171k records):                     │
│      Đọc objects/*.json của BTC (Faster R-CNN)                          │
│      → threshold 0.2, đếm số lượng → "Person Person Car Building"      │
│      → đại diện NỘI DUNG TRỰC QUAN trong từng keyframe                 │
│                                                                          │
│  [3] OCR (frame-level, ~149k records nếu đã chạy OCR):                 │
│      EasyOCR (vi+en) đọc text trong ảnh (news ticker, biển hiệu...)    │
│      frame_idx khớp ground truth qua kf_seq_lookup (seq → frame_idx)   │
│      → đại diện TEXT XUẤT HIỆN trong khung hình                        │
│                                                                          │
│  [4] ASR (audio-level, ~126k records):                                 │
│      Lời thoại bóc tách từ asr_results.json                            │
│      → đại diện LỜI THOẠI TRONG VIDEO theo timestamp                   │
│                                                                          │
│      ↓ Tổng: 449,076 records (873 + 171,741 + 149,677 + 126,785)       │
│  BGE-M3 (FlagEmbedding, đa ngôn ngữ vi+en, dim=1024, chạy CPU)         │
│    → encode tất cả → FAISS IndexFlatIP                                  │
│    → index/btc_text_index.faiss  +  btc_text_id_map.json               │
│                                                                          │
│  BM25 Okapi (rank-bm25, tokenizer underthesea nếu đã cài)              │
│    → same records → sparse keyword index                                │
│    → index/btc_bm25_index.pkl                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Giai đoạn 2 — Tìm kiếm (real-time, Streamlit)

```
QUERY (tiếng Việt hoặc tiếng Anh)
  │
  ▼  langid.classify()  — phát hiện ngôn ngữ (local, <1ms)
  │
  ├── Nếu tiếng Anh → dùng thẳng
  └── Nếu tiếng Việt → DỊCH 1 LẦN, dùng chung cho nhánh Ảnh + Text:
        [1] Gemini API (gemini-3.5-flash-lite, ~0.5s)  ← ưu tiên
        [2] NLLB-200-distilled-600M local (~2s, CPU)   ← fallback offline
        [3] Query gốc + cảnh báo                        ← nếu cả 2 lỗi


════════════ 3 NHÁNH TÌM KIẾM CHẠY SONG SONG ════════════

┌─────────────────────────────────────────────────────────┐
│  NHÁNH ẢNH — Semantic Image Search                      │
│                                                         │
│  Input : query đã dịch sang tiếng Anh                  │
│  Model : CLIP ViT-B/32 text encoder  (GPU, ~10ms)      │
│    → encode query → vector dim=512                     │
│    → FAISS cosine similarity với 177k keyframe vectors  │
│  Output: Top-100 keyframes (sorted by cosine sim)      │
│                                                         │
│  Strengths : hiểu NGỮ NGHĨA hình ảnh                   │
│  Weakness  : không nhận diện tên riêng, số, text nhỏ  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  NHÁNH TEXT — Semantic Text Search                      │
│                                                         │
│  Input : query GỐC (vi/en, không dịch — BGE-M3 đa ng)  │
│  Model : BGE-M3 FlagModel  (CPU, tiết kiệm VRAM)       │
│    → encode query → vector dim=1024                    │
│    → FAISS cosine sim với 177k text records             │
│      (metadata + object labels + OCR text)             │
│  Output: Top-100 records → map về keyframe (frame_idx) │
│                                                         │
│  Strengths : hiểu tiếng Việt, tên người, địa danh      │
│  Weakness  : phụ thuộc chất lượng text trong index     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  NHÁNH BM25 — Sparse Keyword Search                     │
│                                                         │
│  Input : query gốc (vi/en)                             │
│          + Query Expansion nếu bật checkbox:           │
│            Gemini sinh 2 câu đồng nghĩa → search 3 lần │
│            kết quả merge, cached để không gọi API lại  │
│  Model : BM25 Okapi (rank-bm25, CPU, <100ms)           │
│    → tokenize (underthesea hoặc str.split())           │
│    → TF-IDF weighted exact keyword match               │
│  Output: Top-100 records theo BM25 score               │
│                                                         │
│  Strengths : exact match tên riêng, số liệu, địa danh  │
│  Weakness  : không hiểu synonym, ngữ nghĩa             │
└─────────────────────────────────────────────────────────┘


════════════ FUSION & RERANKING ════════════

  ▼  RRF Fusion  (Reciprocal Rank Fusion, k=60)
  ┌────────────────────────────────────────────────────────┐
  │  score(frame) = Σᵢ  weightᵢ / (k + rankᵢ(frame))        │
  │  Trọng số: CLIP×3.0 / SigLIP2×2.5 / BGE-M3×1.0 / BM25×0.5│
  │  → CLIP + SigLIP2 được ưu tiên vì dataset visual-heavy   │
  │  → Khử trùng: cùng video_id + frame_idx → giữ cao       │
  │  → Output: Top-100 candidates hợp nhất                   │
  └────────────────────────────────────────────────────────┘
         │
         ▼  CrossEncoder Reranker
  ┌────────────────────────────────────────────────────────┐
  │  Model: bge-reranker-v2-m3  (GPU fp16, ~1.1GB VRAM)  │
  │  Input: cặp (query, passage) cho mỗi candidate        │
  │  Passage = (theo thứ tự ưu tiên):                     │
  │    caption cache → OCR text → object labels           │
  │    → fallback: "Video L22_V001 | at 120.5s"          │
  │    (không bao giờ để passage rỗng → reranker vô nghĩa)│
  │  → Chấm relevance score, sigmoid normalize 0–1        │
  │                                                        │
  │  final_score = 0.5 × norm_fused + 0.5 × rerank_score │
  │  → Sort lại Top-100 theo final_score                  │
  └────────────────────────────────────────────────────────┘
         │
         ▼  (tuỳ chọn — bật checkbox "Gemini VLM Verify")
  ┌────────────────────────────────────────────────────────┐
  │  Gemini VLM Verify                                     │
  │  Gửi ảnh thật (bytes) + query cho Gemini API          │
  │  Rubric 0–10:                                          │
  │    9-10 : Khớp hoàn toàn (đúng object + hành động)   │
  │    6-8  : Khớp một phần (đúng object, sai context)    │
  │    3-5  : Chủ đề liên quan nhưng sai nội dung         │
  │    0-2  : Không liên quan                             │
  │  Chạy 10 ảnh song song (ThreadPoolExecutor)           │
  │  → Sort lại theo verify_score                         │
  └────────────────────────────────────────────────────────┘
         │
         ▼
  🖥️  Hiển thị Grid kết quả → Tick chọn → Export JSON/CSV


════════════ TRAKE (chuỗi sự kiện theo thứ tự thời gian) ════════════

  Query VD: "Người đàn ông cầm mic → sau đó bắt tay → cuối cùng vẫy tay"
    │
    ▼  trake_module.py
  Tách thành N sự kiện con:
    [A] "Người đàn ông cầm mic"
    [B] "bắt tay"
    [C] "vẫy tay"
  (Gemini parse nếu có API key, fallback: cắt theo "rồi/sau đó/tiếp theo")
    │
    ▼
  Search từng sự kiện độc lập (skip_rerank=True, top_k=300)
    → Tập_A, Tập_B, Tập_C (mỗi tập là list {video_id, frame_idx, timestamp})
    │
    ▼
  Giao video_id: chỉ giữ video xuất hiện trong TẤT CẢ các tập
  (soft-constraint: chấp nhận miss 1 sự kiện nếu không có video đủ)
    │
    ▼
  Ghép chuỗi: chọn tuple (f_A, f_B, f_C) sao cho
    timestamp_A < timestamp_B < timestamp_C  (đúng thứ tự thời gian)
    │
    ▼
  Output: {video_id, frame_ids: [f_A, f_B, f_C]}
```

---

## Model & Tài Nguyên Phần Cứng

| Thành phần | Model | Chạy trên | VRAM/RAM |
|---|---|---|---|
| Image encoder #1 | **CLIP ViT-B/32** (OpenAI, dim=512) | GPU/CPU | ~600MB VRAM |
| Image encoder #2 | **SigLIP2-SO400M** (dim=1152) | GPU fp16 / CPU | ~3.5GB VRAM |
| Text embedding | **BGE-M3** (dense, dim=1024) | **CPU** | ~4GB RAM |
| Sparse retrieval | **BM25 Okapi** | CPU | ~80MB RAM |
| Reranker | **bge-reranker-v2-m3** fp16 | GPU/CPU fallback | ~1.1GB VRAM |
| OCR (build-time) | **EasyOCR** (vi+en) | GPU → CPU | — |
| Language detect | **langid** | CPU | ~15MB |
| Dịch fallback | **NLLB-200-distilled-600M** | CPU | ~2.4GB RAM |
| Dịch / Q&A / Verify | **Gemini API** (`gemini-3.5-flash-lite`) | Cloud | — |

> **VRAM khi search (đủ index):** CLIP (~600MB) + SigLIP2 fp16 (~3.5GB) + Reranker fp16 (~1.1GB) ≈ **~5GB**  
> → Trên card 4GB: SigLIP2 + Reranker tự động fallback CPU (OOM Catcher) — pipeline vẫn chạy đủ 4 nhánh.

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

# Bước 2 — OCR toàn bộ keyframe (tuỳ chọn, để máy chạy qua đêm)
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
    ├── btc_image_index.faiss      # CLIP ViT-B/32, dim=512, 177,321 vectors
    ├── btc_image_id_map.json
    ├── siglip_image_index.faiss   # SigLIP2-SO400M, dim=1152, 177,321 vectors
    ├── siglip_image_id_map.json
    ├── btc_text_index.faiss       # BGE-M3, dim=1024, 449,076 records
    ├── btc_text_id_map.json
    └── btc_bm25_index.pkl         # BM25 Okapi, 449,076 records
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