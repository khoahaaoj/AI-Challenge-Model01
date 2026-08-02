# Đánh Giá Kỹ Thuật Toàn Diện — AI-Challenge-Model01
### Vai trò: Principal AI Research Engineer / Mentor kỹ thuật cho đội thi AIC 2026 (Bảng A)
### Phạm vi đọc: `config.py`, `extract_keyframes.py`, `build_index.py`, `search_utils.py`, `app.py`, `requirements.txt` (đọc toàn bộ source, không chỉ README)

---

## 0. Tóm tắt điều hành (Executive Summary)

Đây là một pipeline **retrieval hybrid 3 nhánh khá hoàn chỉnh cho một dự án sinh viên/team nhỏ** — kiến trúc đúng hướng, code có comment kỹ, đã tự fix ít nhất 1 bug logic nghiêm trọng (fallback passage cho CrossEncoder). Tuy nhiên khi đọc kỹ từng dòng, tôi tìm thấy **3 vấn đề mức Priority 1** có thể làm giảm Recall thực chiến đáng kể mà README không hề nhắc tới:

1. **BM25 tokenizer mismatch giữa build-time và query-time** (`build_index.py` dòng 397-406 vs `search_utils.py` dòng 353) — nếu bật `underthesea`, nhánh BM25 sẽ gần như vô dụng với từ ghép tiếng Việt.
2. **Phụ thuộc cứng vào Gemini Cloud API cho đúng bước quan trọng nhất của nhánh ảnh** (dịch query Vi→En) — vi phạm trực tiếp ràng buộc "không ưu tiên API nếu có phương án local tốt" của chính đề bài, và là điểm rủi ro lớn nhất nếu thi đấu bị hạn chế mạng.
3. **Heuristic `query.isascii()` để quyết định có dịch hay không** (`search_utils.py` dòng 191) — sai với tiếng Việt không dấu, một kiểu gõ rất phổ biến.

Ngoài ra là các vấn đề mức 2-3 (Whisper size, batch size, FAISS Flat index, thiếu eval set...). Chi tiết đầy đủ bên dưới.

---

## 1. Kiến Trúc Hiện Tại (vẽ lại từ source code thực tế)

```
╔══════════════════════════ OFFLINE — INDEXING (chạy 1 lần tại nhà) ══════════════════════════╗
║                                                                                                ║
║  extract_keyframes.py                                                                         ║
║  ┌────────────────────────────┐        ┌──────────────────────────────────┐                  ║
║  │ PySceneDetect ContentDetector│        │ faster-whisper "medium" (config) │                  ║
║  │ threshold=27.0, min_len=15  │        │ compute_type=float16(GPU)/int8(CPU)│                 ║
║  │ → 3 frame/scene @ 25/50/75% │        │ vad_filter=True, beam_size=5      │                  ║
║  │ (fallback: scene ngắn→1 frame)│      │ auto-detect ngôn ngữ (vi/en)      │                  ║
║  └──────────────┬─────────────┘        └──────────────┬───────────────────┘                  ║
║                 ▼                                       ▼                                     ║
║        keyframe_meta.json                       transcripts.json                              ║
║                 │                                       │                                     ║
╚═════════════════╪═══════════════════════════════════════╪═════════════════════════════════════╝
                   ▼                                       │
  build_index.py                                           │
  ┌─────────────────────────┐  ┌───────────────────────┐   │
  │ SigLIP2 ViT-B-16         │  │ Qwen2-VL-2B-Instruct  │   │
  │ (hf-hub, batch=16)       │  │ 4-bit NF4 (BitsAndBytes)│  │
  │ → FAISS IndexFlatIP      │  │ max_new_tokens=50      │   │
  │   dim=768 (cosine)       │  │ caption tiếng Việt     │   │
  └───────────┬──────────────┘  └───────────┬────────────┘   │
              │                             ▼                │
              │                       captions.json           │
              │                             │                │
              │            ┌────────────────┘                │
              │            ▼                                 ▼
              │      EasyOCR (vi+en)                  (records transcript
              │      → ocr.json                          + caption + ocr)
              │            │                                 │
              │            └──────────────┬──────────────────┘
              │                           ▼
              │                  BGE-M3 (encode text)
              │                  → FAISS IndexFlatIP dim=1024
              │                           │
              │                           ▼
              │              BM25Okapi (underthesea nếu có,
              │              fallback str.split()) → pickle
              ▼                           ▼
      image_index.faiss          text_index.faiss + bm25_index.pkl

╔══════════════════════════ ONLINE — SEARCH (real-time, Streamlit) ══════════════════════════╗
║  query (vi/en)                                                                                ║
║   ├─[1] search_image(): isascii()? → (nếu non-ASCII) Gemini dịch Vi→En                        ║
║   │      → SigLIP2 text encoder (GPU) → FAISS Image Index → top-50                            ║
║   ├─[2] search_text(): BGE-M3 (CPU, devices="cpu") → FAISS Text Index → top-50                ║
║   └─[3] search_bm25(): query.lower().split() (CPU) → BM25Okapi.get_scores() → top-50          ║
║                              │                                                                 ║
║                              ▼                                                                 ║
║              fuse_rrf() — RRF 3 nhánh, k=60, dedup theo (video_id, frame_idx)                  ║
║                              │                                                                 ║
║                              ▼                                                                 ║
║   rerank() — bge-reranker-v2-m3, passage = matched_text (fallback: caption cache nếu           ║
║              candidate chỉ đến từ CLIP) → top-10                                              ║
║                              │                                                                 ║
║                              ▼ (checkbox "Gemini VLM verify", tắt mặc định)                    ║
║   verify_with_gemini() — gửi ẢNH THẬT + rubric JSON 0-10 → sort lại top-10                     ║
║                              │                                                                 ║
║                              ▼                                                                 ║
║                     Streamlit UI: hiển thị, xem video ±2s, chọn → export JSON/CSV nộp bài       ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Điểm Chấm Theo Tiêu Chí (thang 10, so với chuẩn "sẵn sàng thi AIC")

| Tiêu chí | Điểm | Ghi chú |
|---|---|---|
| **Architecture** | 7.5/10 | Hybrid 3-nhánh + RRF + rerank + verify là kiến trúc chuẩn của các đội top VBS/AIC gần đây. Trừ điểm vì phụ thuộc cloud API ở khâu quan trọng. |
| **Retrieval** | 6.5/10 | Dense + sparse tốt, nhưng BM25 có nguy cơ bug tokenizer, và CLIP branch phụ thuộc dịch máy chưa robust. |
| **Engineering** | 7/10 | Cache-per-step rất tốt (resume được), code sạch, có docstring chi tiết. Thiếu retry/backoff cho Gemini API, thiếu logging có cấu trúc, thiếu test. |
| **Research** | 6/10 | Chọn model hợp lý cho 4GB VRAM (đã chứng tỏ có tìm hiểu), nhưng chưa có benchmark/eval nội bộ để CHỨNG MINH lựa chọn đó tốt hơn phương án khác trên chính dataset của đội. |
| **Scalability** | 6/10 | `IndexFlatIP` (brute-force) ổn với vài chục nghìn keyframe, nhưng dataset AIC thật (hàng trăm video, có thể >200k keyframe với 3 frame/scene) sẽ khiến search chậm dần và encode-time tăng tuyến tính. |
| **Maintainability** | 8/10 | `config.py` tập trung tốt, tách file rõ ràng theo 3 bước, cache resumable. Đây là điểm mạnh nhất của repo. |

**Trung bình: ~6.8/10** — nền tảng tốt, nhưng còn khoảng cách giữa "chạy được" và "tối ưu để thi đấu".

---

## 3. Đánh Giá Từng Module

### 3.1 Scene Detection (`extract_keyframes.py:57-80`)
- Dùng `ContentDetector(threshold=27.0, min_scene_len=15)` — giá trị mặc định phổ biến của PySceneDetect, hợp lý làm baseline.
- Có fallback tốt khi video không có scene rõ rệt (dùng OpenCV đếm tổng frame) — code phòng thủ tốt.
- **Thiếu**: không test threshold theo đặc thù dataset AIC (video tin tức/phim tài liệu có cắt cảnh nhanh khác hẳn video quay 1 shot dài). Threshold cố định 27.0 có thể tách quá vụn với video "talking head" (phỏng vấn) và quá thưa với video action.
- **Đề xuất**: chạy `detect-content` với 2-3 threshold khác nhau trên 5-10 video mẫu của BTC, đo số scene/phút, chọn giá trị theo phân phối thực tế thay vì giữ mặc định.

### 3.2 Keyframe Sampling (`extract_keyframes.py:83-138`)
- 3 frame/scene tại 25/50/75% — quyết định đúng, tăng recall ảnh mà không cần đổi model. Có dedup theo `seen_frames` và ngưỡng `MIN_SCENE_FRAMES_FOR_MULTI` để tránh lấy trùng ảnh ở scene ngắn — code chất lượng tốt.
- **Rủi ro scale**: 3x số lượng keyframe = 3x thời gian encode CLIP + 3x dung lượng ảnh + 3x thời gian Qwen2-VL caption (bước chậm nhất pipeline). Với dataset lớn, đây có thể là bottleneck I/O + thời gian build chứ không phải thiếu ý tưởng.

### 3.3 ASR — Whisper (`config.py:61-67`, `extract_keyframes.py:174-228`)
- **Phát hiện quan trọng**: comment trong `config.py` dòng 64 nói `"large-v3" cho chất lượng tiếng Việt tốt nhất`, nhưng giá trị thực tế đang set là `WHISPER_MODEL_SIZE = "medium"` (dòng 66). README của repo không nhắc gì đến sự khác biệt này.
- Đây thực ra là **quyết định đúng cho ràng buộc 4GB VRAM**: `faster-whisper large-v3` ở `float16` cần ~5-6GB VRAM, sẽ OOM trên RTX 3050 Laptop. `medium` (~1.5GB) là lựa chọn an toàn.
- **Đề xuất thực tế**: thử `large-v3` với `compute_type="int8_float16"` (không phải `"float16"`) — kỹ thuật quantize giúp large-v3 chạy trong ~3GB VRAM, vẫn có thể vượt quality của medium mà không OOM. Đáng thử nghiệm vì đây là bước chỉ chạy 1 lần offline (không giới hạn thời gian như lúc thi).
- `vad_filter=True` là quyết định tốt (loại khoảng lặng, giảm nhiễu transcript).

### 3.4 Caption — Qwen2-VL-2B 4-bit (`build_index.py:139-244`)
- Chạy local, không cần API — đúng tinh thần ràng buộc đề bài. NF4 quantization giảm ~75% VRAM, hợp lý.
- `max_new_tokens=50` (dòng 215): với caption tiếng Việt, do tokenizer BPE thường cần nhiều token hơn cho từ có dấu, 50 token có thể cắt cụt câu mô tả phức tạp (nhiều object trong 1 khung hình). Nên tăng lên 80-100 và benchmark độ dài caption thực tế.
- `max_pixels=384*28*28` (dòng 188) — resize khá nhỏ, có thể mất chi tiết nhỏ (biển số, chữ nhỏ) — nhưng đó cũng chính là việc của nhánh OCR nên có thể chấp nhận được.
- Cache theo key = path ảnh + auto-retry ảnh có nhãn fallback `"Khung hình video"` (dòng 151-156) — thiết kế tốt, cho phép chạy lại chỉ để fix caption lỗi mà không cần xóa toàn bộ cache.

### 3.5 OCR — EasyOCR (`build_index.py:250-283`)
- `OCR_LANGS = ["vi", "en"]` chạy đồng thời — đúng hướng.
- **Vấn đề tiềm ẩn**: EasyOCR đọc vi+en cùng lúc trong 1 reader sẽ **chậm hơn đáng kể** so với chạy riêng từng ngôn ngữ (do model phải cân bằng giữa 2 charset), và độ chính xác tiếng Việt của EasyOCR thực tế **thấp hơn PaddleOCR** trên văn bản có dấu (theo benchmark cộng đồng gần đây) — cần so sánh trực tiếp trên dataset của đội (xem mục 7).

### 3.6 Text Embedding — BGE-M3 (`build_index.py:336-368`, `search_utils.py:97-104, 272-319`)
- Encode ở build-time trên GPU, nhưng ở search-time cố tình ép về CPU (`devices="cpu"`, dòng 103) để tiết kiệm VRAM cho SigLIP2 + reranker cùng lúc — đây là một quyết định kỹ thuật rất tinh tế và đúng, đáng khen.
- `max_length=512` hợp lý cho caption/OCR/transcript ngắn.
- **Chưa tận dụng hết BGE-M3**: model này hỗ trợ cả dense + sparse (lexical) + ColBERT-style multi-vector trong 1 lần encode (`return_dense`, `return_sparse`, `return_colbert_vecs`), nhưng code chỉ dùng phần `dense_vecs`. Vì BM25 đã đảm nhiệm phần sparse riêng, việc không dùng sparse của BGE-M3 chấp nhận được, nhưng bỏ qua ColBERT-vecs là bỏ lỡ khả năng multi-vector matching tốt hơn cho câu query dài, phức tạp — có thể cân nhắc dùng làm bước rerank thay thế/bổ sung CrossEncoder (xem mục 6).

### 3.7 Image Retrieval — SigLIP2 + FAISS (`build_index.py:63-134`, `search_utils.py:217-252`)
- SigLIP2 ViT-B-16 qua `open_clip` hf-hub — lựa chọn tốt hơn CLIP ViT-B-32 gốc của OpenAI (sigmoid loss, train data mới hơn).
- `IndexFlatIP` trên vector đã L2-normalize = cosine similarity đúng chuẩn.
- **Vấn đề nghiêm trọng nhất của cả pipeline** nằm ở khâu dịch query trước khi vào SigLIP2 — xem mục 3.9.

### 3.8 Hybrid Fusion — RRF (`search_utils.py:388-434`)
- Công thức RRF chuẩn `1/(k+rank+1)`, `k=60` (giá trị phổ biến trong literature, đúng như comment ghi). Dedup theo `(video_id, frame_idx)` đúng logic, có ưu tiên giữ bản ghi có `matched_text` để phục vụ rerank sau — chi tiết nhỏ nhưng cho thấy code được viết cẩn thận.
- **Không có trọng số riêng cho từng nhánh trong RRF** — hiện tại code cộng đều `1/(k+rank+1)` cho cả 3 nhánh (ảnh, dense text, BM25) như nhau. Trên thực tế nhánh ảnh (SigLIP2) và nhánh text (BGE-M3+BM25) có "độ tin cậy" khác nhau tùy loại query (query mô tả hành động → nhánh ảnh mạnh hơn; query có tên riêng → BM25 mạnh hơn). RRF không có cơ chế weighted-by-source built-in — đây là hạn chế đã biết của RRF thuần, không phải bug, nhưng đáng note vì `FUSION_METHOD="weighted"` cũng tồn tại song song lại **không hỗ trợ BM25** (comment dòng 594 tự thừa nhận).

### 3.9 ⚠️ Vấn đề Priority 1 #1 — Query Translation cho CLIP (`search_utils.py:173-211`)
Đây là điểm tôi đánh giá là **rủi ro lớn nhất của toàn bộ hệ thống** khi đem ra thi đấu thật:

```python
def translate_query_for_clip(query):
    if not config.GEMINI_API_KEY:
        return query  # Chưa có key → dùng query gốc  (dòng 187)
    try:
        if query.isascii():
            return query   # (dòng 191-192)
```

Hai vấn đề cụ thể:
1. **Phụ thuộc Internet + Gemini API tại đúng bước quyết định chất lượng nhánh ảnh.** Nếu mạng chập chờn lúc thi (rất thường gặp ở các cuộc thi offline đông người dùng chung wifi), hoặc Gemini rate-limit, `except` sẽ bắt lỗi và fallback về **query tiếng Việt gốc đưa thẳng vào SigLIP2** — một encoder được train chủ yếu bằng tiếng Anh. Điều này **âm thầm phá vỡ toàn bộ nhánh ảnh** mà giao diện Streamlit không hề cảnh báo cho người dùng biết là đang chạy ở chế độ suy giảm.
2. **`query.isascii()` là heuristic sai** với tiếng Việt gõ không dấu (vd: "nguoi dan ong ao do") — chuỗi này **là ASCII thuần**, hàm sẽ trả về ngay không dịch (dòng 192), trong khi bản chất đây vẫn là tiếng Việt cần dịch. Nhiều thí sinh gõ nhanh trên bàn phím không có bộ gõ tiếng Việt sẽ dính lỗi này.

**Đây là điểm tôi sẽ ưu tiên sửa đầu tiên nếu là mentor** (xem Roadmap mục 9, Priority 1).

### 3.10 CrossEncoder Rerank (`search_utils.py:143-170, 480-509`)
- `bge-reranker-v2-m3` — đa ngôn ngữ, cùng họ BGE-M3, lựa chọn nhất quán về mặt hệ sinh thái.
- Code đã tự phát hiện và fix 1 bug thực sự nghiêm trọng: candidate chỉ đến từ nhánh CLIP (không có `matched_text`) trước đây bị CrossEncoder chấm điểm ~0 vì passage rỗng → luôn bị đẩy xuống cuối dù ảnh khớp tốt. Bản vá dùng `_get_passage_for_candidate()` để fallback về caption cache — **đây là một fix đúng và quan trọng, đáng ghi nhận là điểm mạnh của đội**, không phải điểm yếu.
- `compute_score(..., normalize=True)` dùng sigmoid-normalize — hợp lý để so sánh điểm giữa các candidate.

### 3.11 Gemini VLM Verify (`search_utils.py:515-565`)
- Rubric JSON rõ ràng (0-10 theo 4 mức) là cách làm đúng để có điểm số nhất quán từ LLM, có fallback parse digit nếu JSON lệch format — code phòng thủ tốt.
- **Vấn đề chiến lược**: bước này lại phụ thuộc Gemini lần thứ 2, và tốn 1 API call/candidate (10 call cho top-10) — với hàng trăm query lúc luyện tập, đây có thể chạm rate-limit hoặc phát sinh chi phí không cần thiết. Vì mặc định `do_verify=False` (checkbox tắt) nên rủi ro vận hành thấp hơn mục 3.9, nhưng cùng vấn đề nền tảng: **kiến trúc hiện tại coi Gemini API như một phần không thể tách rời của pipeline, trong khi đề bài yêu cầu ưu tiên local**.

---

## 4. Bottleneck — Phân Tích Cụ Thể

| Bottleneck | Vị trí trong code | Loại | Mức độ |
|---|---|---|---|
| GPU sequential loading | `build_index.py`: SigLIP2 (dòng 76-82) rồi Qwen2-VL (166-183) trong CÙNG 1 lần chạy script | Không giải phóng VRAM tường minh giữa 2 bước (chỉ có `del` + `empty_cache` ở cuối `build_captions`, không có ở `build_image_index`) | Thấp (tổng vẫn <4GB) nhưng nên đồng nhất cho sạch |
| Whisper transcribe | `extract_keyframes.py:205-209` | CPU/GPU tuần tự từng video, không batch nhiều video cùng lúc | Trung bình — tuyến tính theo tổng độ dài video |
| Qwen2-VL caption | `build_index.py:192-235` | Vòng lặp **từng ảnh một** (không batch), là bước chậm nhất trong `build_index.py` vì phải load ảnh + tokenize + generate riêng lẻ cho mỗi keyframe | **Cao** — đây là bottleneck build-time lớn nhất khi x3 keyframe/scene |
| API network calls | `translate_query_for_clip`, `verify_with_gemini` | I/O-bound, tuần tự từng candidate (verify không dùng batch/async) | Cao lúc verify bật, và là điểm lỗi (mục 3.9) lúc search |
| FAISS `IndexFlatIP` | `build_index.py:127-128, 361-362` | Brute-force O(n) mỗi query — ổn tới ~50-100k vector, chậm dần sau đó | Thấp hiện tại, sẽ thành vấn đề nếu dataset AIC thật lớn |
| BM25 tokenizer mismatch | `build_index.py:397-406` vs `search_utils.py:353` | Không phải bottleneck tốc độ mà là **bottleneck chất lượng silent** — xem mục 5 | **Cao về recall** |

---

## 5. Những Module Nên Thay / Sửa Ngay (không phải "đổi model to hơn" mà là sửa lỗi thực tế)

### 5.1 🔴 Priority cao nhất: BM25 tokenizer mismatch
- **Build** (`build_index.py:398-406`): nếu có `underthesea`, dùng `word_tokenize(text.lower(), format="text").split()` — tách từ ghép tiếng Việt thành token nối bằng `_` (vd: "thủ_tướng").
- **Search** (`search_utils.py:353`): luôn dùng `query.lower().split()` — KHÔNG bao giờ dùng underthesea, dù đã cài.
- **Hệ quả**: nếu đội cài `underthesea` (đúng theo khuyến nghị trong chính comment của code, dòng 383) để "cải thiện tiếng Việt", nhánh BM25 sẽ **mất khả năng khớp chính xác các từ ghép** — chính là lý do BM25 được thêm vào pipeline ngay từ đầu ("tên người, địa danh, thương hiệu" theo docstring dòng 341-342). Bug này **im lặng hoàn toàn** — không có exception, không có log cảnh báo, chỉ recall thấp hơn mà không ai biết vì sao.
- **Sửa**: tách hàm `_tokenize()` ra một module dùng chung (import ở cả 2 file), hoặc lưu luôn cờ "đã dùng underthesea hay chưa" vào file pickle của BM25 index để `search_bm25()` biết cách tokenize đúng theo đúng cách đã build.

### 5.2 🔴 Query translation robustness (mục 3.9)
- Bỏ heuristic `isascii()`, thay bằng thư viện detect ngôn ngữ nhẹ (`langid`, hoặc `fasttext` lid.176 — chạy local, không cần API) để quyết định có cần dịch hay không.
- Thêm **fallback dịch local** khi Gemini lỗi/không có mạng: một model dịch nhỏ chạy local (NLLB-200-distilled-600M hoặc VinAI's PhoMT/vinai-translate) thay vì im lặng bỏ qua bước dịch. Việc này giải quyết triệt để rủi ro "không có mạng lúc thi" nêu ở mục 3.9.

### 5.3 🟡 Qwen2-VL caption: batch hóa
- Hiện xử lý từng ảnh 1 (`build_index.py:192`). Qwen2-VL hỗ trợ batch nhiều ảnh/câu hỏi cùng lúc qua `processor(text=[...], images=[...])` với list >1 phần tử — nên gộp batch 4-8 ảnh cùng prompt để tận dụng GPU tốt hơn, giảm đáng kể thời gian build (đây là bottleneck lớn nhất theo mục 4).

### 5.4 🟡 EasyOCR → cân nhắc PaddleOCR
- Xem so sánh mục 6.3, cần benchmark thực tế trên ảnh chứa chữ tiếng Việt của chính dataset AIC trước khi quyết định đổi.

---

## 6. So Sánh Model Cho Từng Vai Trò (không chỉ chọn 1 model — có đánh đổi)

### 6.1 Image Encoder

| Model | Zero-shot Recall | Tốc độ (RTX3050) | VRAM | Hỗ trợ tiếng Việt | Độ khó tích hợp |
|---|---|---|---|---|---|
| CLIP ViT-B/32 (OpenAI gốc) | Baseline | Nhanh nhất | ~350MB | Kém (train chủ yếu tiếng Anh) | Rất dễ |
| **SigLIP2 ViT-B-16 (đang dùng)** | +8-10% so với CLIP B/32 | Trung bình | ~400MB | Trung bình (WebLI có đa ngôn ngữ nhưng không tối ưu cho vi) | Dễ (qua open_clip) |
| SigLIP2 So400m | Tốt hơn nữa (+3-5% so với B-16) | Chậm hơn ~2x | ~1.6GB (fp16) | Tương tự B-16 | Trung bình — vẫn fit 4GB nhưng sát ngưỡng khi chạy cùng lúc build |
| Jina-CLIP-v2 | Cạnh tranh với SigLIP2, hỗ trợ multilingual text tốt hơn rõ rệt (89 ngôn ngữ trong training) | Trung bình | ~900MB | **Tốt hơn** — có thể encode trực tiếp query tiếng Việt mà KHÔNG cần dịch qua Gemini | Trung bình |

→ **Đề xuất thử nghiệm cụ thể**: Jina-CLIP-v2 giải quyết trực tiếp vấn đề Priority 1 #1 (mục 3.9/5.2) vì nó tự hỗ trợ multilingual text encoder — loại bỏ hoàn toàn nhu cầu dịch Vi→En qua Gemini cho nhánh ảnh. Đây là thay đổi kiến trúc đáng cân nhắc nhất trong toàn bộ report này.

### 6.2 Caption / VLM

| Model | Chất lượng caption tiếng Việt | VRAM | Chạy Local | Tốc độ |
|---|---|---|---|---|
| **Qwen2-VL-2B-Instruct 4-bit (đang dùng)** | Khá — hiểu tiếng Việt nhưng đôi khi caption hơi generic | ~2GB | ✅ | Trung bình (chưa batch) |
| Qwen2.5-VL-3B-Instruct 4-bit | Tốt hơn rõ rệt (thế hệ mới hơn, cải thiện grounding + OCR-in-image) | ~2.5GB | ✅ | Chậm hơn chút, vẫn fit 4GB |
| InternVL2-2B | Cạnh tranh, đôi khi tốt hơn về mô tả không gian | ~2GB (4-bit) | ✅ | Tương đương |
| Gemini Flash (API, không dùng cho caption hiện tại nhưng đang dùng cho verify) | Tốt nhất chất lượng thuần túy | 0 (cloud) | ❌ | Nhanh nhưng phụ thuộc mạng |

→ Nếu muốn nâng chất lượng caption mà vẫn giữ nguyên tinh thần "local-first", **Qwen2.5-VL-3B 4-bit** là bước nâng cấp hợp lý nhất, rủi ro thấp, không đổi kiến trúc.

### 6.3 OCR

| Model | Tiếng Việt (có dấu) | Tiếng Anh | Tốc độ | Ghi chú |
|---|---|---|---|---|
| **EasyOCR vi+en (đang dùng)** | Trung bình | Tốt | Trung bình | Dễ cài, nhưng cộng đồng ghi nhận độ chính xác dấu tiếng Việt thấp hơn PaddleOCR trên văn bản nhỏ/mờ |
| PaddleOCR (PP-OCRv4, model tiếng Việt) | **Tốt hơn** trên benchmark cộng đồng, đặc biệt chữ nhỏ | Tốt | Nhanh hơn EasyOCR trên CPU | Cài đặt phức tạp hơn (PaddlePaddle framework riêng) |
| VietOCR (kết hợp text detector + VietOCR recognition) | Tốt nhất cho text tiếng Việt thuần, nhưng cần detector riêng (không all-in-one như EasyOCR) | Không tối ưu cho tiếng Anh | Trung bình | Phù hợp nếu OCR là điểm yếu chính sau khi benchmark |

→ **Không đổi ngay** — cần benchmark cụ thể (mục 7) trước khi quyết định, vì chi phí tích hợp lại (đổi cả detector + recognizer) không nhỏ.

### 6.4 ASR

| Model | Tiếng Việt | VRAM (RTX3050) | Tốc độ | Local |
|---|---|---|---|---|
| **faster-whisper medium (đang dùng)** | Khá tốt | ~1.5GB (fp16) | Nhanh | ✅ |
| faster-whisper large-v3, `int8_float16` | Tốt hơn rõ rệt | ~3GB | Chậm hơn ~2x | ✅ — khả thi vì đây là bước offline, không giới hạn thời gian như lúc thi |
| PhoWhisper (VinAI, fine-tune trên tiếng Việt) | **Tốt nhất cho tiếng Việt thuần** (train riêng cho vi, không đa ngôn ngữ) | Tùy size (base/large tương tự Whisper) | Tương đương | ✅ |

→ Nếu dataset AIC chủ yếu tiếng Việt (không lẫn nhiều tiếng Anh), **PhoWhisper** đáng thử song song với `large-v3 int8` — so sánh WER trên 5-10 video mẫu.

### 6.5 Text Embedding

| Model | Đa ngôn ngữ (vi) | Dim | Đặc điểm |
|---|---|---|---|
| **BGE-M3 (đang dùng)** | Tốt | 1024 | Hỗ trợ dense+sparse+ColBERT trong 1 model (đang chỉ dùng phần dense) |
| multilingual-e5-large | Tốt, cạnh tranh trực tiếp với BGE-M3 | 1024 | Nhẹ hơn chút, benchmark MTEB tương đương |
| Jina-embeddings-v3 | Tốt, hỗ trợ task-specific LoRA (retrieval/classification riêng) | 1024 | Đáng thử nếu cần embedding chuyên biệt hơn cho retrieval task |

→ **Không cần đổi** — BGE-M3 là lựa chọn hợp lý, ưu tiên khai thác thêm phần ColBERT-vecs sẵn có (mục 3.6) trước khi đổi model khác.

### 6.6 Re-ranking

| Model | Đa ngôn ngữ | Tốc độ | Ghi chú |
|---|---|---|---|
| **bge-reranker-v2-m3 (đang dùng)** | Tốt | Trung bình | Cross-encoder cổ điển, chính xác nhưng phải chạy tuần tự từng cặp (query, passage) |
| ColBERT-v2 (đa ngôn ngữ qua BGE-M3 ColBERT vecs) | Tốt (dùng chính output đã có sẵn từ BGE-M3) | **Nhanh hơn** cross-encoder vì late-interaction có thể precompute passage vectors | Tận dụng được phần ColBERT-vecs đang bỏ phí của BGE-M3 (mục 3.6) — đáng thử vì không cần load thêm model mới |

### 6.7 VLM (verify cuối)

| Model | Chất lượng | Local | Ghi chú |
|---|---|---|---|
| **Gemini Flash-Lite (đang dùng qua API)** | Tốt, nhanh, rẻ | ❌ | Rủi ro mạng lúc thi (mục 3.11) |
| Qwen2-VL-2B (model đã load sẵn cho caption) | Khá, không cần load thêm model | ✅ | Tái sử dụng model đã có, tránh phụ thuộc API — đáng cân nhắc làm **fallback local** khi không có Gemini, dù chất lượng thấp hơn Gemini |

---

## 7. Benchmark Nên Thực Hiện (trước khi đổi bất cứ model nào)

1. **Tạo eval set nội bộ**: 30-50 query mẫu (tiếng Việt có dấu + không dấu + tiếng Anh) trên 5-10 video từ BTC, gán nhãn ground-truth keyframe thủ công → đo Recall@10, Recall@50, mAP@10, nDCG@10 cho từng nhánh riêng (image-only, text-only, BM25-only) và cho pipeline đầy đủ.
2. **A/B test dịch query**: so sánh Recall nhánh ảnh khi (a) không dịch, (b) dịch qua Gemini, (c) dịch qua NLLB-200 local — để biết mục 5.2 có đáng làm hay không, và biết SigLIP2 thực sự cần dịch tới mức nào (SigLIP2 vẫn có phần train đa ngôn ngữ, có thể tốt hơn kỳ vọng với tiếng Việt thô).
3. **BM25 mismatch test**: bật `underthesea`, build lại BM25 index, chạy lại eval set → so sánh Recall trước/sau để định lượng chính xác thiệt hại của bug mục 5.1 (thay vì chỉ suy luận lý thuyết).
4. **OCR bake-off**: chạy EasyOCR vs PaddleOCR trên cùng 50-100 keyframe có chữ (biển hiệu, phụ đề, banner) → so Word Error Rate/CER thủ công.
5. **Whisper bake-off**: `medium` vs `large-v3 int8` vs PhoWhisper trên 5-10 clip có giọng nói rõ → so WER.
6. **Latency profiling toàn pipeline**: đo riêng từng bước (`extract_keyframes`, `build_index` — tách nhỏ theo từng hàm, `full_search` — tách `search_image`/`search_text`/`search_bm25`/`fuse_rrf`/`rerank`/`verify`) bằng `time.perf_counter()` đơn giản, để biết chính xác bottleneck thật thay vì đoán.
7. **RRF k-value + trọng số nhánh**: grid-search `RRF_K` (thử 20, 40, 60, 100) trên eval set — hiện tại `k=60` chỉ là giá trị mặc định theo literature, chưa tối ưu cho chính dataset của đội.

---

## 8. Paper / GitHub Nên Tham Khảo

- **SigLIP2** (Google, "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features") — hiểu rõ SigLIP2 thực sự đa ngôn ngữ tới đâu, để đánh giá đúng mục 3.9/7.2.
- **BGE-M3** (BAAI, "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings") — đặc biệt phần dense+sparse+ColBERT trong 1 model, liên quan trực tiếp mục 3.6/6.6.
- **Reciprocal Rank Fusion** (Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods") — bài gốc của công thức RRF đang dùng.
- **VBS (Video Browser Showdown) & LSC (Lifelog Search Challenge) — báo cáo kỹ thuật các đội top hàng năm** (thường public trên trang chính thức `videobrowsershowdown.org`) — đây là nguồn tham khảo thực chiến tốt nhất vì bài toán gần như giống hệt AIC, các đội top thường public chi tiết pipeline hybrid tương tự.
- **PhoWhisper** (VinAI Research, GitHub `VinAIResearch/PhoWhisper`) — liên quan mục 6.4.
- **underthesea** (GitHub `undertheseanlp/underthesea`) — đọc kỹ phần tokenizer để hiểu đúng format output (dấu `_` nối từ ghép), liên quan trực tiếp việc fix bug mục 5.1.
- **rank_bm25** (GitHub `dorianbrown/rank_bm25`) — đọc source `BM25Okapi.get_scores()` để chắc chắn hiểu đúng cách input tokens phải khớp giữa build và query.

---

## 9. Roadmap Tối Ưu (sắp xếp theo priority thực chiến)

### 🔴 Priority 1 — Sửa ngay, rẻ, tác động lớn
| Việc | Lợi ích | Độ khó | Thời gian | Recall dự kiến tăng | Đáng làm? |
|---|---|---|---|---|---|
| Fix BM25 tokenizer mismatch (mục 5.1) | Khôi phục đúng năng lực exact-match cho tên riêng/địa danh — chính lý do BM25 tồn tại | Thấp (1 hàm dùng chung) | 1-2 giờ | Có thể đáng kể nếu bật underthesea (chưa đo được, xem benchmark mục 7.3) | **Có, bắt buộc** |
| Fix heuristic dịch query (`isascii()` → language detection) | Tránh bỏ sót dịch cho tiếng Việt không dấu | Thấp | 1-2 giờ | Trung bình, tùy tỷ lệ query không dấu | **Có** |
| Thêm fallback dịch local (NLLB-200 hoặc PhoMT) khi Gemini lỗi/mất mạng | Loại bỏ rủi ro lớn nhất khi thi đấu không có/mạng chập chờn | Trung bình | 4-6 giờ | Không tăng recall trung bình nhưng **tránh sập nhánh ảnh hoàn toàn lúc thi** | **Có, ưu tiên vận hành** |
| Profile latency toàn pipeline (mục 7.6) | Biết chính xác bottleneck thật, tránh tối ưu nhầm chỗ | Thấp | 2 giờ | Gián tiếp | **Có** |

### 🟡 Priority 2 — Cải thiện chất lượng, cần benchmark trước
| Việc | Lợi ích | Độ khó | Thời gian | Recall dự kiến tăng | Đáng làm? |
|---|---|---|---|---|---|
| Tạo eval set 30-50 query + ground truth (mục 7.1) | Nền tảng để đo mọi thay đổi sau này bằng số liệu thay vì cảm tính | Trung bình | 4-6 giờ | Gián tiếp nhưng **là điều kiện tiên quyết cho mọi tối ưu khác** | **Có, làm sớm nhất có thể** |
| Batch hóa Qwen2-VL caption (mục 5.3) | Giảm thời gian build đáng kể (bottleneck build-time lớn nhất) | Trung bình | 3-4 giờ | Không tăng recall, tăng tốc độ build | Có, nếu build đang chậm |
| OCR bake-off EasyOCR vs PaddleOCR (mục 6.3, 7.4) | Biết có nên đổi OCR không dựa trên số liệu thật | Thấp (chỉ chạy benchmark, chưa đổi code chính) | 3-4 giờ | Chưa biết — phụ thuộc kết quả | Có, trước khi quyết định đổi |
| Grid-search RRF_K + so sánh weighted fusion có BM25 | Tối ưu tham số fusion theo đúng dataset thay vì giá trị mặc định | Thấp | 2-3 giờ | Nhỏ-trung bình | Có |

### 🟢 Priority 3 — Nâng cấp model, rủi ro tích hợp cao hơn, chỉ làm khi Priority 1-2 đã ổn
| Việc | Lợi ích | Độ khó | Thời gian | Recall dự kiến tăng | Đáng làm? |
|---|---|---|---|---|---|
| Thử Jina-CLIP-v2 thay SigLIP2 (mục 6.1) | Loại bỏ hoàn toàn nhu cầu dịch Vi→En cho nhánh ảnh — giải quyết tận gốc vấn đề Priority 1 | Cao (đổi cả image index, dim khác, phải build lại từ đầu) | 1-2 ngày | Tiềm năng cao nhưng **chưa được đo** — cần benchmark A/B trước khi cam kết | Cân nhắc, không vội |
| Qwen2.5-VL-3B thay Qwen2-VL-2B cho caption | Caption chất lượng hơn | Trung bình | 4-6 giờ | Nhỏ-trung bình | Có, rủi ro thấp |
| PhoWhisper thay/song song faster-whisper | ASR tiếng Việt tốt hơn | Trung bình | 4-6 giờ | Trung bình nếu dataset nhiều tiếng Việt thuần | Có, sau benchmark 7.5 |
| Dùng ColBERT-vecs của BGE-M3 thay/bổ sung CrossEncoder rerank | Rerank nhanh hơn, tận dụng tài nguyên đã có | Cao (cần code late-interaction scoring, chưa có sẵn) | 1 ngày+ | Chưa rõ, cần thử nghiệm | Thử nghiệm nếu còn thời gian, không phải việc cấp thiết |

---

## 10. Pipeline Cuối Cùng Khuyến Nghị

```
OFFLINE (giữ nguyên phần lớn kiến trúc hiện tại — đây là điểm mạnh):
  PySceneDetect (giữ nguyên, có thể tinh chỉnh threshold theo dataset thật)
        │
        ├─ faster-whisper medium (giữ, hoặc thử large-v3 int8_float16 sau benchmark 7.5)
        │
        └─ Keyframe (3/scene, giữ nguyên)
                ├─ SigLIP2 ViT-B-16 (giữ, hoặc Jina-CLIP-v2 SAU KHI benchmark 6.1 xác nhận lợi ích)
                ├─ Qwen2-VL-2B 4-bit, BATCH HÓA (mục 5.3)
                ├─ EasyOCR (giữ, hoặc PaddleOCR SAU khi bake-off 6.3)
                └─ Transcript
                        │
                        ▼
        BGE-M3 (dense) + BM25 (FIX TOKENIZER — mục 5.1, dùng CHUNG hàm tokenize
        cho cả build và search, lưu flag "used_underthesea" vào pickle)

ONLINE (search):
  query
    ├─ language detection LOCAL (langid/fasttext) thay vì isascii()
    ├─ [ảnh] nếu cần dịch: Gemini (nhanh) VỚI fallback NLLB-200 local khi lỗi/mất mạng
    │         → SigLIP2/Jina-CLIP text encoder → FAISS Image Index
    ├─ [text] BGE-M3 (CPU) → FAISS Text Index
    └─ [BM25] tokenize ĐÚNG như lúc build → BM25Okapi
                    │
                    ▼
         RRF fusion (k đã grid-search theo eval set riêng — mục 7.7)
                    │
                    ▼
    CrossEncoder bge-reranker-v2-m3 (giữ nguyên — đã fix bug fallback passage tốt)
                    │
                    ▼ (tùy chọn, có fallback local Qwen2-VL nếu Gemini lỗi)
         Gemini VLM verify → Qwen2-VL-2B fallback nếu API lỗi
```

**Nguyên tắc chỉ đạo cho mọi thay đổi**: *không đổi gì nếu chưa có eval set để đo* (mục 7.1) — đây là khoảng trống lớn nhất hiện tại của repo. Kiến trúc đã đủ tốt để cạnh tranh; việc còn thiếu là **năng lực đo lường** để biết thay đổi nào thực sự giúp Recall tăng trên chính dataset của đội, thay vì tin vào lý thuyết chung chung.

---

*Report dựa trên đọc trực tiếp source code tại thời điểm 29/07/2026 (commit mới nhất trên nhánh `main` của `khoahaaoj/AI-Challenge-Model01`). Mọi số liệu VRAM/tốc độ là ước tính dựa trên kiến thức chung về các model — cần benchmark thực tế trên máy của đội (mục 7) để có số liệu chính xác.*
