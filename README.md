# AIC 2026 (Bảng A) — Multimedia Retrieval Pipeline

Hệ thống tìm kiếm keyframe/video từ truy vấn văn bản (vi/en), theo kiến trúc:

```
VIDEO ─┬─ PySceneDetect ──────────► Keyframes
       └─ Whisper (vi+en) ───────► Transcript

Keyframes ─┬─ CLIP (image)  ──────────────► FAISS Image Index
           ├─ Gwen2-VL caption(en) ─ BGE-M3 ─┐
           └─ EasyOCR (vi+en)   ─ BGE-M3 ──┼──► FAISS Text Index
Transcript ────────────────────── BGE-M3 ──┘

QUERY(vi/en) ─┬─ CLIP text  → search Image Index → top50 ─┐
              └─ BGE-M3     → search Text  Index → top50 ─┼─► Hybrid Fusion (RRF) → top50
                                                             │
                                          CrossEncoder rerank ▼ top10
                                                             │
                                        Gemini/VLM verify ────► Kết quả cuối
```

## 1. Cài đặt

```bash
# 1. (khuyến nghị) tạo virtualenv riêng
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Cài torch đúng bản CUDA của máy TRƯỚC (kiểm tra CUDA version bằng `nvidia-smi`)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Cài các package còn lại
pip install -r requirements.txt

# 4. (tuỳ chọn nhưng nên có) cấu hình Gemini API key — dùng cho caption + verify
#    Lấy key miễn phí tại https://aistudio.google.com/apikey
setx GEMINI_API_KEY "your-key-here"      # Windows (cần mở lại terminal sau khi setx)
# export GEMINI_API_KEY="your-key-here"  # Linux/Mac
```

Nếu KHÔNG cấu hình `GEMINI_API_KEY`, pipeline vẫn chạy được bình thường — chỉ mất
tín hiệu caption trong nhánh text và bỏ qua bước verify cuối (rerank vẫn hoạt động
dựa trên CLIP + OCR + transcript).

## 2. Chạy pipeline

```bash
# Bước 0: copy video vào data/videos/

# Bước 1: tách keyframe + transcript (cache ra cache/*.json + data/keyframes/)
python extract_keyframes.py

# Bước 2: build FAISS index (CLIP image index + BGE-M3 text index)
python build_index.py

# Bước 3: chạy UI tìm kiếm
streamlit run app.py
```

Mỗi bước đều **cache theo video/ảnh cụ thể** — nếu bị lỗi hoặc dừng giữa chừng,
chạy lại đúng lệnh trên sẽ tiếp tục từ chỗ dang dở, không phải làm lại từ đầu.
Muốn build lại từ đầu 1 bước nào đó, xoá file cache tương ứng trong `cache/` hoặc
`index/` rồi chạy lại.

## 3. Cấu trúc thư mục

```
config.py              # cấu hình trung tâm (path, tên model, tham số)
extract_keyframes.py   # bước 1: PySceneDetect + Whisper
build_index.py         # bước 2: CLIP/Gemini/EasyOCR/BGE-M3 -> FAISS index
search_utils.py        # bước 3: logic search + fusion + rerank + verify
app.py                 # UI Streamlit
data/videos/            # <- bỏ video gốc vào đây
data/keyframes/          # ảnh keyframe được sinh ra tự động
cache/                  # cache JSON: keyframe_meta, transcripts, captions, ocr
index/                  # FAISS index đã build (.faiss + id_map.json)
```

## 4. Troubleshooting thường gặp

- **`RuntimeError: Chưa có index/image_index.faiss`** → chưa chạy `build_index.py`.
- **`FlagEmbedding` cài lỗi** → cần `pip install -U FlagEmbedding`, đôi khi cần cài
  thêm `pip install -U transformers accelerate`.
- **EasyOCR tải model rất chậm lần đầu** → model tải về `~/.EasyOCR/model`, chỉ chậm
  lần đầu tiên, các lần sau nhanh.
- **Hết VRAM khi build_index.py** → giảm `batch_size` trong `model.encode(...)` ở
  `build_text_index()`, hoặc đổi `CLIP_MODEL_NAME` sang bản nhỏ hơn trong `config.py`.
- **Gemini bị rate limit (429)** → `build_captions()` đã tự `time.sleep(2)` khi lỗi
  và cache theo từng ảnh, chỉ cần chạy lại `build_index.py` sau vài phút.
