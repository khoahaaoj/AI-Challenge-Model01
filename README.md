# AIC 2026 (Bảng A) — Multimedia Retrieval Pipeline

Hệ thống tìm kiếm keyframe/video từ truy vấn văn bản (vi/en), theo kiến trúc:

```
VIDEO ─┬─ PySceneDetect ──────────► Keyframes
       └─ Whisper (vi+en) ───────► Transcript

Keyframes ─┬─ CLIP (image)  ──────────────► FAISS Image Index
           ├─ Gemini caption(en) ─ BGE-M3 ─┐
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

## 3. Dataset để luyện tập (ĐÃ ĐƯỢC ĐẶT SẴN VÀO `data/videos/`)

Mình đã tải sẵn **6 video mẫu nhẹ (~24MB tổng)** từ repo
[`intel-iot-devkit/sample-videos`](https://github.com/intel-iot-devkit/sample-videos)
(license **CC-BY 4.0** — dùng thoải mái cho mục đích luyện tập/nghiên cứu, chỉ cần
giữ ghi công) và bỏ thẳng vào `data/videos/` — bạn **không cần tải gì thêm**,
chạy `extract_keyframes.py` là dùng được ngay:

| File | Nội dung | Dung lượng |
|---|---|---|
| `bottle-detection.mp4` | chai lọ trên băng chuyền | 0.5 MB |
| `car-detection.mp4` | xe cộ ngoài đường | 2.8 MB |
| `one-by-one-person-detection.mp4` | người đi qua camera lần lượt | 3.2 MB |
| `people-detection.mp4` | nhiều người đi lại | 5.4 MB |
| `person-bicycle-car-detection.mp4` | người + xe đạp + ô tô | 6.0 MB |
| `face-demographics-walking.mp4` | người đi bộ, cận mặt | 6.4 MB |

**Lưu ý quan trọng:** đây là video giám sát (surveillance demo) của Intel/OpenVINO,
**hầu như không có lời thoại** — nên nhánh Whisper (transcript) sẽ gần như rỗng với
bộ này. Bộ dataset này dùng để test NHANH phần cơ khí của pipeline (tách cảnh →
keyframe → CLIP → OCR → FAISS → search ảnh) bằng tiếng Anh (vd: query thử
"a person riding a bicycle", "bottles on a conveyor belt"). Khi cần test riêng
nhánh Whisper tiếng Việt + OCR tiếng Việt + query tiếng Việt, xem gợi ý (c) bên dưới
để có thêm video tiếng Việt thật.

Ngoài bộ đã có sẵn, vẫn còn 2 nguồn khác nếu muốn mở rộng:

**a) BetterDay-Tool (open-source từ chính AI Challenge HCMC 2023)**
Repo: `github.com/Nhathuy1305/BetterDay-Tool` — đây là 1 video search engine
mã nguồn mở được đơn giản hoá từ prototype dự thi AIC 2023 thật, kèm theo
1 bộ demo dataset nhỏ (tải qua Weaviate Cloud dataset trong repo). Rất đáng xem
qua vì gần sát nhất với format dữ liệu (tin tức tiếng Việt) và cách các đội khác
đã giải bài toán này.

**b) MSR-VTT (subset nhỏ)**
Bộ dữ liệu video-text retrieval chuẩn trong nghiên cứu (10.000 video clip ngắn
kèm caption tiếng Anh). Chỉ cần tải một vài chục clip đầu là đủ để test cơ khí
toàn bộ pipeline (tách cảnh → keyframe → CLIP/BGE-M3 → FAISS → search) mà không
cần tải hết vài chục GB. Phù hợp để kiểm tra pipeline chạy đúng logic (tiếng Anh),
trước khi thử với video tiếng Việt thật.

**c) Video tiếng Việt thật (để test riêng nhánh Whisper vi + OCR vi + query vi)**
Tự tải vài video công khai (tin tức, vlog ngắn) bằng `yt-dlp` để có dữ liệu tiếng
Việt thật kiểm tra chất lượng ASR/OCR tiếng Việt — vì MSR-VTT và phần lớn dataset
academic chuẩn đều chỉ có tiếng Anh. Nhớ chỉ dùng cho mục đích thử nghiệm nội bộ,
không dùng lại nội dung có bản quyền cho sản phẩm cuối.

> Khi có dữ liệu thật từ BTC: cấu trúc thư mục/pipeline không đổi, chỉ cần copy
> video thật vào `data/videos/` và chạy lại bước 1–2. Với dataset lớn (hàng nghìn
> giờ video như các năm trước — AIC 2024 có 1.471 video / 328 giờ), nên cân nhắc
> đổi `faiss.IndexFlatIP` sang `faiss.IndexIVFFlat` hoặc `IndexHNSWFlat` để search
> nhanh hơn (Flat index chỉ phù hợp tới vài trăm nghìn vector).

## 4. Cấu trúc thư mục

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

## 5. Troubleshooting thường gặp

- **`RuntimeError: Chưa có index/image_index.faiss`** → chưa chạy `build_index.py`.
- **`FlagEmbedding` cài lỗi** → cần `pip install -U FlagEmbedding`, đôi khi cần cài
  thêm `pip install -U transformers accelerate`.
- **EasyOCR tải model rất chậm lần đầu** → model tải về `~/.EasyOCR/model`, chỉ chậm
  lần đầu tiên, các lần sau nhanh.
- **Hết VRAM khi build_index.py** → giảm `batch_size` trong `model.encode(...)` ở
  `build_text_index()`, hoặc đổi `CLIP_MODEL_NAME` sang bản nhỏ hơn trong `config.py`.
- **Gemini bị rate limit (429)** → `build_captions()` đã tự `time.sleep(2)` khi lỗi
  và cache theo từng ảnh, chỉ cần chạy lại `build_index.py` sau vài phút.
