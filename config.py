"""
config.py
==========
File cấu hình TRUNG TÂM cho toàn bộ pipeline.
Mọi file khác (extract_keyframes.py, build_index.py, search_utils.py, app.py)
đều `import config` để lấy đường dẫn / tên model / tham số từ đây.

-> Muốn đổi model, đổi tham số, đổi thư mục dữ liệu... chỉ cần sửa Ở ĐÂY,
   không cần lục lại từng file.
"""

import os
import torch
from dotenv import load_dotenv
load_dotenv()  # Tự động đọc các biến trong file .env

# =================================================================
# THIẾT BỊ (GPU nếu có, không thì fallback CPU)
# =================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[config] Đang chạy trên thiết bị: {DEVICE}")

# =================================================================
# ĐƯỜNG DẪN THƯ MỤC
# =================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_DIR    = os.path.join(BASE_DIR, "data", "videos")      # nơi bạn bỏ video gốc vào
KEYFRAME_DIR = os.path.join(BASE_DIR, "data", "keyframes")  # ảnh keyframe được sinh ra (tự trích)
CACHE_DIR    = os.path.join(BASE_DIR, "cache")               # cache JSON (transcript, caption, ocr...)
INDEX_DIR    = os.path.join(BASE_DIR, "index")               # FAISS index đã build

# ---- Thư mục dữ liệu chính thức của BTC ----
BTC_DIR       = os.path.join(BASE_DIR, "data", "btc")
BTC_MAP_KF    = os.path.join(BTC_DIR, "map_keyframes", "map-keyframes")   # 873 CSV keyframe maps
BTC_MEDIA_DIR = os.path.join(BTC_DIR, "media_info",   "media-info")       # 873 JSON YouTube metadata
BTC_CLIP_DIR  = os.path.join(BTC_DIR, "clip_features", "clip-features-32") # 873 .npy CLIP features
BTC_OBJ_DIR   = os.path.join(BTC_DIR, "objects")                           # object detection JSON (nếu đã tải)

for _d in [VIDEO_DIR, KEYFRAME_DIR, CACHE_DIR, INDEX_DIR, BTC_DIR]:
    os.makedirs(_d, exist_ok=True)

# ---- Các file cache cụ thể ----
KEYFRAME_META_CACHE  = os.path.join(CACHE_DIR, "keyframe_meta.json")   # metadata keyframe tự trích (cũ)
BTC_KF_META_CACHE    = os.path.join(CACHE_DIR, "btc_keyframe_meta.json") # metadata keyframe BTC (chính thức)
BTC_MEDIA_INFO_CACHE = os.path.join(CACHE_DIR, "btc_media_info.json")   # YouTube metadata BTC
TRANSCRIPT_CACHE     = os.path.join(CACHE_DIR, "transcripts.json")      # transcript Whisper
CAPTION_CACHE        = os.path.join(CACHE_DIR, "captions.json")         # caption Qwen2-VL
OCR_CACHE            = os.path.join(CACHE_DIR, "ocr.json")              # text OCR

# =================================================================
# CHỌN NGUỒN DỮ LIỆU: BTC (chính thức) vs TỰ TRÍCH (cũ)
# =================================================================
# USE_BTC_DATA = True  → dùng keyframe map + CLIP features của BTC
#   - frame_idx khớp với ground truth BTC → điểm chính xác
#   - Image FAISS dim=512 (ViT-B/32), 177.321 keyframes
# USE_BTC_DATA = False → dùng keyframe tự trích (chỉ 2 video demo)
USE_BTC_DATA = True

# ---- Các file FAISS index ----
if USE_BTC_DATA:
    IMAGE_INDEX_PATH  = os.path.join(INDEX_DIR, "btc_image_index.faiss")  # BTC CLIP ViT-B/32, dim=512
    IMAGE_ID_MAP_PATH = os.path.join(INDEX_DIR, "btc_image_id_map.json")
    TEXT_INDEX_PATH   = os.path.join(INDEX_DIR, "btc_text_index.faiss")
    TEXT_ID_MAP_PATH  = os.path.join(INDEX_DIR, "btc_text_id_map.json")
    BM25_INDEX_PATH   = os.path.join(INDEX_DIR, "btc_bm25_index.pkl")
else:
    IMAGE_INDEX_PATH  = os.path.join(INDEX_DIR, "image_index.faiss")      # SigLIP2, dim=768
    IMAGE_ID_MAP_PATH = os.path.join(INDEX_DIR, "image_id_map.json")
    TEXT_INDEX_PATH   = os.path.join(INDEX_DIR, "text_index.faiss")
    TEXT_ID_MAP_PATH  = os.path.join(INDEX_DIR, "text_id_map.json")
    BM25_INDEX_PATH   = os.path.join(INDEX_DIR, "bm25_index.pkl")

# =================================================================
# BƯỚC TÁCH CẢNH (PySceneDetect)
# =================================================================
SCENE_THRESHOLD = 27.0   # ngưỡng đổi cảnh của ContentDetector
MIN_SCENE_LEN = 15       # số frame tối thiểu của 1 cảnh (tránh tách quá vụn)

# Vị trí lấy keyframe trong mỗi scene (theo tỷ lệ % độ dài scene).
# [0.25, 0.5, 0.75] = 3 frame/scene (đầu, giữa, cuối) -> tăng recall ~3x.
# Đổi về [0.5] để quay lại chế độ 1 frame/scene nếu cần tiết kiệm thời gian build.
KEYFRAME_POSITIONS = [0.25, 0.5, 0.75]

# =================================================================
# WHISPER (Speech-to-Text, đa ngôn ngữ - hỗ trợ tiếng Việt)
# =================================================================
# faster-whisper cho tốc độ nhanh hơn nhiều so với openai-whisper gốc, đặc biệt trên GPU.
# "large-v3" cho chất lượng tiếng Việt tốt nhất trong họ Whisper hiện tại.
# Nếu máy yếu / muốn build nhanh khi thử nghiệm, có thể đổi sang "medium" hoặc "small".
WHISPER_MODEL_SIZE = "medium"
WHISPER_COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# =================================================================
# CLIP (Image encoder - nhánh tìm theo nội dung hình ảnh)
# =================================================================
# Khi USE_BTC_DATA=True: dùng ViT-B/32 OpenAI (dim=512) để khớp với
# CLIP features đã có sẵn của BTC (không cần re-encode ảnh).
# Khi USE_BTC_DATA=False: dùng SigLIP2 ViT-B-16 (dim=768, chất lượng cao hơn)
# nhưng phải tự encode keyframe từ ảnh → tốn VRAM + thời gian.
# ⚠️  CLIP text encoder phải CÙNG model với image encoder đã dùng khi build index!
if USE_BTC_DATA:
    CLIP_MODEL_NAME = "ViT-B-32"        # OpenAI ViT-B/32 — khớp với BTC CLIP features
    CLIP_PRETRAINED = "openai"          # pretrained weights
else:
    CLIP_MODEL_NAME = "hf-hub:timm/ViT-B-16-SigLIP2"  # SigLIP2, chất lượng cao hơn
    CLIP_PRETRAINED = None
# Batch size khi encode ảnh (build_index.py). 16 an toàn với 4GB VRAM.
CLIP_BATCH_SIZE = 16

# =================================================================
# BGE-M3 (Text embedding đa ngôn ngữ - hỗ trợ tốt cả vi & en)
# =================================================================
BGE_MODEL_NAME = "BAAI/bge-m3"

# =================================================================
# CROSS-ENCODER RERANKER (đa ngôn ngữ, cùng họ BGE)
# =================================================================
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# =================================================================
# GEMINI API (caption ảnh + verify bước cuối)
# =================================================================
# Đặt API key qua biến môi trường, KHÔNG hard-code key vào code:
#   Windows (PowerShell):  $env:GEMINI_API_KEY = "xxxx"
#   Linux/Mac:              export GEMINI_API_KEY="xxxx"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.5-flash-lite"  # nhanh, rẻ, đủ tốt cho caption + verify hàng loạt

# =================================================================
# EASYOCR
# =================================================================
OCR_LANGS = ["vi", "en"]  # đọc được cả chữ tiếng Việt (có dấu) lẫn tiếng Anh trong khung hình

# =================================================================
# THAM SỐ TÌM KIẾM (search / fusion / rerank)
# =================================================================
TOP_K_RETRIEVE = 100     # số kết quả lấy ra ở mỗi nhánh trước khi fusion
TOP_K_RERANK   = 100     # BTC cho phép nộp tối đa 100 câu → rerank hết 100 để tận dụng R@50/R@100

FUSION_METHOD = "rrf"    # "rrf" (khuyến nghị) hoặc "weighted"
RRF_K = 60                # hằng số k trong công thức Reciprocal Rank Fusion (giá trị phổ biến trong literature)
IMAGE_WEIGHT = 0.5        # trọng số nhánh ảnh (chỉ dùng khi FUSION_METHOD = "weighted")
TEXT_WEIGHT = 0.5         # trọng số nhánh text (chỉ dùng khi FUSION_METHOD = "weighted")
