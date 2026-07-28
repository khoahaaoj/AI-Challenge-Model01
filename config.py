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

VIDEO_DIR = os.path.join(BASE_DIR, "data", "videos")        # nơi bạn bỏ video gốc vào
KEYFRAME_DIR = os.path.join(BASE_DIR, "data", "keyframes")  # ảnh keyframe được sinh ra
CACHE_DIR = os.path.join(BASE_DIR, "cache")                 # cache JSON (transcript, caption, ocr...)
INDEX_DIR = os.path.join(BASE_DIR, "index")                 # FAISS index đã build

for _d in [VIDEO_DIR, KEYFRAME_DIR, CACHE_DIR, INDEX_DIR]:
    os.makedirs(_d, exist_ok=True)

# ---- Các file cache cụ thể ----
KEYFRAME_META_CACHE = os.path.join(CACHE_DIR, "keyframe_meta.json")  # metadata keyframe: video_id, frame_idx, timestamp, path
TRANSCRIPT_CACHE = os.path.join(CACHE_DIR, "transcripts.json")       # transcript Whisper theo từng video
CAPTION_CACHE = os.path.join(CACHE_DIR, "captions.json")             # caption Qwen2-VL theo từng ảnh keyframe (key = path ảnh)
OCR_CACHE = os.path.join(CACHE_DIR, "ocr.json")                      # text OCR theo từng ảnh keyframe (key = path ảnh)

# ---- Các file FAISS index ----
IMAGE_INDEX_PATH = os.path.join(INDEX_DIR, "image_index.faiss")
IMAGE_ID_MAP_PATH = os.path.join(INDEX_DIR, "image_id_map.json")     # vị trí trong FAISS -> metadata keyframe
TEXT_INDEX_PATH = os.path.join(INDEX_DIR, "text_index.faiss")
TEXT_ID_MAP_PATH = os.path.join(INDEX_DIR, "text_id_map.json")       # vị trí trong FAISS -> record text (caption/ocr/transcript)

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
# open_clip: ViT-B-32 bản gốc OpenAI - nhẹ, chạy nhanh, đủ tốt để bắt đầu.
# Nếu cần độ chính xác cao hơn (và có GPU khỏe), có thể nâng cấp lên:
#   CLIP_MODEL_NAME = "ViT-L-14", CLIP_PRETRAINED = "openai"
# Lưu ý: CLIP gốc train chủ yếu bằng tiếng Anh -> query tiếng Việt sẽ kém chính xác hơn
# ở NHÁNH NÀY, đó là lý do pipeline có thêm nhánh BGE-M3 (đa ngôn ngữ) để bù lại.
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
# Batch size khi encode ảnh bằng CLIP (build_index.py).
# 16 = an toàn với 4GB VRAM. Tăng lên 32/64 nếu VRAM lớn hơn để nhanh hơn.
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
TOP_K_RETRIEVE = 50      # số kết quả lấy ra ở mỗi nhánh (CLIP / BGE-M3) trước khi fusion
TOP_K_RERANK = 10        # số kết quả cuối cùng sau khi CrossEncoder rerank

FUSION_METHOD = "rrf"    # "rrf" (khuyến nghị) hoặc "weighted"
RRF_K = 60                # hằng số k trong công thức Reciprocal Rank Fusion (giá trị phổ biến trong literature)
IMAGE_WEIGHT = 0.5        # trọng số nhánh ảnh (chỉ dùng khi FUSION_METHOD = "weighted")
TEXT_WEIGHT = 0.5         # trọng số nhánh text (chỉ dùng khi FUSION_METHOD = "weighted")
