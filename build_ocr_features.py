import os
import json
import torch
from tqdm import tqdm

import config

def build_ocr_for_btc():
    """
    Trích xuất OCR cho tất cả keyframes trong data/keyframes.
    Sử dụng EasyOCR (nếu có cài đặt) hoặc báo lỗi.
    Lưu kết quả vào cache/ocr.json.
    """
    try:
        import easyocr
    except ImportError:
        print("❌ Chưa cài đặt easyocr. Vui lòng chạy: pip install easyocr")
        return

    print(f"[OCR] Khởi tạo EasyOCR cho ngôn ngữ {config.OCR_LANGS} (chạy trên CPU để tránh tràn VRAM)...")
    # Tắt GPU vì Streamlit (BGE-M3, SigLIP) đang chiếm dụng VRAM
    reader = easyocr.Reader(config.OCR_LANGS, gpu=False)

    if not os.path.exists(config.KEYFRAME_DIR):
        print(f"❌ Thư mục {config.KEYFRAME_DIR} không tồn tại. Vui lòng tải keyframes trước!")
        return

    ocr_cache_path = config.OCR_CACHE
    if os.path.exists(ocr_cache_path):
        with open(ocr_cache_path, "r", encoding="utf-8") as f:
            ocr_cache = json.load(f)
    else:
        ocr_cache = {}

    print("[OCR] Đang quét danh sách ảnh keyframe...")
    all_image_paths = []
    for root, dirs, files in os.walk(config.KEYFRAME_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Lưu đường dẫn tương đối (ví dụ: L10_V010/001.jpg) 
                # hoặc tuyệt đối tùy nhu cầu, ở đây lưu tuyệt đối để dễ đọc
                full_path = os.path.join(root, file)
                all_image_paths.append(full_path)

    print(f"[OCR] Tìm thấy {len(all_image_paths)} ảnh trong {config.KEYFRAME_DIR}.")
    
    pending_paths = [p for p in all_image_paths if p not in ocr_cache]
    print(f"[OCR] Còn lại {len(pending_paths)} ảnh chưa được OCR.")

    if not pending_paths:
        print("🎉 Đã hoàn thành OCR cho tất cả ảnh!")
        return

    save_interval = 200
    success_count = 0

    for i, path in enumerate(tqdm(pending_paths, desc="OCR Keyframes")):
        try:
            results = reader.readtext(path)
            # results: list of (bbox, text, confidence)
            # Lọc text có confidence > 0.3
            texts = [res[1] for res in results if res[2] > 0.3]
            ocr_text = " ".join(texts).strip()
            
            ocr_cache[path] = ocr_text
            success_count += 1
        except Exception as e:
            print(f"Lỗi đọc {path}: {e}")
            ocr_cache[path] = ""

        # Lưu lại đều đặn để tránh mất dữ liệu khi bị ngắt
        if (i + 1) % save_interval == 0:
            with open(ocr_cache_path, "w", encoding="utf-8") as f:
                json.dump(ocr_cache, f, ensure_ascii=False, indent=2)

    # Lưu lần cuối
    with open(ocr_cache_path, "w", encoding="utf-8") as f:
        json.dump(ocr_cache, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Hoàn thành! Đã OCR thành công {success_count} ảnh.")
    print(f"   Kết quả được lưu tại: {ocr_cache_path}")

if __name__ == "__main__":
    build_ocr_for_btc()
