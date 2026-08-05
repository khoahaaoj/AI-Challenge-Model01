"""
build_ocr_features.py
======================
OCR toàn bộ keyframe BTC → cache/ocr.json

Dùng EasyOCR (vi+en) với GPU. Checkpoint tự động mỗi 500 ảnh.
Nếu bị ngắt giữa chừng, chạy lại sẽ tiếp tục từ chỗ dở.

Sau khi chạy xong, cần rebuild text index:
    python build_btc_text_index.py
"""
import os
import json
import torch
from tqdm import tqdm
import config


def build_ocr_for_btc():
    try:
        import easyocr
    except ImportError:
        print("❌ Chưa cài easyocr. Chạy: pip install easyocr")
        return

    if not os.path.exists(config.KEYFRAME_DIR):
        print(f"❌ Thư mục {config.KEYFRAME_DIR} không tồn tại. Tải keyframes trước!")
        return

    # Load cache hiện có (checkpoint)
    ocr_cache = {}
    if os.path.exists(config.OCR_CACHE):
        with open(config.OCR_CACHE, "r", encoding="utf-8") as f:
            ocr_cache = json.load(f)

    # Tìm toàn bộ ảnh cần xử lý
    print("[OCR] Đang quét danh sách keyframe...")
    all_paths = []
    for root, _, files in os.walk(config.KEYFRAME_DIR):
        for file in sorted(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_paths.append(os.path.join(root, file))

    pending = [p for p in all_paths if p not in ocr_cache]
    total = len(all_paths)
    done = total - len(pending)
    print(f"[OCR] Tổng: {total:,} ảnh | Đã xong: {done:,} | Còn lại: {len(pending):,}")

    if not pending:
        print("🎉 Đã hoàn thành OCR cho tất cả ảnh!")
        return

    # Khởi tạo EasyOCR — GPU nếu có
    use_gpu = torch.cuda.is_available()
    print(f"[OCR] Khởi tạo EasyOCR (langs={config.OCR_LANGS}, gpu={use_gpu})...")
    print("[OCR] ⚠️  Nếu bị OOM: tắt Streamlit trước rồi chạy lại!\n")
    reader = easyocr.Reader(config.OCR_LANGS, gpu=use_gpu)

    SAVE_EVERY = 500
    success = 0

    for i, path in enumerate(tqdm(pending, desc="[EasyOCR]")):
        try:
            results = reader.readtext(path, detail=1)
            # Lọc theo confidence > 0.3
            texts = [res[1] for res in results if res[2] > 0.3]
            ocr_cache[path] = " ".join(texts).strip()
            success += 1
        except Exception as e:
            ocr_cache[path] = ""

        # Checkpoint định kỳ
        if (i + 1) % SAVE_EVERY == 0:
            with open(config.OCR_CACHE, "w", encoding="utf-8") as f:
                json.dump(ocr_cache, f, ensure_ascii=False)
            tqdm.write(f"[OCR] ✅ Checkpoint: {len(ocr_cache):,} ảnh đã lưu")

    # Lưu lần cuối
    with open(config.OCR_CACHE, "w", encoding="utf-8") as f:
        json.dump(ocr_cache, f, ensure_ascii=False)

    has_text = sum(1 for v in ocr_cache.values() if v.strip())
    print(f"\n✅ Hoàn thành! {success:,} ảnh mới | Có text: {has_text:,}/{len(ocr_cache):,}")
    print(f"   → {config.OCR_CACHE}")
    print("\n⚠️  Nhớ rebuild text index sau khi xong:")
    print("   python build_btc_text_index.py")


if __name__ == "__main__":
    build_ocr_for_btc()
