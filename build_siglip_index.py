"""
build_siglip_index.py
======================
Encode toàn bộ keyframe BTC bằng SigLIP2-SO400M (dim=1152) → FAISS index.

Tại sao cần script riêng thay vì dùng BTC CLIP features?
  - BTC chỉ cung cấp CLIP ViT-B/32 (dim=512) — chất lượng trung bình
  - SigLIP2-SO400M vượt trội ~15-20% trên visual retrieval benchmark
  - Chạy song song: nhánh CLIP (BTC) + nhánh SigLIP2 (self-encoded)
    → fusion 4 nhánh → recall tốt hơn đáng kể

Yêu cầu:
  - data/keyframes/<video_id>/*.jpg đã có sẵn
  - GPU khuyến nghị (CPU chạy được nhưng ~10x chậm hơn)
  - ~2GB VRAM cho model, batch_size=16

Kết quả:
  - index/siglip_image_index.faiss   (dim=1152)
  - index/siglip_image_id_map.json

Sau khi chạy xong, SigLIP2 tự động được thêm vào pipeline fusion khi search.
Checkpoint mỗi 1000 ảnh — nếu bị ngắt giữa chừng, chạy lại sẽ tiếp tục.
"""

import os
import json
import numpy as np
import faiss
from tqdm import tqdm
import torch

import config

SIGLIP_INDEX_PATH  = os.path.join(config.INDEX_DIR, "siglip_image_index.faiss")
SIGLIP_ID_MAP_PATH = os.path.join(config.INDEX_DIR, "siglip_image_id_map.json")
SIGLIP_PARTIAL_PATH = os.path.join(config.INDEX_DIR, "siglip_partial.npz")  # checkpoint

MODEL_NAME   = "hf-hub:timm/ViT-SO400M-14-SigLIP2-384"
BATCH_SIZE   = 8    # an toàn với 4GB VRAM; tăng lên 16 nếu không bị OOM
SAVE_EVERY   = 1000  # checkpoint mỗi N ảnh


def _load_siglip_model():
    """Load SigLIP2-SO400M model từ HuggingFace Hub (tải ~3.5GB lần đầu)."""
    import open_clip
    print(f"[SigLIP2] Load model: {MODEL_NAME}")
    print("[SigLIP2] Lần đầu sẽ tải ~3.5GB từ HuggingFace, sau đó cache tại ~/.cache/")
    model, preprocess = open_clip.create_model_from_pretrained(MODEL_NAME)
    model = model.to(config.DEVICE).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    print(f"[SigLIP2] Model ready trên {config.DEVICE}")
    return model, preprocess, tokenizer


def _collect_image_paths():
    """Thu thập tất cả ảnh keyframe theo thứ tự nhất quán với btc_keyframe_meta."""
    if not os.path.exists(config.BTC_KF_META_CACHE):
        print("❌ Chưa có cache/btc_keyframe_meta.json. Chạy parse_btc_data.py trước!")
        return [], []

    with open(config.BTC_KF_META_CACHE, "r", encoding="utf-8") as f:
        kf_map = json.load(f)

    paths = []
    metas = []
    for video_id, frames in sorted(kf_map.items()):
        base_dir = os.path.join(config.KEYFRAME_DIR, video_id)
        for i, frame in enumerate(frames):
            # BTC naming: 001.jpg, 002.jpg...
            img_path = os.path.join(base_dir, f"{i+1:03d}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(base_dir, f"{i+1:04d}.jpg")
            if not os.path.exists(img_path):
                continue  # bỏ qua nếu chưa có ảnh
            paths.append(img_path)
            metas.append({
                "video_id":     video_id,
                "frame_idx":    frame["frame_idx"],
                "timestamp_sec": frame["timestamp_sec"],
                "path":         img_path,
            })
    return paths, metas


def build_siglip_index():
    # ── Load checkpoint nếu có ──────────────────────────────────────
    all_vecs = []
    all_metas = []
    done_paths = set()

    if os.path.exists(SIGLIP_PARTIAL_PATH) and os.path.exists(SIGLIP_ID_MAP_PATH):
        print("[SigLIP2] Phát hiện checkpoint — tiếp tục từ chỗ dở...")
        data = np.load(SIGLIP_PARTIAL_PATH)
        all_vecs = list(data["vecs"])
        with open(SIGLIP_ID_MAP_PATH, "r", encoding="utf-8") as f:
            all_metas = json.load(f)
        done_paths = {m["path"] for m in all_metas}
        print(f"[SigLIP2] Đã có {len(all_metas)} ảnh từ checkpoint.")

    # ── Thu thập ảnh cần encode ────────────────────────────────────
    paths, metas = _collect_image_paths()
    if not paths:
        print("❌ Không tìm thấy ảnh keyframe. Kiểm tra data/keyframes/ và parse_btc_data.py.")
        return

    pending_paths  = [p for p, m in zip(paths, metas) if p not in done_paths]
    pending_metas  = [m for p, m in zip(paths, metas) if p not in done_paths]

    print(f"[SigLIP2] Tổng: {len(paths):,} ảnh | Đã encode: {len(done_paths):,} | Còn lại: {len(pending_paths):,}")
    if not pending_paths:
        print("[SigLIP2] ✅ Đã encode toàn bộ ảnh! Chỉ cần finalize index...")
    else:
        # ── Load model ──────────────────────────────────────────────
        model, preprocess, _ = _load_siglip_model()

        # ── Encode từng batch ───────────────────────────────────────
        from PIL import Image as PILImage

        for batch_start in tqdm(range(0, len(pending_paths), BATCH_SIZE), desc="[SigLIP2] Encode"):
            batch_paths = pending_paths[batch_start: batch_start + BATCH_SIZE]
            batch_metas = pending_metas[batch_start: batch_start + BATCH_SIZE]

            imgs = []
            valid_metas = []
            for p, m in zip(batch_paths, batch_metas):
                try:
                    img = preprocess(PILImage.open(p).convert("RGB"))
                    imgs.append(img)
                    valid_metas.append(m)
                except Exception as e:
                    tqdm.write(f"[SigLIP2] Bỏ qua {p}: {e}")

            if not imgs:
                continue

            with torch.no_grad():
                batch_tensor = torch.stack(imgs).to(config.DEVICE)
                feats = model.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                vecs = feats.cpu().numpy().astype("float32")

            all_vecs.extend(list(vecs))
            all_metas.extend(valid_metas)

            # Checkpoint
            processed = len(done_paths) + batch_start + len(batch_paths)
            if processed % SAVE_EVERY < BATCH_SIZE:
                np.savez_compressed(SIGLIP_PARTIAL_PATH, vecs=np.array(all_vecs))
                with open(SIGLIP_ID_MAP_PATH, "w", encoding="utf-8") as f:
                    json.dump(all_metas, f, ensure_ascii=False)
                tqdm.write(f"[SigLIP2] Checkpoint: {len(all_metas):,} ảnh")

    # ── Build FAISS Index ──────────────────────────────────────────
    if not all_vecs:
        print("❌ Không có vector nào để build index.")
        return

    print(f"\n[SigLIP2] Build FAISS index từ {len(all_vecs):,} vectors (dim={len(all_vecs[0])})...")
    mat = np.array(all_vecs, dtype="float32")
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    faiss.write_index(index, SIGLIP_INDEX_PATH)
    with open(SIGLIP_ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metas, f, ensure_ascii=False)

    # Xóa checkpoint tạm khi hoàn tất
    if os.path.exists(SIGLIP_PARTIAL_PATH):
        os.remove(SIGLIP_PARTIAL_PATH)

    print(f"\n✅ SigLIP2 Index hoàn tất!")
    print(f"   → {SIGLIP_INDEX_PATH}  ({mat.shape[0]:,} vectors, dim={mat.shape[1]})")
    print(f"   → {SIGLIP_ID_MAP_PATH}")
    print("\nSigLIP2 sẽ tự động được dùng trong pipeline khi chạy streamlit run app.py")


if __name__ == "__main__":
    build_siglip_index()
