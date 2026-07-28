"""
app.py
=======
Giao diện Streamlit cho hệ thống multimedia retrieval.
File này CHỈ lo phần hiển thị (UI), toàn bộ logic tìm kiếm nằm ở search_utils.py.

Cách chạy (SAU KHI đã chạy extract_keyframes.py và build_index.py):
    streamlit run app.py
"""

import streamlit as st
from PIL import Image

import config
import search_utils

st.set_page_config(page_title="AIC 2026 - Multimedia Retrieval", layout="wide")

st.title("🔍 AI Challenge HCMC 2026 — Multimedia Retrieval")
st.caption("Nhập truy vấn bằng tiếng Việt hoặc tiếng Anh để tìm keyframe/video khớp nhất.")

# ---------------- SIDEBAR: các tuỳ chọn tìm kiếm ----------------
with st.sidebar:
    st.header("⚙️ Cài đặt tìm kiếm")
    fusion_method = st.radio(
        "Phương pháp hybrid fusion",
        ["rrf", "weighted"],
        index=0,
        help="RRF (Reciprocal Rank Fusion): kết hợp theo THỨ HẠNG, ổn định hơn. "
             "Weighted: cộng điểm có trọng số sau khi chuẩn hóa min-max.",
    )
    do_verify = st.checkbox(
        "Xác thực lại bằng Gemini (VLM)",
        value=False,
        help="Gửi ảnh + query cho Gemini chấm điểm 0-10 để lọc kết quả sai. "
             "Chính xác hơn nhưng CHẬM hơn và tốn API call.",
    )
    top_k_display = st.slider("Số kết quả hiển thị", 1, 10, 10)

    st.divider()
    st.caption(
        "Pipeline: CLIP (ảnh) + BGE-M3 (caption/OCR/transcript) "
        "→ hybrid fusion → CrossEncoder rerank → (tuỳ chọn) Gemini verify"
    )

# ---------------- Ô NHẬP QUERY ----------------
query = st.text_input(
    "Nhập truy vấn:",
    placeholder="VD: người đàn ông mặc áo đỏ đang lái xe máy trên đường phố...",
)
search_clicked = st.button("🔎 Tìm kiếm", type="primary")

if search_clicked and query.strip():
    with st.spinner("Đang tìm kiếm..."):
        try:
            fused, reranked, verified = search_utils.full_search(
                query, fusion_method=fusion_method, do_verify=do_verify
            )
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    final_results = (verified if verified is not None else reranked)[:top_k_display]

    st.subheader(f'Kết quả cho: "{query}"')
    if not final_results:
        st.warning(
            "Không tìm thấy kết quả nào. Kiểm tra lại đã chạy build_index.py "
            "và thư mục data/videos/ có video chưa."
        )

    # Hiển thị dạng lưới ảnh, mỗi hàng 5 kết quả
    cols_per_row = 5
    for row_start in range(0, len(final_results), cols_per_row):
        row_items = final_results[row_start:row_start + cols_per_row]
        cols = st.columns(len(row_items))
        for col, item in zip(cols, row_items):
            with col:
                try:
                    st.image(Image.open(item["path"]), use_container_width=True)
                except Exception:
                    st.write("⚠️ Không tải được ảnh")

                st.markdown(f"**Video:** `{item['video_id']}`")
                st.markdown(f"**Thời điểm:** {item['timestamp_sec']}s")
                if item.get("verify_score") is not None:
                    st.markdown(f"**Điểm Gemini:** {item['verify_score']}/10")
                st.markdown(f"**Điểm rerank:** {item.get('rerank_score', 0):.3f}")
                if item.get("matched_text"):
                    with st.expander("Text khớp"):
                        st.write(f"({item.get('text_source', '')}) {item['matched_text']}")

elif search_clicked:
    st.warning("Vui lòng nhập truy vấn trước khi tìm kiếm.")

with st.expander("ℹ️ Hướng dẫn build index trước khi dùng"):
    st.markdown(
        """
        1. Copy video vào thư mục `data/videos/`
        2. Chạy `python extract_keyframes.py` → tách keyframe + transcript
        3. (Tuỳ chọn) đặt biến môi trường `GEMINI_API_KEY` để bật caption + verify
        4. Chạy `python build_index.py` → build FAISS index (ảnh + text)
        5. Chạy `streamlit run app.py` → tìm kiếm tại đây
        """
    )
