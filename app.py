"""
app.py
=======
Giao diện Streamlit cho AIC 2026 Multimedia Retrieval.

Tính năng:
  - Tìm kiếm 3 nhánh: SigLIP2 (ảnh) + BGE-M3 (text dense) + BM25 (keyword)
  - Hybrid RRF Fusion → CrossEncoder rerank → (tuỳ chọn) Gemini VLM verify
  - Xem video tại timestamp ±2s ngay trong giao diện
  - Chọn keyframe và export JSON/CSV theo format nộp bài AIC
"""
import os
import json
import streamlit as st
from PIL import Image

import config
import search_utils

st.set_page_config(
    page_title="AIC 2026 — Video Retrieval",
    page_icon="🔍",
    layout="wide",
)


# =================================================================
# TIỆN ÍCH
# =================================================================
def find_video_path(video_id: str) -> str | None:
    """Tìm file video gốc từ video_id (thử lần lượt các extension phổ biến)."""
    for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
        p = os.path.join(config.VIDEO_DIR, video_id + ext)
        if os.path.exists(p):
            return p
    return None


def to_submission_record(item: dict) -> dict:
    """Chuyển 1 kết quả sang format nộp bài AIC: video_id + frame_idx + timestamp."""
    return {
        "video_id":      item["video_id"],
        "frame_idx":     item["frame_idx"],
        "timestamp_sec": round(item["timestamp_sec"], 2),
        "keyframe_path": item.get("path", ""),
    }


# =================================================================
# SESSION STATE INIT
# =================================================================
if "selected" not in st.session_state:
    # dict[(video_id, frame_idx)] -> item — giữ danh sách keyframe đã chọn để nộp
    st.session_state.selected = {}
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# =================================================================
# HEADER
# =================================================================
st.title("🔍 AIC 2026 — Multimedia Video Retrieval")
st.caption(
    "SigLIP2 (ảnh) · BGE-M3 dense (text) · BM25 sparse (keyword) "
    "→ RRF Fusion → CrossEncoder Rerank → Gemini Verify"
)


# =================================================================
# SIDEBAR — Cài đặt + Submit panel
# =================================================================
with st.sidebar:
    st.header("⚙️ Cài đặt tìm kiếm")
    fusion_method = st.radio(
        "Hybrid fusion",
        ["rrf", "weighted"],
        index=0,
        help="RRF: kết hợp theo thứ hạng (khuyến nghị). Weighted: cộng điểm có trọng số.",
    )
    do_verify = st.checkbox(
        "Gemini VLM verify",
        value=False,
        help="Gửi ảnh thật cho Gemini chấm điểm 0-10. Chính xác hơn nhưng chậm hơn.",
    )
    cols_per_row = st.slider("Cột hiển thị", 2, 5, 3)
    top_k_display = st.slider("Số kết quả", 1, 10, 10)

    st.divider()

    # ---- Panel nộp bài ----
    n_selected = len(st.session_state.selected)
    st.subheader(f"📤 Nộp bài ({n_selected} đã chọn)")

    if n_selected == 0:
        st.caption("Tick ✅ vào keyframe đúng bên dưới để thêm vào danh sách nộp.")
    else:
        records = [to_submission_record(v) for v in st.session_state.selected.values()]

        # Hiển thị preview danh sách đã chọn
        with st.expander("Xem danh sách đã chọn", expanded=False):
            for r in records:
                st.markdown(f"- `{r['video_id']}` · frame `{r['frame_idx']}` · `{r['timestamp_sec']}s`")

        st.download_button(
            label="⬇️ Tải JSON (AIC format)",
            data=json.dumps(records, ensure_ascii=False, indent=2),
            file_name="aic_submission.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            label="⬇️ Tải CSV",
            data="\n".join(
                ["video_id,frame_idx,timestamp_sec,keyframe_path"]
                + [
                    f"{r['video_id']},{r['frame_idx']},{r['timestamp_sec']},{r['keyframe_path']}"
                    for r in records
                ]
            ),
            file_name="aic_submission.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("🗑️ Xóa tất cả đã chọn", use_container_width=True):
            st.session_state.selected = {}
            st.rerun()

    st.divider()
    with st.expander("ℹ️ Hướng dẫn build index"):
        st.code(
            "python extract_keyframes.py\npython build_index.py\nstreamlit run app.py",
            language="bash",
        )


# =================================================================
# SEARCH BAR
# =================================================================
col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input(
        "Truy vấn:",
        placeholder="VD: người đàn ông mặc áo đỏ đang lái xe máy trên đường phố...",
        label_visibility="collapsed",
    )
with col_btn:
    search_clicked = st.button("🔎 Tìm", type="primary", use_container_width=True)


# =================================================================
# SEARCH LOGIC
# =================================================================
if search_clicked and query.strip():
    with st.spinner("Đang tìm kiếm (SigLIP2 + BGE-M3 + BM25 → fusion → rerank)..."):
        try:
            fused, reranked, verified = search_utils.full_search(
                query, fusion_method=fusion_method, do_verify=do_verify
            )
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    st.session_state.last_results = (
        (verified if verified is not None else reranked)[:top_k_display]
    )
    st.session_state.last_query = query

elif search_clicked:
    st.warning("Vui lòng nhập truy vấn trước khi tìm kiếm.")


# =================================================================
# HIỂN THỊ KẾT QUẢ
# =================================================================
results = st.session_state.last_results
if results:
    q_display = st.session_state.last_query
    st.subheader(f'Kết quả cho: "{q_display}" — {len(results)} keyframe')
    st.divider()

    for row_start in range(0, len(results), cols_per_row):
        row_items = results[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, item in zip(cols, row_items):
            sel_key = (item["video_id"], item["frame_idx"])
            is_selected = sel_key in st.session_state.selected

            with col:
                # ---- Ảnh keyframe ----
                try:
                    img = Image.open(item["path"]).convert("RGB")
                    st.image(img, use_container_width=True)
                except Exception:
                    st.warning("⚠️ Không tải được ảnh")

                # ---- Metadata chính ----
                st.markdown(
                    f"**`{item['video_id']}`**  \n"
                    f"⏱ `{item['timestamp_sec']:.1f}s` · Frame `{item['frame_idx']}`"
                )

                # ---- Điểm số ----
                badges = []
                if item.get("verify_score") is not None:
                    badges.append(f"🎯 Gemini `{item['verify_score']}/10`")
                badges.append(f"📊 Rerank `{item.get('rerank_score', 0):.3f}`")
                if item.get("fused_score") is not None:
                    badges.append(f"🔀 RRF `{item['fused_score']:.4f}`")
                st.caption("  ·  ".join(badges))

                # ---- Text khớp (nếu có) ----
                if item.get("matched_text"):
                    src = item.get("text_source", "")
                    icon = {"caption": "💬", "ocr": "🔤", "transcript": "🎙️"}.get(src, "📝")
                    with st.expander(f"{icon} Text ({src})"):
                        st.write(item["matched_text"])

                # ---- Video player ----
                video_path = find_video_path(item["video_id"])
                if video_path:
                    with st.expander("▶️ Xem video tại timestamp"):
                        # Phát từ 2 giây trước keyframe để thấy context
                        start_sec = max(0, int(item["timestamp_sec"]) - 2)
                        st.video(video_path, start_time=start_sec)
                        st.caption(
                            f"Phát từ `{start_sec}s` "
                            f"(keyframe tại `{item['timestamp_sec']:.1f}s`)"
                        )
                else:
                    st.caption("_(Không tìm thấy file video gốc trong data/videos/)_")

                # ---- Checkbox chọn để nộp bài ----
                checked = st.checkbox(
                    "✅ Chọn để nộp",
                    value=is_selected,
                    key=f"chk_{item['video_id']}_{item['frame_idx']}",
                )
                if checked:
                    st.session_state.selected[sel_key] = item
                elif sel_key in st.session_state.selected:
                    del st.session_state.selected[sel_key]

                st.markdown("---")
