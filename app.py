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
import qa_module
import trake_module

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


def to_submission_record(item: dict, query_type: str) -> dict:
    """Chuyển 1 kết quả sang format nộp bài AIC."""
    if query_type == "trake":
        if "edited_frame_ids" in item:
            frame_ids = item["edited_frame_ids"]
        elif "trake_events" in item:
            frame_ids = [ev["frame_idx"] for ev in item["trake_events"]]
        else:
            frame_ids = [item["frame_idx"]]
        return {
            "video_id": item["video_id"],
            "frame_ids": frame_ids
        }

    record = {
        "video_id":      item["video_id"],
        "frame_idx":     item["frame_idx"],
        "timestamp_sec": round(item.get("timestamp_sec", 0), 2),
        "keyframe_path": item.get("path", ""),
    }
    # Thêm câu trả lời nếu có (cho dạng QA)
    if "answer" in item:
        record["answer"] = item["answer"]
    return record


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
if "query_type" not in st.session_state:
    st.session_state.query_type = "kis"


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
    use_expansion = st.checkbox(
        "🔍 Query Expansion (Gemini)",
        value=False,
        help="Gemini tự sinh 2 cách diễn đạt khác nhau, tăng Recall cho BM25. Chậm hơn ~1-2s lần đầu."
    )

    st.divider()

    # ---- Panel nộp bài ----
    n_selected = len(st.session_state.selected)
    st.subheader(f"📤 Nộp bài ({n_selected} đã chọn)")

    if n_selected == 0:
        st.caption("Tick ✅ vào keyframe đúng bên dưới để thêm vào danh sách nộp.")
    else:
        records = [to_submission_record(v, st.session_state.query_type) for v in st.session_state.selected.values()]

        # Hiển thị preview danh sách đã chọn
        with st.expander("Xem danh sách đã chọn", expanded=False):
            for r in records:
                if st.session_state.query_type == "trake":
                    st.markdown(f"- `{r['video_id']}` · frames `{r['frame_ids']}`")
                else:
                    st.markdown(f"- `{r['video_id']}` · frame `{r['frame_idx']}` · `{r.get('timestamp_sec', '')}s`")

        # [P10] Quick Select buttons
        col_s1, col_s5, col_clr = st.columns(3)
        with col_s1:
            if st.button("⚡ Top 1", use_container_width=True, help="Chọn nhanh kết quả #1"):
                if st.session_state.last_results:
                    item = st.session_state.last_results[0]
                    k = (item["video_id"], item["frame_idx"])
                    st.session_state.selected[k] = item.copy()
                    st.rerun()
        with col_s5:
            if st.button("⚡ Top 5", use_container_width=True, help="Chọn nhanh kết quả #1-5"):
                for item in st.session_state.last_results[:5]:
                    k = (item["video_id"], item["frame_idx"])
                    st.session_state.selected[k] = item.copy()
                st.rerun()
        with col_clr:
            if st.button("🗑️ Xóa hết", use_container_width=True):
                st.session_state.selected = {}
                st.rerun()
        
        st.download_button(
            label="⬇️ Tải JSON (AIC format)",
            data=json.dumps(records, ensure_ascii=False, indent=2),
            file_name="aic_submission.json",
            mime="application/json",
            use_container_width=True,
        )
        # CSV Header
        if st.session_state.query_type == "trake":
            csv_header = "video_id,frame_ids"
        else:
            csv_header = "video_id,frame_idx,timestamp_sec,keyframe_path"
            if st.session_state.query_type == "qa":
                csv_header += ",answer"
            
        csv_lines = [csv_header]
        # Quy chế BTC: Nộp tối đa 100 kết quả
        for r in records[:100]:
            if st.session_state.query_type == "trake":
                frame_ids_str = str(r['frame_ids']).replace('"', '""')
                csv_lines.append(f"{r['video_id']},\"{frame_ids_str}\"")
            else:
                line = f"{r['video_id']},{r['frame_idx']},{r.get('timestamp_sec', '')},{r.get('keyframe_path', '')}"
                if st.session_state.query_type == "qa":
                    ans = str(r.get('answer', '')).replace('"', '""')
                    line += f',"{ans}"'
                csv_lines.append(line)

        st.download_button(
            label="⬇️ Tải CSV",
            data="\n".join(csv_lines),
            file_name="aic_submission.csv",
            mime="text/csv",
            use_container_width=True,
        )
        
        # [P10] Xóa từng item riêng lẻ
        st.markdown("**Xóa từng ảnh khỏi danh sách:**")
        for r in records:
            vid = r['video_id']
            fidx = r.get('frame_idx', '')
            label = f"`{vid}` · #{fidx}"
            skey = next((k for k in st.session_state.selected if k[0]==vid and k[1]==fidx), None)
            if skey and st.button(f"❌ {label}", key=f"del_{vid}_{fidx}", use_container_width=True):
                del st.session_state.selected[skey]
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
            q_type = qa_module.detect_query_type(query)
            st.session_state.query_type = q_type
            st.session_state.last_query = query
            
            if q_type == "trake":
                st.info("🔄 Đang xử lý truy vấn TRAKE (Tìm chuỗi sự kiện)...")
                sequences = trake_module.search_trake_sequence(query, top_k_per_event=300, time_window_sec=60)
                
                flat_results = []
                for seq in sequences:
                    rep_item = seq["events"][0].copy()
                    rep_item["trake_info"] = f"Chuỗi {len(seq['events'])} sự kiện"
                    rep_item["trake_events"] = seq["events"]
                    rep_item["fused_score"] = seq["total_score"]
                    flat_results.append(rep_item)
                    
                st.session_state.last_results = flat_results[:top_k_display]
                
            else:
                search_q = qa_module.remove_qa_keywords(query) if q_type == "qa" else query
                fused, reranked, verified = search_utils.full_search(
                    search_q, fusion_method=fusion_method, do_verify=do_verify,
                    use_expansion=use_expansion
                )
                st.session_state.last_results = (
                    (verified if verified is not None else reranked)[:top_k_display]
                )
        except Exception as e:
            import traceback
            st.error(f"❌ Lỗi: {str(e)}")
            with st.expander("🔍 Full Traceback (debug)"):
                st.code(traceback.format_exc())
            st.stop()

elif search_clicked:
    st.warning("Vui lòng nhập truy vấn trước khi tìm kiếm.")


# =================================================================
# HIỂN THỊ KẾT QUẢ
# =================================================================
results = st.session_state.last_results
if results:
    q_display = st.session_state.last_query
    q_type = st.session_state.query_type
    type_badge = {"kis": "Nomal (KIS)", "qa": "Hỏi Đáp (Q&A)", "trake": "Sự kiện (TRAKE)"}[q_type]
    
    st.subheader(f'Kết quả cho: "{q_display}" — {len(results)} keyframe [{type_badge}]')
    st.divider()

    for row_start in range(0, len(results), cols_per_row):
        row_items = results[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, item in zip(cols, row_items):
            sel_key = (item["video_id"], item["frame_idx"])
            is_selected = sel_key in st.session_state.selected

            with col:
                # ---- Ảnh keyframe ----
                img_path = item.get("path", "")
                has_image = os.path.exists(img_path) if img_path else False
                
                try:
                    if has_image:
                        img = Image.open(img_path).convert("RGB")
                        st.image(img, use_container_width=True)
                    else:
                        # Với data BTC chưa tải keyframes zip, sẽ không có ảnh cục bộ
                        st.info("🖼️ Không có file ảnh cục bộ (chưa tải Keyframes)")
                except Exception:
                    st.warning("⚠️ Không tải được ảnh")

                # ---- Metadata chính ----
                st.markdown(
                    f"**`{item['video_id']}`**  \n"
                    f"⏱ `{item['timestamp_sec']:.1f}s` · Frame `{item['frame_idx']}`"
                )

                # ---- Điểm số ----
                badges = []
                if "trake_info" in item:
                    badges.append(f"🔄 **{item['trake_info']}**")
                if item.get("verify_score") is not None:
                    badges.append(f"🎯 Gemini `{item['verify_score']}/10`")
                if item.get("rerank_score"):
                    badges.append(f"📊 Rerank `{item.get('rerank_score', 0):.3f}`")
                if item.get("fused_score") is not None:
                    badges.append(f"🔀 Score `{item['fused_score']:.4f}`")
                st.caption("  ·  ".join(badges))
                
                if "trake_events" in item and len(item["trake_events"]) > 1:
                    with st.expander("👁️ Xem và chỉnh sửa chuỗi sự kiện (TRAKE)"):
                        default_frames = ", ".join(str(ev["frame_idx"]) for ev in item["trake_events"])
                        edit_key = f"trake_edit_{item['video_id']}_{item['frame_idx']}"
                        
                        edited_str = st.text_input(
                            "Danh sách Frame IDs (cách nhau bởi dấu phẩy):", 
                            value=default_frames, 
                            key=edit_key
                        )
                        item["edited_frame_ids"] = [int(x.strip()) for x in edited_str.split(",") if x.strip().isdigit()]
                        
                        for ev_idx, ev in enumerate(item["trake_events"]):
                            st.caption(f"- Sự kiện {ev_idx+1}: Frame `{ev['frame_idx']}` lúc `{ev['timestamp_sec']}s`")

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

                # ---- Q&A Module (Nếu là query QA) ----
                if q_type == "qa":
                    ans_key = f"ans_{item['video_id']}_{item['frame_idx']}"
                    
                    st.markdown("**Câu trả lời cho Q&A:**")
                    if ans_key not in st.session_state:
                        st.session_state[ans_key] = item.get("answer", "")
                        
                    col_ans, col_btn_ans = st.columns([3, 1])
                    with col_ans:
                        ans_val = st.text_input(
                            "Đáp án", 
                            key=ans_key, 
                            label_visibility="collapsed"
                        )
                        # Lưu lại vào item để xuất file
                        if ans_val:
                            item["answer"] = ans_val
                            
                    def ai_callback(i_dict, i_path, q_disp, a_key):
                        a_ans = qa_module.generate_answer_for_frame(i_path, q_disp)
                        i_dict["answer"] = a_ans
                        st.session_state[a_key] = a_ans
                        
                    with col_btn_ans:
                        st.button(
                            "🤖 AI", 
                            key=f"btn_qa_{item['video_id']}_{item['frame_idx']}", 
                            help="Dùng Gemini sinh câu trả lời",
                            on_click=ai_callback,
                            args=(item, img_path, q_display, ans_key),
                            disabled=not has_image
                        )

                # ---- Checkbox chọn để nộp bài ----
                chk_key = f"chk_{item['video_id']}_{item['frame_idx']}"
                
                def toggle_selection(s_key, c_item, c_key):
                    # Đọc trạng thái mới nhất của checkbox từ session_state
                    if st.session_state.get(c_key, False):
                        st.session_state.selected[s_key] = c_item.copy()
                    elif s_key in st.session_state.selected:
                        del st.session_state.selected[s_key]

                st.checkbox(
                    "✅ Chọn để nộp",
                    value=is_selected,
                    key=chk_key,
                    on_change=toggle_selection,
                    args=(sel_key, item, chk_key)
                )

                st.markdown("---")
