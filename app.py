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
    top_k_display = st.slider("Số kết quả", 1, 50, 10, help="Tăng lên 50 để nộp nhiều dòng CSV hơn (tối đa 100 theo quy chế BTC). Dùng fused results khi >10.")
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
                    st.markdown(f"- `{r['video_id']}` · frame `{r['frame_idx']}`")

        # [P10] Quick Select buttons
        col_s1, col_s5, col_all, col_clr = st.columns(4)
        with col_s1:
            if st.button("⚡ Top 1", use_container_width=True, help="Chọn nhanh kết quả #1"):
                if st.session_state.last_results:
                    item = st.session_state.last_results[0]
                    k = (item["video_id"], item["frame_idx"])
                    st.session_state.selected[k] = item.copy()
                    st.session_state[f"chk_{item['video_id']}_{item['frame_idx']}"] = True
                    st.rerun()
        with col_s5:
            if st.button("⚡ Top 5", use_container_width=True, help="Chọn nhanh kết quả #1-5"):
                for item in st.session_state.last_results[:5]:
                    k = (item["video_id"], item["frame_idx"])
                    st.session_state.selected[k] = item.copy()
                    st.session_state[f"chk_{item['video_id']}_{item['frame_idx']}"] = True
                st.rerun()
        with col_all:
            n_all = len(st.session_state.last_results)
            if st.button(f"✅ Tất cả ({n_all})", use_container_width=True, help="Chọn toàn bộ kết quả đang hiển thị"):
                for item in st.session_state.last_results:
                    k = (item["video_id"], item["frame_idx"])
                    st.session_state.selected[k] = item.copy()
                    st.session_state[f"chk_{item['video_id']}_{item['frame_idx']}"] = True
                st.rerun()
        with col_clr:
            if st.button("🗑️ Xóa hết", use_container_width=True):
                st.session_state.selected = {}
                for key in list(st.session_state.keys()):
                    if key.startswith("chk_"):
                        st.session_state[key] = False
                st.rerun()

        # ── Ô nhập tên file CSV (để đặt đúng tên query-p1-X-xxx.csv) ──
        q_type = st.session_state.query_type
        default_csv_name = f"query-p1-1-{q_type}.csv"
        csv_filename = st.text_input(
            "📝 Tên file CSV (đặt đúng theo BTC)",
            value=st.session_state.get("csv_filename", default_csv_name),
            help="VD: query-p1-1-kis.csv | query-p1-15-qa.csv | query-p1-4-trake.csv",
            key="csv_filename",
        )
        if not csv_filename.endswith(".csv"):
            csv_filename += ".csv"

        # ── Build CSV đúng format BTC (KHÔNG có header) ──
        csv_lines = []
        for r in records[:100]:          # tối đa 100 dòng theo quy chế
            if q_type == "trake":
                frame_ids = r.get("frame_ids", [])
                ids_str = ",".join(str(x) for x in frame_ids)
                csv_lines.append(f"{r['video_id']},{ids_str}")
            elif q_type == "qa":
                vid   = r["video_id"]
                fidx  = r["frame_idx"]
                ans   = str(r.get("answer", ""))
                # Escape dấu phẩy / ngoặc kép trong answer theo chuẩn CSV RFC 4180
                if "," in ans or '"' in ans or "\n" in ans:
                    ans = '"' + ans.replace('"', '""') + '"'
                csv_lines.append(f"{vid},{fidx},{ans}")
            else:  # kis
                csv_lines.append(f"{r['video_id']},{r['frame_idx']}")

        csv_data = "\n".join(csv_lines) + "\n"

        # Preview
        with st.expander("👁️ Preview CSV (3 dòng đầu)", expanded=False):
            st.code("\n".join(csv_lines[:3]), language="text")

        st.download_button(
            label=f"⬇️ Tải {csv_filename}",
            data=csv_data,
            file_name=csv_filename,
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
                if f"chk_{vid}_{fidx}" in st.session_state:
                    st.session_state[f"chk_{vid}_{fidx}"] = False
                st.rerun()

    st.divider()

    with st.expander("ℹ️ Hướng dẫn build index"):
        st.code(
            "python extract_keyframes.py\npython build_index.py\nstreamlit run app.py",
            language="bash",
        )



# =================================================================
# FRAME BROWSER (TRAKE helper) — đặt ở main area, ảnh to dễ nhìn
# =================================================================
with st.expander("🏞️ Duyệt frame theo video (TRAKE)", expanded=False):
    st.caption("Gõ Video ID → kéo slider → tìm đúng khoảnh khắc từng event E1/E2/E3/E4 rồi nậm frame_idx")
    fb_vid = st.text_input("Video ID", placeholder="VD: L24_V028", key="fb_vid").strip()
    if fb_vid:
        search_utils.load_indices()
        kf_meta = search_utils._keyframe_meta or {}
        fb_frames = kf_meta.get(fb_vid, [])
        if not fb_frames:
            kf_dir = os.path.join(config.KEYFRAME_DIR, fb_vid)
            if os.path.isdir(kf_dir):
                imgs = sorted([f for f in os.listdir(kf_dir) if f.endswith((".jpg", ".png"))])
                fb_frames = [{"frame_idx": int(f.split(".")[0]) if f.split(".")[0].isdigit() else i,
                              "timestamp_sec": i * 2.0,
                              "path": os.path.join(kf_dir, f)}
                             for i, f in enumerate(imgs)]
        if fb_frames:
            fb_total = len(fb_frames)
            fb_pos = st.slider(f"📂 {fb_vid} — {fb_total} keyframes", 0, fb_total - 1, 0, key="fb_pos")
            fb_fr = fb_frames[fb_pos]
            fb_img = fb_fr.get("path") or search_utils._get_btc_image_path(fb_vid, fb_fr["frame_idx"])

            # Hiển thị ảnh to + 2 frame kế bên (dedup để tránh trùng khi ở đầu/cuối)
            show_idx = list(dict.fromkeys([max(0, fb_pos - 1), fb_pos, min(fb_total - 1, fb_pos + 1)]))
            cols = st.columns(len(show_idx))
            for ci, si in enumerate(show_idx):
                fr_i = fb_frames[si]
                img_i = fr_i.get("path") or search_utils._get_btc_image_path(fb_vid, fr_i["frame_idx"])
                with cols[ci]:
                    label = "▶️ **Frame hiện tại**" if si == fb_pos else f"Frame {si + 1}"
                    st.caption(label)
                    if os.path.exists(img_i):
                        st.image(img_i, use_container_width=True)
                    st.markdown(f"`frame_idx = {fr_i['frame_idx']}` | `{fr_i.get('timestamp_sec', 0):.1f}s`")
                    # Key dùng frame_idx (unique trong video) thay vì position si
                    if st.button(f"➕ Chọn frame {fr_i['frame_idx']}", key=f"fb_add_{fb_vid}_{fr_i['frame_idx']}"):
                        sel = {"video_id": fb_vid, "frame_idx": fr_i["frame_idx"],
                               "timestamp_sec": fr_i.get("timestamp_sec", 0), "path": img_i}
                        st.session_state.selected[(fb_vid, fr_i["frame_idx"])] = sel
                        st.success(f"✅ Đã chọn frame {fr_i['frame_idx']}")


            st.info(f"📋 CSV: `{fb_vid},{fb_fr['frame_idx']}`")
        else:
            st.warning(f"Không tìm thấy keyframe cho `{fb_vid}`")


# =================================================================
# Chọn loại query thủ công
qtype_labels = {"kis": "🔍 KIS (Tìm video)", "qa": "❓ Q&A (Hỏi đáp)", "trake": "⏱️ TRAKE (Chuỗi sự kiện)"}
manual_qtype = st.radio(
    "Loại truy vấn:",
    options=["kis", "qa", "trake"],
    format_func=lambda x: qtype_labels[x],
    index=["kis", "qa", "trake"].index(st.session_state.query_type),
    horizontal=True,
    key="manual_qtype",
)
# Cập nhật session state nếu người dùng đổi loại
if manual_qtype != st.session_state.query_type:
    st.session_state.query_type = manual_qtype

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
            # Dùng lựa chọn thủ công thay vì auto-detect
            q_type = st.session_state.query_type
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
                # Nếu cần nhiều hơn 10 kết quả → dùng fused (RRF, top 50) thay vì reranked (top 10)
                need_more = top_k_display > 10
                fused, reranked, verified = search_utils.full_search(
                    search_q, fusion_method=fusion_method, do_verify=do_verify,
                    use_expansion=use_expansion,
                    top_k=max(top_k_display, 50) if need_more else None
                )
                if verified is not None:
                    results_pool = verified
                elif need_more:
                    results_pool = fused  # fused có nhiều kết quả hơn reranked
                else:
                    results_pool = reranked
                st.session_state.last_results = results_pool[:top_k_display]
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
                        
                        # Hiển thị chuỗi ảnh để dễ nhìn
                        ev_cols = st.columns(len(item["trake_events"]))
                        import os
                        for ev_idx, (col, ev) in enumerate(zip(ev_cols, item["trake_events"])):
                            with col:
                                st.caption(f"Sự kiện {ev_idx+1}: Frame `{ev['frame_idx']}`")
                                ev_img_path = search_utils._get_btc_image_path(ev["video_id"], ev["frame_idx"])
                                if os.path.exists(ev_img_path):
                                    st.image(ev_img_path, use_container_width=True)

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

                if chk_key not in st.session_state:
                    st.session_state[chk_key] = is_selected
                    
                st.checkbox(
                    "✅ Chọn để nộp",
                    key=chk_key,
                    on_change=toggle_selection,
                    args=(sel_key, item, chk_key)
                )

                st.markdown("---")
