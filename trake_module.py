"""
trake_module.py
=======================
Module xử lý truy vấn chuỗi sự kiện (TRAKE - Temporal & Relational Actions Key Events)
Giải quyết các câu truy vấn có tính chất thứ tự thời gian.
Ví dụ: "Người đàn ông bước vào phòng, sau đó bật đèn"
"""

import json
import config
import search_utils

import re as _re


def _extract_video_context(query: str) -> str:
    """Lấy phần mô tả context video (text trước E1:)."""
    match = _re.search(r'\bE\s*1\s*:', query, _re.IGNORECASE)
    if match:
        ctx = query[:match.start()].strip().rstrip(',.:;-\n').strip()
        return ctx[:150] if ctx else ""
    return ""


def _parse_by_regex(query: str) -> list[str] | None:
    """
    Ưu tiên 1: Tách events theo nhãn E1: E2: E3: ... bằng regex.
    Không cần gọi API. Chính xác với format chuẩn BTC.
    """
    pattern = _re.compile(r'E\s*(\d+)\s*:\s*(.*?)(?=E\s*\d+\s*:|$)', _re.DOTALL | _re.IGNORECASE)
    matches = pattern.findall(query)
    if not matches or len(matches) < 2:
        return None
    matches_sorted = sorted(matches, key=lambda x: int(x[0]))
    events = [m[1].strip() for m in matches_sorted if m[1].strip()]
    if len(events) < 2:
        return None
    # Thêm context video vào từng event để tăng recall
    ctx = _extract_video_context(query)
    if ctx:
        events = [f"{ctx}. {e}" for e in events]
    return events


def parse_trake_query(query: str) -> list[str]:
    """
    Tách câu truy vấn TRAKE thành danh sách các sự kiện theo thứ tự.
    Ưu tiên:
    1. Regex (E1:/E2:/E3: format chuẩn BTC) — nhanh, không cần API
    2. Gemini (query dạng tự do)
    3. Fallback cắt keyword thời gian
    """
    # --- 1. Regex (BTC format) ---
    regex_result = _parse_by_regex(query)
    if regex_result:
        print(f"[TRAKE][Regex] {len(regex_result)} events: {[e[:60] for e in regex_result]}")
        return regex_result

    # --- 2. Gemini (query tự do) ---
    if config.GEMINI_API_KEY:
        import json as _json
        from google import genai as _genai
        client = _genai.Client(api_key=config.GEMINI_API_KEY)
        ctx = _extract_video_context(query)
        prompt = (
            f"Bạn là chuyên gia phân tích video. Hãy tách câu truy vấn dưới đây thành một danh sách "
            f"các sự kiện diễn ra theo thứ tự thời gian.\n"
            f"Truy vấn: '{query}'\n\n"
            f"Chỉ trả về 1 mảng JSON hợp lệ chứa các chuỗi (không thêm text nào khác, không có markdown).\n"
            f"Mỗi event nên bao gồm context video để dễ tìm kiếm.\n"
            f"Ví dụ: [\"Người đàn ông bước vào phòng\", \"Bật đèn sáng\"]"
        )
        try:
            response = client.models.generate_content(model=config.GEMINI_MODEL, contents=prompt)
            text = (response.text or "").strip()
            if text.startswith("```"):
                text = text.split("```")[1].lstrip("json").strip()
            sub_queries = _json.loads(text)
            if isinstance(sub_queries, list) and len(sub_queries) > 0:
                if ctx:
                    sub_queries = [
                        f"{ctx}. {e}" if not e.startswith(ctx[:30]) else e
                        for e in sub_queries
                    ]
                print(f"[TRAKE][Gemini] {len(sub_queries)} events")
                return sub_queries
        except Exception as e:
            print(f"[TRAKE] Gemini parse lỗi ({e}), dùng fallback.")

    # --- 3. Fallback keyword ---
    for kw in ["sau đó", "tiếp theo", "rồi", ", rồi", "rồi đến"]:
        if kw in query:
            return [q.strip() for q in query.split(kw) if q.strip()]

    return [query]



def search_trake_sequence(query: str, top_k_per_event=500, time_window_sec=60):
    """
    Tìm kiếm chuỗi sự kiện trong cùng 1 video.
    Trả về danh sách các sequence tìm được.
    Mỗi sequence chứa thông tin video_id, tổng điểm, và danh sách các frame.
    """
    sub_queries = parse_trake_query(query)
    print(f"[TRAKE] Tách truy vấn thành {len(sub_queries)} sự kiện: {sub_queries}")
    
    if len(sub_queries) <= 1:
        # Nếu chỉ có 1 sự kiện, fallback về search bình thường
        fused, _, _ = search_utils.full_search(query, top_k=top_k_per_event, skip_rerank=True)
        # Chuyển về định dạng sequence để app.py dễ xử lý chung
        return [{
            "video_id": item["video_id"],
            "total_score": item.get("fused_score", item.get("score", 0)),
            "events": [item]
        } for item in fused]
        
    # 1. Lấy kết quả độc lập cho từng sự kiện
    events_results = []
    for sq in sub_queries:
        fused, _, _ = search_utils.full_search(sq, fusion_method="rrf", do_verify=False, top_k=top_k_per_event, skip_rerank=True)
        events_results.append(fused)
        
    # 2. Tìm video_id xuất hiện trong ít nhất N-1 sự kiện (soft-constraint)
    from collections import Counter
    vid_counts = Counter()
    for results in events_results:
        vid_counts.update(set(item["video_id"] for item in results))
    
    # Chỉ giữ các video thỏa ít nhất N-1 sự kiện
    min_match = max(1, len(sub_queries) - 1)
    common_videos = {vid for vid, count in vid_counts.items() if count >= min_match}
    
    if not common_videos:
        print("[TRAKE] Không có video nào chứa đủ sự kiện.")
        return []

    # 3. Group by video_id
    grouped_events = {vid: [] for vid in common_videos}
    for e_idx, results in enumerate(events_results):
        for item in results:
            vid = item["video_id"]
            if vid in common_videos:
                grouped_events[vid].append((e_idx, item))
                
    # 4. Tìm chuỗi thỏa mãn thứ tự thời gian
    valid_sequences = []
    for vid, items in grouped_events.items():
        # Group candidates theo event index
        event_candidates = {}  # {e_idx: [item, ...]}
        for e_idx, it in items:
            event_candidates.setdefault(e_idx, []).append(it)

        # Build sequence: lặp qua từng event 0..N-1, luôn đảm bảo đủ N frames
        current_seq = []
        last_time = -1.0

        for e_idx in range(len(sub_queries)):
            candidates = event_candidates.get(e_idx, [])

            # Ưu tiên frame hợp lệ trong time window (sau last_time, không quá 2 phút)
            valid_nexts = [
                c for c in candidates
                if 0 < (c["timestamp_sec"] - last_time) <= time_window_sec
            ]

            if valid_nexts:
                # Lấy frame tốt nhất trong window
                best = max(valid_nexts, key=lambda x: x.get("fused_score", x.get("score", 0)))
                last_time = best["timestamp_sec"]
            elif candidates:
                # Fallback: ưu tiên frame SAU last_time (dù vượt window)
                after = [c for c in candidates if c["timestamp_sec"] > last_time]
                if after:
                    best = max(after, key=lambda x: x.get("fused_score", x.get("score", 0)))
                    last_time = best["timestamp_sec"]
                else:
                    # Tất cả candidate đều TRƯỚC last_time → dùng frame tốt nhất
                    # nhưng ĐẶT timestamp giả = last_time+1 để last_time luôn tiến lên
                    best = dict(max(candidates, key=lambda x: x.get("fused_score", x.get("score", 0))))
                    last_time = last_time + 1.0
                    best["timestamp_sec"] = last_time  # ghi nhớ vị trí giả
            elif current_seq:
                # Không có candidate nào → lặp frame trước, timestamp +1s
                best = dict(current_seq[-1])
                last_time = last_time + 1.0
                best["timestamp_sec"] = last_time
            else:
                break

            current_seq.append(best)

        # Chỉ nhận sequence có đủ N frames
        if len(current_seq) == len(sub_queries):
            # Safety: đảm bảo frame_idx tăng dần (BTC yêu cầu thứ tự thời gian)
            is_ordered = all(
                current_seq[i]["frame_idx"] <= current_seq[i+1]["frame_idx"]
                for i in range(len(current_seq) - 1)
            )
            if not is_ordered:
                print(f"[TRAKE] ⚠️ {vid}: sequence không theo thứ tự frame_idx, bỏ qua")
                continue
            total_score = sum(ev.get("fused_score", ev.get("score", 0)) for ev in current_seq)
            valid_sequences.append({
                "video_id": vid,
                "total_score": total_score,
                "events": current_seq,
            })
        else:
            print(f"[TRAKE] Bỏ qua {vid}: chỉ tìm được {len(current_seq)}/{len(sub_queries)} events")

    # 5. Sort theo điểm tổng
    valid_sequences.sort(key=lambda x: x["total_score"], reverse=True)
    return valid_sequences
