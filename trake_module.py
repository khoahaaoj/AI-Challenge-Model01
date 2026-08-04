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

def parse_trake_query(query: str) -> list[str]:
    """
    Dùng Gemini để tách câu truy vấn TRAKE thành danh sách các sự kiện theo thứ tự.
    Trả về list các câu sub-query.
    """
    if not config.GEMINI_API_KEY:
        # Fallback chia tay bằng keyword đơn giản nếu không có API
        for kw in ["sau đó", "tiếp theo", "rồi", ", rồi"]:
            if kw in query:
                return [q.strip() for q in query.split(kw) if q.strip()]
        return [query]

    import google.generativeai as genai
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    
    prompt = (
        f"Bạn là chuyên gia phân tích video. Hãy tách câu truy vấn dưới đây thành một danh sách "
        f"các sự kiện diễn ra theo thứ tự thời gian.\n"
        f"Truy vấn: '{query}'\n\n"
        f"Chỉ trả về 1 mảng JSON hợp lệ chứa các chuỗi (không thêm text nào khác, không có markdown).\n"
        f"Ví dụ: [\"Người đàn ông bước vào phòng\", \"Bật đèn sáng\"]"
    )
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
            
        sub_queries = json.loads(text)
        if isinstance(sub_queries, list) and len(sub_queries) > 0:
            return sub_queries
    except Exception as e:
        print(f"[TRAKE] Lỗi parse LLM ({e}), dùng fallback cắt chuỗi.")
        
    # Fallback
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
        # Lấy tất cả frame của sự kiện 0 (nếu có, hoặc sự kiện 1 nếu sự kiện 0 bị miss)
        starts = [it for idx, it in items if idx == 0]
        if not starts:
            # Nếu sự kiện đầu bị miss, thử bắt đầu từ sự kiện 1
            starts = [it for idx, it in items if idx == 1]
            start_idx = 1
        else:
            start_idx = 0
            
        for start_item in starts:
            current_seq = [start_item]
            last_time = start_item["timestamp_sec"]
            is_valid = True
            missed_events = 1 if start_idx == 1 else 0
            
            for target_e_idx in range(start_idx + 1, len(sub_queries)):
                candidates = [it for idx, it in items if idx == target_e_idx]
                valid_nexts = [
                    c for c in candidates 
                    if 0 < (c["timestamp_sec"] - last_time) <= time_window_sec
                ]
                
                if not valid_nexts:
                    missed_events += 1
                    if missed_events > (len(sub_queries) - min_match):
                        is_valid = False
                        break
                    continue
                    
                best_next = max(valid_nexts, key=lambda x: x.get("fused_score", x.get("score", 0)))
                current_seq.append(best_next)
                last_time = best_next["timestamp_sec"]
                
            if is_valid and len(current_seq) >= min_match:
                total_score = sum(ev.get("fused_score", ev.get("score", 0)) for ev in current_seq)
                valid_sequences.append({
                    "video_id": vid,
                    "total_score": total_score,
                    "events": current_seq
                })
                
    # 5. Sort theo điểm tổng
    valid_sequences.sort(key=lambda x: x["total_score"], reverse=True)
    return valid_sequences
