"""
eval_retrieval.py
===================
[P7] Script đo tự động Recall@K cho toàn bộ pipeline.

Cách dùng:
    python eval_retrieval.py

Định dạng Ground Truth (eval_queries.json):
[
    {
        "query": "Người đàn ông mặc áo trắng đứng trước biển quảng cáo",
        "gt_video_id": "L21_V001",
        "gt_frame_idx": 1234
    },
    ...
]

Script sẽ in ra:
    R@1, R@5, R@20, R@50, R@100
    Final Score = (R@1 + R@5 + R@20 + R@50 + R@100) / 5
"""

import os
import json
import time
import argparse
import search_utils
import qa_module

GT_FILE = os.path.join(os.path.dirname(__file__), "eval_queries.json")

SAMPLE_QUERIES = [
    # Thêm ground truth thật từ AIC 2025 batch 1 vào đây
    # Format: {"query": "...", "gt_video_id": "L22_V011", "gt_frame_idx": 5007}
]


def recall_at_k(ranked_results, gt_video_id, gt_frame_idx, k):
    """Kiểm tra xem ground truth có nằm trong top-k hay không."""
    for item in ranked_results[:k]:
        if item["video_id"] == gt_video_id and item["frame_idx"] == gt_frame_idx:
            return 1
    return 0


def evaluate(queries: list, use_expansion=False, verbose=True):
    Ks = [1, 5, 20, 50, 100]
    hits = {k: 0 for k in Ks}
    total = len(queries)

    if total == 0:
        print("❌ Không có query nào. Vui lòng thêm ground truth vào eval_queries.json")
        return

    print(f"\n{'='*60}")
    print(f"Đánh giá {total} queries | Query Expansion: {use_expansion}")
    print(f"{'='*60}")

    for i, q_item in enumerate(queries):
        query = q_item["query"]
        gt_vid = q_item["gt_video_id"]
        gt_frame = q_item["gt_frame_idx"]

        q_type = qa_module.detect_query_type(query)
        search_q = qa_module.remove_qa_keywords(query) if q_type == "qa" else query

        t0 = time.time()
        fused, reranked, _ = search_utils.full_search(
            search_q, do_verify=False, use_expansion=use_expansion
        )
        elapsed = time.time() - t0

        # Dùng reranked nếu có
        results = reranked if reranked else fused

        row_hits = {k: recall_at_k(results, gt_vid, gt_frame, k) for k in Ks}
        for k in Ks:
            hits[k] += row_hits[k]

        if verbose:
            r1 = "✅" if row_hits[1] else "❌"
            r5 = "✅" if row_hits[5] else ("❌" if not row_hits[5] else "")
            found_at = next(
                (rank+1 for rank, r in enumerate(results)
                 if r["video_id"] == gt_vid and r["frame_idx"] == gt_frame),
                None
            )
            rank_str = f"Rank {found_at}" if found_at else "Not found"
            print(f"[{i+1:02d}] R@1:{r1} | {rank_str} | {elapsed:.1f}s | {query[:50]}...")
            if not found_at:
                print(f"    Top 3 found: [{results[0]['video_id']}:{results[0]['frame_idx']}], [{results[1]['video_id']}:{results[1]['frame_idx']}], [{results[2]['video_id']}:{results[2]['frame_idx']}]")
                print(f"    Ground truth: {gt_vid}:{gt_frame}")

    print(f"\n{'='*60}")
    print("KẾT QUẢ RECALL@K:")
    recalls = {}
    for k in Ks:
        r = hits[k] / total
        recalls[k] = r
        print(f"  R@{k:3d} = {r:.4f}  ({hits[k]}/{total})")

    final_score = sum(recalls.values()) / len(Ks)
    print(f"\n  🏆 Final Score = {final_score:.4f}")
    print(f"{'='*60}\n")
    return recalls, final_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expansion", action="store_true", help="Bật Query Expansion")
    parser.add_argument("--gt", default=GT_FILE, help="Đường dẫn file ground truth JSON")
    args = parser.parse_args()

    if os.path.exists(args.gt):
        with open(args.gt, "r", encoding="utf-8") as f:
            queries = json.load(f)
        print(f"✅ Đã load {len(queries)} queries từ {args.gt}")
    elif SAMPLE_QUERIES:
        queries = SAMPLE_QUERIES
        print(f"⚠️  Không tìm thấy {args.gt}, dùng SAMPLE_QUERIES trong code.")
    else:
        print(f"❌ Vui lòng tạo file {GT_FILE} với danh sách queries.")
        print("""
Ví dụ nội dung eval_queries.json:
[
    {
        "query": "Xe tải lật trên đường cao tốc",
        "gt_video_id": "L22_V011",
        "gt_frame_idx": 5007
    }
]
""")
        raise SystemExit(1)

    evaluate(queries, use_expansion=args.expansion)
