import sys
import search_utils
import qa_module

query = "Chiếc xe tải bị lật trên đường cao tốc có màu gì"
search_q = qa_module.remove_qa_keywords(query)
print(f"Original: {query}")
print(f"Cleaned: {search_q}")

fused, reranked, verified = search_utils.full_search(search_q, fusion_method="rrf", do_verify=False)

print("\n--- RESULTS ---")
for i, c in enumerate(reranked[:10]):
    print(f"Rank {i+1}: Video {c['video_id']}, Frame {c['frame_idx']}, Text: {c.get('matched_text', '')[:30]}, Rerank: {c.get('rerank_score', 0):.3f}, Final: {c.get('final_score', 0):.3f}")
