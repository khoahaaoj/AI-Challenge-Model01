import search_utils

query = "phở sẵn"
print(f"🔎 Testing text search with query: '{query}'")

try:
    results = search_utils.search_bm25(query, top_k=5)
    print("\nBM25 Results:")
    for r in results:
        print(f" - Video: {r['video_id']}, Score: {r.get('score', 0):.2f}, Text: {r.get('matched_text', '')}")
except Exception as e:
    print("Error BM25:", e)

try:
    results = search_utils.search_text(query, top_k=5)
    print("\nBGE-M3 Results:")
    for r in results:
        print(f" - Video: {r['video_id']}, Score: {r.get('score', 0):.2f}, Text: {r.get('matched_text', '')}")
except Exception as e:
    print("Error BGE-M3:", e)
