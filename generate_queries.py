import json
import random

try:
    with open("index/btc_text_id_map.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    keys = list(range(len(data)))
    random.shuffle(keys)
    count = 0
    queries = []
    
    for k in keys:
        item = data[k]
        text = item.get("text", "")
        # Look for frames with meaningful text
        if len(text) > 80:
            queries.append({
                "video_id": item["video_id"],
                "frame_idx": item["frame_idx"],
                "raw_text": text
            })
            count += 1
            if count == 5:
                break
                
    for q in queries:
        print(f"Video: {q['video_id']}, Frame: {q['frame_idx']}")
        print(f"Metadata: {q['raw_text']}")
        print("-" * 50)
except Exception as e:
    print(e)
