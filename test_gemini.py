import time
import search_utils
candidates = [{"path": "/home/khoaha/PyCharmMiscProject/khoa/aic_retrieval/data/keyframes/L22_V011/001.jpg"} for _ in range(5)]
start = time.time()
print("Starting verification...")
res = search_utils.verify_with_gemini("a dog", candidates)
print(f"Done in {time.time()-start:.2f} seconds")
print(res)
