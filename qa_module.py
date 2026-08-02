"""
qa_module.py
======================
Module xử lý truy vấn dạng Question & Answering (Q&A) cho AIC 2026.
Bao gồm:
1. detect_query_type: Phân loại câu hỏi (KIS, QA, TRAKE)
2. generate_answer_for_frame: Dùng VLM (Gemini) để trả lời câu hỏi dựa trên nội dung bức ảnh.
"""

import os
from PIL import Image
import config

def detect_query_type(query: str) -> str:
    """
    Phân loại query thuộc dạng nào dựa trên keyword.
    - KIS: Text-to-Video thông thường.
    - QA: Có từ để hỏi.
    - TRAKE: Có các cụm từ chỉ thứ tự thời gian, sự kiện.
    """
    q = query.lower()
    
    # 1. Kiểm tra TRAKE (chuỗi sự kiện)
    trake_keywords = [
        "khoảnh khắc", "giai đoạn", "bước", "thứ tự", "lần lượt", 
        "trước khi", "sau khi", "chuỗi", "diễn biến", "tiếp theo", "liên tiếp"
    ]
    if any(kw in q for kw in trake_keywords):
        return "trake"
        
    # 2. Kiểm tra Q&A (có câu hỏi)
    qa_keywords = [
        "?", "bao nhiêu", "màu gì", "ai là", "là gì", "như thế nào", 
        "ở đâu", "khi nào", "tên gì", "làm gì", "gì", "ai", "mấy"
    ]
    if "?" in query or any(kw in q.split() for kw in qa_keywords):
        return "qa"
        
    # Mặc định là KIS (Known-Item Search)
    return "kis"

def remove_qa_keywords(query: str) -> str:
    """Loại bỏ các từ để hỏi khỏi query để hệ thống search (SigLIP, BM25) tìm chính xác subject hơn."""
    qa_keywords = [
        "có màu gì", "màu gì", "là ai", "ai là", "là gì", "như thế nào", 
        "ở đâu", "khi nào", "tên gì", "làm gì", "có bao nhiêu", "bao nhiêu", "cái gì"
    ]
    q = query
    q_lower = query.lower()
    for kw in qa_keywords:
        if kw in q_lower:
            # Tìm vị trí và cắt bỏ (giữ nguyên case các chữ khác)
            idx = q_lower.find(kw)
            q = q[:idx] + q[idx+len(kw):]
            q_lower = q.lower()
    return q.replace("?", "").strip()


def generate_answer_for_frame(image_path: str, question: str) -> str:
    """
    Dùng Gemini để trả lời câu hỏi dựa trên bức ảnh.
    Ưu tiên trả lời cực kỳ ngắn gọn (thường là 1-3 từ) đúng như yêu cầu của BTC.
    """
    if not config.GEMINI_API_KEY:
        return "Lỗi: Chưa cấu hình GEMINI_API_KEY trong config.py"

    if not os.path.exists(image_path):
        return f"Lỗi: Không tìm thấy ảnh tại {image_path}"

    import google.generativeai as genai
    genai.configure(api_key=config.GEMINI_API_KEY)
    
    # Dùng flash-lite cho tốc độ nhanh, hoặc pro nếu cần siêu chính xác
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    
    prompt = (
        f"Bạn là một trợ lý phân tích video. Hãy nhìn vào bức ảnh này và trả lời câu hỏi dưới đây bằng tiếng Việt.\n"
        f"Câu hỏi: {question}\n\n"
        f"Yêu cầu QUAN TRỌNG: Hãy trả lời RẤT NGẮN GỌN (chỉ 1 đến 5 từ, không giải thích, không viết thành câu hoàn chỉnh). "
        f"Ví dụ: Nếu hỏi màu gì thì chỉ trả lời 'màu đỏ' hoặc 'đỏ'."
    )
    
    try:
        image = Image.open(image_path).convert("RGB")
        response = model.generate_content([prompt, image])
        return (response.text or "").strip()
    except Exception as e:
        return f"Lỗi API: {str(e)}"

# Test nhanh
if __name__ == "__main__":
    q1 = "Tìm video có người đàn ông đang chạy"
    q2 = "Người đàn ông trong video mặc áo màu gì?"
    q3 = "Khoảnh khắc 1 người đàn ông chạy, khoảnh khắc 2 ngã"
    print(f"'{q1}' -> {detect_query_type(q1)}")
    print(f"'{q2}' -> {detect_query_type(q2)}")
    print(f"'{q3}' -> {detect_query_type(q3)}")
