"""
tokenizer_utils.py
===================
Module dùng CHUNG cho cả build_index.py và search_utils.py để tokenize text
cho BM25. Đảm bảo build-time và query-time dùng CÙNG 1 hàm tokenize → tránh bug
BM25 tokenizer mismatch (chi tiết xem AIC2026_Deep_Audit_Report.md mục 5.1).

Vấn đề cũ:
  - build_index.py: dùng underthesea.word_tokenize nếu có → tạo token kiểu "thủ_tướng"
  - search_utils.py: luôn dùng query.lower().split() → token "thủ" + "tướng" riêng
  → BM25 không bao giờ khớp đúng từ ghép tiếng Việt khi bật underthesea

Giải pháp:
  - Tập trung logic tokenize vào đây (hàm get_tokenizer())
  - build_index.py gọi get_tokenizer() và lưu flag "used_underthesea" vào pickle
  - search_utils.py đọc flag từ pickle → gọi get_tokenizer(used_underthesea=flag)
    để tokenize query ĐÚNG theo cách đã build
"""


def get_tokenizer(use_underthesea: bool = None):
    """
    Trả về hàm tokenize(text: str) -> list[str] phù hợp.

    Args:
        use_underthesea: True/False để ép buộc dùng/không dùng underthesea.
                         None (mặc định) = auto-detect (thử import, dùng nếu có).

    Returns:
        tuple(tokenize_fn, used_underthesea_flag)
          - tokenize_fn: callable(str) -> list[str]
          - used_underthesea_flag: bool (True nếu đang dùng underthesea)
    """
    if use_underthesea is True:
        # Ép buộc dùng underthesea (ví dụ: đọc từ pickle biết đã build với underthesea)
        try:
            from underthesea import word_tokenize
            def _tok(text: str) -> list:
                return word_tokenize(text.lower(), format="text").split()
            return _tok, True
        except ImportError:
            # underthesea không còn được cài nữa → fallback an toàn
            # (có thể xảy ra nếu môi trường thay đổi sau khi build)
            print("[tokenizer_utils] ⚠️  BM25 index được build với underthesea nhưng hiện không "
                  "cài. Fallback về str.split() → recall BM25 có thể giảm với từ ghép tiếng Việt. "
                  "Khuyến nghị: pip install underthesea.")
            def _tok(text: str) -> list:
                return text.lower().split()
            return _tok, False

    elif use_underthesea is False:
        # Ép buộc dùng str.split() (index được build mà không có underthesea)
        def _tok(text: str) -> list:
            return text.lower().split()
        return _tok, False

    else:
        # Auto-detect: thử import underthesea
        try:
            from underthesea import word_tokenize
            def _tok(text: str) -> list:
                return word_tokenize(text.lower(), format="text").split()
            return _tok, True
        except ImportError:
            def _tok(text: str) -> list:
                return text.lower().split()
            return _tok, False


def detect_language(text: str) -> str:
    """
    Phát hiện ngôn ngữ của câu text. Trả về mã ngôn ngữ ISO 639-1 (ví dụ: 'vi', 'en').

    Dùng thư viện langid (local, không cần API, rất nhẹ ~15MB) thay vì heuristic
    isascii() sai với tiếng Việt không dấu (ví dụ: "nguoi dan ong ao do" là ASCII
    nhưng thực chất là tiếng Việt → cần dịch trước khi đưa vào SigLIP2).

    Lưu ý: langid đôi khi nhầm tiếng Việt không dấu sang pt/it (Latin gần giống).
    Vì vậy hàm này TRẢ VỀ 'en' CHỈ KHI tự tin cao là tiếng Anh.
    Với mọi ngôn ngữ khác (vi, pt, unknown...) → caller nên dịch cho an toàn.

    Cài thêm: pip install langid
    """
    try:
        import langid
        lang, confidence = langid.classify(text)
        # Chỉ tin là tiếng Anh khi langid nói 'en' VÀ có độ tự tin tương đối
        # (confidence là log-prob âm: giá trị càng gần 0 = càng chắc)
        if lang == "en":
            return "en"
        # Với mọi ngôn ngữ khác (kể cả tiếng Việt không dấu bị nhận nhầm thành pt/it)
        # → trả về 'vi' để caller dịch. An toàn hơn là bỏ qua dịch.
        return lang
    except ImportError:
        # langid chưa cài → fallback heuristic cũ (isascii) nhưng cảnh báo
        print("[tokenizer_utils] ⚠️  langid chưa được cài. Dùng fallback isascii() "
              "để detect ngôn ngữ — kém chính xác với tiếng Việt không dấu. "
              "Khuyến nghị: pip install langid")
        return "en" if text.isascii() else "vi"
    except Exception as e:
        print(f"[tokenizer_utils] ⚠️  langid lỗi ({e}), fallback về 'unknown'.")
        return "unknown"
