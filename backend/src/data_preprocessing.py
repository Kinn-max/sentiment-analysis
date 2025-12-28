# src/data_preprocessing.py
"""
Tiền xử lý text tiếng Việt cho sentiment analysis.
Hàm chính: clean_text() và preprocess_series()
"""

import re
from underthesea import word_tokenize


# Tùy bạn có thể mở file stopwords Việt ở data/vn_stopwords.txt nếu muốn
DEFAULT_STOPWORDS = {
    "của","và","là","có","cho","không","một","những","rất","nói",
    "đã","sẽ","đang","được","cái","mình","anh","chị","em","với","ra"
}

def remove_urls_mentions_hashtags(text: str) -> str:
    text = re.sub(r"https?:\/\/\S+", " ", text)  # url
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)           # mentions
    text = re.sub(r"#\w+", " ", text)           # hashtags
    return text

def normalize_unicode(text: str) -> str:
    # nếu cần xử lý sâu hơn, có thể dùng thư viện pyvi hoặc vncorenlp
    # ở đây chỉ strip whitespace và lower
    return text.strip().lower()

def remove_special_chars(text: str) -> str:
    # giữ lại tiếng Việt, chữ và khoảng trắng
    text = re.sub(r"[^0-9a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", " ", text)
    # loại nhiều khoảng trắng
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize_vietnamese(text: str) -> str:
    # underthesea.word_tokenize trả về string với dấu _ giữa từ (ví dụ: "học sinh" -> "học sinh")
    tokens = word_tokenize(text, format="text")
    return tokens

def remove_stopwords(tokenized_text: str, stopwords: set = None) -> str:
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    tokens = tokenized_text.split()
    filtered = [t for t in tokens if t not in stopwords and len(t) > 1]
    return " ".join(filtered)

def clean_text(text: str, stopwords: set = None) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    t = remove_urls_mentions_hashtags(t)
    t = normalize_unicode(t)
    t = remove_special_chars(t)
    t = tokenize_vietnamese(t)
    t = remove_stopwords(t, stopwords)
    return t

def preprocess_series(series, stopwords: set = None):
    """
    Input: pandas Series of raw texts
    Output: pandas Series of cleaned (tokenized) texts
    """
    return series.fillna("").astype(str).apply(lambda x: clean_text(x, stopwords))
