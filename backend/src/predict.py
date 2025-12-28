# src/predict.py
"""
Hàm tiện ích để load model + vectorizer + label encoder và predict.
Sử dụng trong app Streamlit hoặc gọi bằng script khác.
"""

import os
import joblib
import numpy as np

MODEL_DIR = "models"

class SentimentPredictor:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        mpath = os.path.join(model_dir, "sentiment_model.pkl")
        vpath = os.path.join(model_dir, "vectorizer.pkl")
        lpath = os.path.join(model_dir, "label_encoder.pkl")

        if not os.path.exists(mpath) or not os.path.exists(vpath) or not os.path.exists(lpath):
            raise FileNotFoundError("Không tìm thấy model/vectorizer/label_encoder trong models/. Hãy chạy train_model.py trước")

        self.clf = joblib.load(mpath)
        self.vectorizer = joblib.load(vpath)
        self.le = joblib.load(lpath)

    def predict(self, texts):
        """
        texts: str hoặc list[str]
        return: list of dict {label: str, prob: float}
        """
        single = False
        if isinstance(texts, str):
            texts = [texts]
            single = True

        X = self.vectorizer.transform(texts)
        probs = self.clf.predict_proba(X)
        preds = self.clf.predict(X)
        labels = self.le.inverse_transform(preds)

        results = []
        for i in range(len(texts)):
            # lấy xác suất của nhãn dự đoán
            pred_label = labels[i]
            pred_prob = float(np.max(probs[i]))
            results.append({"text": texts[i], "label": pred_label, "prob": pred_prob})

        return results[0] if single else results

if __name__ == "__main__":
    # ví dụ quick test
    sp = SentimentPredictor()
    samples = ["Sản phẩm tốt, giao nhanh", "Hàng lỗi, sẽ trả lại"]
    print(sp.predict(samples))
