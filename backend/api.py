
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.predict import SentimentPredictor
from src.crawler import extract_ids_from_url, crawl_shopee_reviews
from src.data_preprocessing import clean_text
import traceback

app = Flask(__name__)
CORS(app)

# Load model khi khởi động
print("🔄 Loading sentiment model...")
try:
    predictor = SentimentPredictor()
    print("✅ Model loaded successfully!\n")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("💡 Hãy chạy: python -m src.train_model")
    predictor = None
except Exception as e:
    print(f"❌ Unexpected error loading model: {e}")
    predictor = None

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": predictor is not None
    })

@app.route("/api/analyze", methods=["POST"])
def analyze():
    # Check if model is loaded
    if predictor is None:
        return jsonify({
            "error": "Model chưa được train. Chạy: python -m src.train_model",
            "items": []
        }), 500
    
    data = request.get_json()
    url = data.get("url", "")
    
    if not url:
        return jsonify({
            "error": "URL không được để trống",
            "items": []
        }), 400

    try:
        # Extract IDs from URL
        print(f"\n📦 Analyzing URL: {url}")
        shop_id, product_id = extract_ids_from_url(url)
        print(f"   Shop ID: {shop_id}, Item ID: {product_id}")
        
        # Crawl reviews (sẽ dùng demo nếu bị block)
        print("🕷️  Crawling reviews from Shopee...")
        reviews = crawl_shopee_reviews(shop_id, product_id, limit=50)
        
        # QUAN TRỌNG: Kiểm tra empty
        if not reviews or len(reviews) == 0:
            return jsonify({
                "error": "Không crawl được reviews. Shopee có thể đang chặn.",
                "items": [],
                "total": 0
            }), 404
        
        print(f"✓ Got {len(reviews)} reviews")
        
        # Clean text
        print("🧹 Cleaning text...")
        cleaned = [clean_text(r) for r in reviews]
        
        # Filter empty cleaned texts
        valid_pairs = [(r, c) for r, c in zip(reviews, cleaned) if c.strip()]
        
        if not valid_pairs:
            return jsonify({
                "error": "Không có reviews hợp lệ sau khi clean",
                "items": [],
                "total": 0
            }), 400
        
        reviews, cleaned = zip(*valid_pairs)
        reviews = list(reviews)
        cleaned = list(cleaned)
        
        # Predict sentiment
        print("🤖 Predicting sentiment...")
        results = predictor.predict(cleaned)
        
        print(f"✅ Analysis completed: {len(reviews)} reviews\n")
        
        # Build response
        items = []
        for i in range(len(reviews)):
            items.append({
                "text": reviews[i],
                "sentiment": results[i]["label"],
                "prob": round(float(results[i]["prob"]), 2)
            })
        
        return jsonify({
            "total": len(items),
            "items": items
        })
        
    except ValueError as e:
        print(f"❌ ValueError: {e}")
        return jsonify({
            "error": f"URL không hợp lệ: {str(e)}",
            "items": []
        }), 400
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Lỗi server: {str(e)}",
            "items": []
        }), 500

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Shopee Review Analyzer API")
    print("=" * 50)
    print(f"📍 Server: http://localhost:5000")
    print(f"📍 Health: http://localhost:5000/api/health")
    print(f"📍 Model loaded: {predictor is not None}")
    print("=" * 50)
    print()
    
    app.run(debug=True, port=5000, host='0.0.0.0')