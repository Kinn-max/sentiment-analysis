import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from src.data_preprocessing import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def main():
    # Load data
    data_path = os.path.join(BASE_DIR, "data", "reviews.csv")
    print(f"📂 Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Không tìm thấy {data_path}. Hãy chạy create_training_data.py trước!")
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} reviews")
    print(f"📊 Label distribution:\n{df['label'].value_counts()}\n")

    # Clean text
    print("🧹 Cleaning text...")
    df["cleaned"] = df["review"].astype(str).apply(clean_text)

    # Encode labels
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])
    print(f"🏷️  Labels: {le.classes_}\n")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"],
        df["label_encoded"],
        test_size=0.2,
        random_state=42,
        stratify=df["label_encoded"]
    )
    print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}\n")

    # Vectorize
    print("🔢 Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    print("🎓 Training model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Training completed!")
    print(f"📈 Accuracy: {accuracy:.2%}\n")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save models
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n💾 Saving models...")
    joblib.dump(model, os.path.join(models_dir, "sentiment_model.pkl"))
    joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))
    joblib.dump(le, os.path.join(models_dir, "label_encoder.pkl"))
    
    print("✅ Models saved to models/")
    print("\n🎉 Training process completed successfully!")

if __name__ == "__main__":
    main()