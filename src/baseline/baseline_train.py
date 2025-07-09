# baseline_train.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# 1. 데이터 로딩
df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
df.columns = df.columns.str.strip()

# 2. 피처와 라벨 정의
X = df["full_text"]
y = df["generated"]

# 3. 벡터화
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_vec = vectorizer.fit_transform(X)

# 4. 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, stratify=y, random_state=42
)

# 5. 모델 학습
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. 모델 저장
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/baseline_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ 모델과 벡터라이저 저장 완료")
