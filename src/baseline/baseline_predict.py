# baseline_predict.py

import pandas as pd
import joblib

# 1. 모델과 벡터라이저 불러오기
model = joblib.load("model/baseline_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# 2. 테스트 데이터 불러오기
df_test = pd.read_csv("data/test.csv", encoding="utf-8-sig")
df_test.columns = df_test.columns.str.strip()

# 3. 텍스트 추출 및 벡터화
X_test = df_test["paragraph_text"]
X_vec = vectorizer.transform(X_test)

# 4. 예측
y_pred = model.predict(X_vec)

# 5. 결과 저장 (generated는 0: Human, 1: AI)
df_result = pd.DataFrame({
    "ID": df_test["ID"],
    "generated": y_pred
})
df_result.to_csv("result.csv", index=False, encoding="utf-8-sig")

print("✅ result.csv 저장 완료")
