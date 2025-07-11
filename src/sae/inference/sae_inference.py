# src/sae/inference/sae_predict_dataloader.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
import pandas as pd
from tqdm import tqdm
from src.sae.model.sae import SAEClassifier
from src.sae.utils.embedding import get_bert_embeddings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 테스트 데이터 불러오기
test_df = pd.read_csv("data/test.csv")
test_df.columns = test_df.columns.str.strip()
texts = test_df["paragraph_text"].tolist()
ids = test_df["ID"].tolist()

# 2. BERT 임베딩 추출
features = get_bert_embeddings(texts, batch_size=32).to(DEVICE)

# 3. SAE 모델 로딩
model = SAEClassifier(input_dim=768, latent_dim=128).to(DEVICE)
model.load_state_dict(torch.load("model/sae_model.pt", map_location=DEVICE))
model.eval()

# 4. 추론
predictions = []
with torch.no_grad():
    for i in tqdm(range(0, len(features), 256), desc="🔍 추론 중"):
        batch = features[i:i+256]
        logits, _ = model(batch)
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().tolist())

# 5. 결과 저장
result_df = pd.DataFrame({
    "ID": ids,
    "generated": predictions
})

result_df.to_csv("result.csv", index=False)
print("✅ result.csv 저장 완료")
