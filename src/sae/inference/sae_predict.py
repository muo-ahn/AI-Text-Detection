import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
import pandas as pd
from tqdm import tqdm

from transformers import BertTokenizer, BertModel
from src.sae.model.sae import SAEClassifier
from src.sae.utils.embedding import get_bert_embeddings

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로딩
df = pd.read_csv("data/test.csv", encoding="utf-8-sig")
df.columns = df.columns.str.strip()
paragraphs = df["paragraph_text"].tolist()

# BERT 임베딩
print("🔄 BERT 임베딩 중...")
features = get_bert_embeddings(paragraphs, batch_size=4).to(DEVICE)

# SAE 모델 로딩
model = SAEClassifier(input_dim=768).to(DEVICE)
model.load_state_dict(torch.load("model/sae_model.pt", map_location=DEVICE))
model.eval()

# 예측
print("🔍 예측 중...")
with torch.no_grad():
    logits, _ = model(features)
    preds = torch.argmax(logits, dim=1).cpu().numpy()

# 결과 저장
result_df = pd.DataFrame({
    "ID": df["ID"],
    "generated": preds
})
result_df.to_csv("result.csv", index=False, encoding="utf-8-sig")

print("✅ result.csv 저장 완료.")
