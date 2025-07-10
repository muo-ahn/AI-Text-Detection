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
DEVICE = torch.device("cuda")

# 데이터 로딩
df = pd.read_csv("data/test.csv")
paragraphs = df["paragraph_text"].tolist()

# BERT 임베딩
print("🔄 BERT 임베딩 중...")
features = get_bert_embeddings(paragraphs, batch_size=32).to(DEVICE)

# 모델 로딩
model = SAEClassifier(input_dim=768).to(DEVICE)
model.load_state_dict(torch.load("model/sae_model.pt", map_location=DEVICE))
model.eval()

# 추론
print("🔍 추론 중...")
with torch.no_grad():
    outputs = model(features)
    
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    preds = torch.argmax(logits, dim=1).cpu().numpy()

# 결과 저장
result_df = pd.DataFrame({
    "ID": df["ID"],
    "generated": preds
})
result_df.to_csv("result.csv", index=False, encoding="utf-8-sig")
print("✅ 결과 저장 완료: result.csv")
