# src/sae/inference/sae_predict_dataloader.py

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
import pandas as pd
from tqdm import tqdm
from src.sae.model.sae import SAEClassifier
from src.sae.utils.embedding import get_bert_embeddings
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 테스트 데이터
df = pd.read_csv("data/test.csv")
paragraphs = df["paragraph_text"].tolist()
ids = df["ID"].tolist()

# 2. 임베딩
features = get_bert_embeddings(paragraphs, batch_size=4).to(DEVICE)

# 3. 모델 로딩
model = SAEClassifier(input_dim=768, latent_dim=128).to(DEVICE)
model.load_state_dict(torch.load("model/sae_model.pt", map_location=DEVICE))
model.eval()

# 4. 소프트맥스 추론
probs = []
with torch.no_grad():
    for i in tqdm(range(0, len(features), 256)):
        batch = features[i:i+256]
        logits, _ = model(batch)
        prob = F.softmax(logits, dim=1)[:, 1]  # AI 확률
        probs.extend(prob.cpu().numpy())

# 5. 저장
pd.DataFrame({
    "ID": ids,
    "generated": probs
}).to_csv("result.csv", index=False)

print("✅ result.csv 저장 완료")
