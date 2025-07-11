import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import torch
import torch.nn as nn
import pandas as pd
from src.sae.model.sae import SAEClassifier
from src.sae.utils.embedding import get_bert_embeddings
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

# 설정
DEVICE = torch.device("cuda")
MODEL_SAVE_PATH = "model/sae_model.pt"

# 데이터 불러오기
df = pd.read_csv("data/train.csv")
df.columns = df.columns.str.strip()
texts = df["full_text"].tolist()
labels = df["generated"].values

# BERT 임베딩 추출
features = get_bert_embeddings(texts).to(DEVICE)
labels = torch.tensor(labels).long().to(DEVICE)

# 모델 정의
model = SAEClassifier().to(DEVICE)
criterion_clf = nn.CrossEntropyLoss()
criterion_recon = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 학습
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    pred, recon = model(features)
    
    loss_clf = criterion_clf(pred, labels)
    loss_recon = criterion_recon(recon, features)
    loss = loss_clf + 0.1 * loss_recon  # reconstruction penalty 비율 조정 가능

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = accuracy_score(labels.cpu().numpy(), torch.argmax(pred, dim=1).cpu().numpy())
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

# 저장
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("✅ 모델 저장 완료:", MODEL_SAVE_PATH)
