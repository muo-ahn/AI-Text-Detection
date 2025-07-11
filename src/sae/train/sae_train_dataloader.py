# src/sae/train/sae_train_dataloader.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from src.sae.model.sae import SAEClassifier
from collections import Counter

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.load("data/embedding/features.pt")
    y = torch.load("data/embedding/labels.pt").long()
    λ = 0.05  # reconstruction loss weight

    # ✅ 클래스 불균형 보정 weight 계산
    class_counts = torch.bincount(y)
    class_weights = class_counts.sum() / (2.0 * class_counts.float())
    print("Class counts:", class_counts.tolist())
    print("Class weights:", class_weights.tolist())

    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    model = SAEClassifier(input_dim=768, latent_dim=128).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ✅ weight 적용된 CrossEntropyLoss
    criterion_clf = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), reduction="mean")
    criterion_recon = nn.MSELoss()

    for epoch in range(10):
        model.train()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits, x_recon = model(xb)

            loss = criterion_clf(logits, yb) + λ * criterion_recon(x_recon, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(yb.cpu().tolist())

            correct += (pred == yb).sum().item()
            total += yb.size(0)

        acc = correct / total
        pred_counter = Counter(all_preds)

        print(classification_report(all_labels, all_preds, target_names=["Human", "AI"]))
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Acc: {acc:.4f} | Pred dist: {pred_counter}")

    # ✅ 모델 저장
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/sae_model.pt")
    print("✅ 모델 저장 완료")
