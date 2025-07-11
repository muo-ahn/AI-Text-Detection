# src/sae/utils/embedding_save.py

import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_bert_embeddings(texts, batch_size=4):
    tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    model = BertModel.from_pretrained("klue/bert-base").to(DEVICE).eval()

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="🔄 임베딩 저장 중"):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            cls = model(**inputs).last_hidden_state[:, 0, :].cpu()
            embeddings.append(cls)
            torch.cuda.empty_cache()

    return torch.cat(embeddings)

if __name__ == "__main__":
    os.makedirs("data/embedding", exist_ok=True)
    df = pd.read_csv("data/train.csv")
    df.columns = df.columns.str.strip()

    features = get_bert_embeddings(df["full_text"].tolist())
    torch.save(features, "data/embedding/features.pt")
    torch.save(torch.tensor(df["generated"].tolist()), "data/embedding/labels.pt")
    print("✅ features.pt, labels.pt 저장 완료")
