import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

DEVICE = torch.device("cuda")

def get_bert_embeddings(text_list, batch_size=4):
    tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    model = BertModel.from_pretrained("klue/bert-base")
    model.to(DEVICE)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size), desc="🔄 BERT 임베딩 중"):
            batch = text_list[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size())
            masked_embeddings = last_hidden * attention_mask
            sum_embeddings = masked_embeddings.sum(1)
            mean_pooled = sum_embeddings / attention_mask.sum(1)
            embeddings.append(mean_pooled.cpu())
        
            torch.cuda.empty_cache()  # ✅ 메모리 수동 정리

    return torch.cat(embeddings, dim=0)
