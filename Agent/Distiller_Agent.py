
import os, json, faiss, numpy as np, torch
from torch import cuda
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from tqdm import tqdm

# ---------- 超参数 ----------
input_json  = ""
output_json = ""
embedding_cache = "embeddings_book.npy"
num_clusters = 10
epsilon = 0.01
batch_size = 256
model_name  = "bert-base-uncased"
# -----------------------------

def sort_by_centroid_distance(embeddings, centroid, descending=True):
    dist = cdist(embeddings, centroid.reshape(1, -1), "euclidean")
    order = np.argsort(dist, axis=0)
    return embeddings[order[::-1 if descending else 1].flatten()]

def load_json_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [rec.get("text", "") for rec in data]
    return data, texts

def save_json_records(records, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def get_embeddings(texts, tokenizer, model, batch_size=16):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings", unit="batch"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        embs.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
    return np.vstack(embs)


records, all_texts = load_json_records(input_json)


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda" if cuda.is_available() else "cpu")

if os.path.exists(embedding_cache):
    embeddings = np.load(embedding_cache)
    print(f"Loaded cached embeddings: {embeddings.shape}")
else:
    embeddings = get_embeddings(all_texts, tokenizer, model, batch_size)
    np.save(embedding_cache, embeddings)
    print(f"Saved embeddings to {embedding_cache}")


embeddings = F.normalize(torch.tensor(embeddings), p=2, dim=-1).numpy()


d = embeddings.shape[1]
kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=True, spherical=True)
kmeans.train(embeddings)

centroids = kmeans.centroids
centroid_index = faiss.IndexFlatL2(d)
centroid_index.add(centroids)
_, cluster_ids = centroid_index.search(embeddings, 1)
cluster_ids = cluster_ids.flatten()


keep_indices = set()
for cid in range(num_clusters):
    inds = np.where(cluster_ids == cid)[0]
    if len(inds) == 0:
        continue
    embs = embeddings[inds]
    sorted_embs = sort_by_centroid_distance(embs, centroids[cid])
    sim = cosine_similarity(sorted_embs)
    max_sim = np.max(np.triu(sim, k=1), axis=0)
    keep_mask = max_sim <= 1 - epsilon
    keep_indices.update(inds[keep_mask])


filtered_records = [records[i] for i in sorted(keep_indices)]
save_json_records(filtered_records, output_json)
print(f"origin: {len(records)}  →  filtering: {len(filtered_records)}")
