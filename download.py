import os
from huggingface_hub import snapshot_download
from datasets import load_dataset

print("=== Downloading SubgraphRAG scored triples ===")
local_dir = snapshot_download(
    repo_id="siqim311/SubgraphRAG",
    repo_type="model",
    local_dir="./subgraphrag_data",
)
print(f"Downloaded to: {local_dir}")

print("\n=== Files in subgraphrag_data/ ===")
for root, dirs, files in os.walk("./subgraphrag_data"):
    dirs[:] = [d for d in dirs if not d.startswith(".")]
    for f in files:
        path = os.path.join(root, f)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {path}  ({size_mb:.1f} MB)")

print("\n=== Caching WebQSP test split ===")
dataset = load_dataset("rmanluo/RoG-webqsp", cache_dir="./hf_cache")
test_split = dataset["test"]
print(f"Test set size: {len(test_split)}")

sample = test_split[0]
print("\n=== Sample test question ===")
print(f"  id:       {sample['id']}")
print(f"  question: {sample['question']}")
print(f"  q_entity: {sample['q_entity']}")
print(f"  a_entity: {sample['a_entity']}")
print(f"  graph (first 3 edges): {sample['graph'][:3]}")
print(f"  keys: {list(sample.keys())}")
