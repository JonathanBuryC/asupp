from pathlib import Path
import sys

# Add the project root directory to sys.path
sys.path.append(str(Path().resolve().parent))

from src.processing.pde_ple import es  # Assuming Elasticsearch instance is here
from elasticsearch import helpers
from sentence_transformers import SentenceTransformer
import torch

# Parameters
chunk_index = "uc202-rex-chunks-elecbert-256"
embedding_index = "uc202-rex-embeddings-elecbert-256"
batch_size = 1
model_name = "/opt/app-root/src/uc202-ipn-rex/src/models/models/model_tigran_finetuned"

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(model_name, device=device)
print("MODEL IS LOADED HALLELUJAH")
# Create embedding index if not exists
if not es.indices.exists(index=embedding_index):
    es.indices.create(
        index=embedding_index,
        body={
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 768,  #fais gaffe 
                        "index": True,
                        "similarity": "cosine",
                    },
                    "chunk_id": {"type": "keyword"},
                    "original_doc_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "chunk_content": {"type": "text"},
                }
            }
        },
    )

# Retrieve already embedded chunk_ids
print("Récupération des chunks déjà embarqués...")
existing_ids = set()

scroll = "15m"
query = {"query": {"match_all": {}}}

response = es.search(index=embedding_index, scroll=scroll, size=10000, body=query)
scroll_id = response["_scroll_id"]
hits = response["hits"]["hits"]

while hits:
    for doc in hits:
        existing_ids.add(doc["_source"]["chunk_id"])
        
    print("LONGUEUR SO FAR : ",len(existing_ids))

    response = es.scroll(scroll_id=scroll_id, scroll=scroll)
    scroll_id = response["_scroll_id"]
    hits = response["hits"]["hits"]

print(f"Nombre de chunks déjà embarqués : {len(existing_ids)}")

# Process new chunks
print("Démarrage de l'indexation des nouveaux chunks...")

query = {"query": {"match_all": {}}}
response = es.search(index=chunk_index, scroll=scroll, size=1000, body=query)
scroll_id = response["_scroll_id"]
hits = response["hits"]["hits"]

actions = []
total_embedded = 0

try:
    while hits:
        chunks = []
        metas = []

        for doc in hits:
            chunk_id = doc["_id"]
            source = doc["_source"]

            if chunk_id in existing_ids:
                continue  # Skip already embedded chunks

            chunk_content = source.get("chunk_content", "").strip()
            if not chunk_content:
                continue

            chunks.append(chunk_content)
            metas.append(
                {
                    "chunk_id": chunk_id,
                    "original_doc_id": source["original_doc_id"],
                    "chunk_index": source["chunk_id"],
                    "metadata": {
                        k: v for k, v in source.items() if k != "chunk_content"
                    },
                }
            )
            print("LONGEUR CHNUKS :",len(chunks))
            if len(chunks) >= batch_size:
                embeddings = model.encode(
                    chunks,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                )
                

                for i, embedding in enumerate(embeddings):
                    action = {
                        "_index": embedding_index,
                        "_source": {
                            "embedding": embedding.tolist(),
                            "chunk_content": chunks[i],
                            **metas[i],
                        },
                    }
                    
                    actions.append(action)
                
                
                helpers.bulk(es, actions)
                
                total_embedded += len(actions)
                
                print(f"{total_embedded} embeddings indexés.")

                actions = []
                chunks = []
                metas = []

        # Scroll to next batch
        response = es.scroll(scroll_id=scroll_id, scroll=scroll)
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]

    # Final batch
    if chunks:
        with torch.no_grad():
            embeddings = model.encode(
                chunks,
                convert_to_numpy=True,
                show_progress_bar=True,
                #batch_size=64
            )

        for i, embedding in enumerate(embeddings):
            action = {
                "_index": embedding_index,
                "_source": {
                    "embedding": embedding.tolist(),
                    "chunk_content": chunks[i],
                    **metas[i],
                },
            }
            actions.append(action)

        helpers.bulk(es, actions)
        total_embedded += len(actions)
        print(f"{total_embedded} embeddings indexés au total.")

except Exception as e:
    print(f"Erreur rencontrée : {e}")
    print(f"{total_embedded} embeddings traités avant l'erreur.")
