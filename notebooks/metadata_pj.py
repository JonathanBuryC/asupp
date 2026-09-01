import pandas as pd
from elasticsearch import Elasticsearch, helpers
from pathlib import Path
import sys

# Add the project root directory to sys.path
sys.path.append(str(Path().resolve().parent))
from src.constants.paths import SECRET_PATH


from src.elasticsearch.indexer import Indexer
from src.processing.pde_ple import PDE, es
from src.constants.constants import S3_RAW_DOCS_PATH

# === 1. Load and deduplicate CSV ===
csv_path = "export_pj.csv"
df = pd.read_csv(csv_path, sep=";")  # adjust sep if needed
df = df.drop_duplicates(subset=["url"])  # keep unique URLs

# Convert into a dict for quick lookup
url_map = {}
for _, row in df.iterrows():
    url_map[row["url"]] = {
        "nom": row["nom"],
        "url": row["url"],
        "reference": row["reference"],
    }

print(f"Loaded {len(url_map)} unique rows from CSV.")

index_name = "uc202-rex-chunks"
scroll_time = "5m"
batch_size = 1000

# === 3. Search docs from uc202-pj ===
query = {"query": {"term": {"source_index.keyword": "uc202-pj"}}}

response = es.search(index=index_name, scroll=scroll_time, size=batch_size, body=query)
scroll_id = response["_scroll_id"]
hits = response["hits"]["hits"]

updates = []
total_updated = 0

while hits:
    for doc in hits:
        doc_id = doc["_id"]
        source = doc["_source"]

        title = source.get("title", "")

        if not title:
            continue

        # Find if title is substring of any URL in CSV

        for url, meta in url_map.items():
            try:
                if title.lower() in url.lower():  # substring check
                    action = {
                        "_op_type": "update",
                        "_index": index_name,
                        "_id": doc_id,
                        "doc": {
                            "nom": meta["nom"],
                            "url": meta["url"],
                            "reference": meta["reference"],
                        },
                    }
                    updates.append(action)
                    break
            except Exception as e:
                print(title, "         :           ", url)
                print(e)  # stop after first match

        if len(updates) >= 500:
            helpers.bulk(es, updates)
            total_updated += len(updates)
            print(f"Updated {total_updated} documents so far.")
            updates = []

    response = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
    scroll_id = response["_scroll_id"]
    hits = response["hits"]["hits"]

# Final batch
if updates:
    helpers.bulk(es, updates)
    total_updated += len(updates)

print(f"Update finished. Total documents updated: {total_updated}")
