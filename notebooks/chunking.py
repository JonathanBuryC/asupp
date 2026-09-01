from pathlib import Path
import sys

# Add the project root directory to sys.path
sys.path.append(str(Path().resolve().parent))
from src.constants.paths import SECRET_PATH

from src.processing.pde_ple import PDE, PLE, es
from src.processing.document import Document

# get all paths
from src.constants.constants import (
    USEFUL_EXTENSIONS,
    ENVIRONNEMENT_PATH,
    S3_RAW_DOCS_PATH,
    SERVICES_PATH,
    ENVIRONNEMENT_RAW_PATH,
)
import pandas as pd
from src.processing.utils import *

from elasticsearch import Elasticsearch, helpers
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

# Configuration
source_indices = ["uc202-rex", "uc202-rex-cameleon","uc202-pj"]
chunk_index = "uc202-rex-chunks-elecbert-256"
scroll = "15m"  # Increase scroll time to avoid timeouts
bulk_size = 1000

# Initialize the splitter
text_splitter = SentenceTransformersTokenTextSplitter(
    model_name="/opt/app-root/src/uc202-ipn-rex/src/models/models/model_tigran_finetuned",
    tokens_per_chunk=256,
    chunk_overlap=25,
)

# Create the target index if it doesn't exist
if not es.indices.exists(index=chunk_index):
    es.indices.create(index=chunk_index)

# Get already processed documents
existing_docs = es.search(
    index=chunk_index,
    body={
        "size": 0,
        "aggs": {"unique_docs": {"cardinality": {"field": "original_doc_id.keyword"}}},
    },
)
print(
    f"Documents already processed: {existing_docs['aggregations']['unique_docs']['value']}"
)

# Retrieve processed IDs
processed_ids = set()
scroll_resp = es.search(
    index=chunk_index, scroll="5m", size=500, body={"query": {"match_all": {}}}
)
while scroll_resp["hits"]["hits"]:
    processed_ids.update(
        [doc["_source"]["original_doc_id"] for doc in scroll_resp["hits"]["hits"]]
    )
    scroll_resp = es.scroll(scroll_id=scroll_resp["_scroll_id"], scroll="5m")

print(
    f"{len(processed_ids)} documents already present. Processing will resume from remaining documents."
)

total_processed = 0

try:
    for original_index in source_indices:
        print(f"\nProcessing index: {original_index}")

        query = {"query": {"match_all": {}}}
        response = es.search(index=original_index, scroll=scroll, size=1000, body=query)
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]

        actions = []

        while hits:
            for doc in hits:
                doc_id = doc["_id"]
                source = doc["_source"]
                content = source.get("content", "")

                if doc_id in processed_ids:
                    continue  # Skip already processed documents

                if not content.strip():
                    continue  # Skip empty documents

                chunks = text_splitter.split_text(content)

                for idx, chunk in enumerate(chunks):
                    chunk_doc = {
                        "_index": chunk_index,
                        "_source": {
                            "chunk_content": chunk,
                            "original_doc_id": doc_id,
                            "source_index": original_index,  # Keep track of source index
                            "chunk_id": idx,
                            **{k: v for k, v in source.items() if k != "content"},
                        },
                    }
                    actions.append(chunk_doc)

                    if len(actions) >= bulk_size:
                        helpers.bulk(es, actions)
                        total_processed += len(actions)
                        print(f"{total_processed} chunks indexed.")
                        actions = []

            response = es.scroll(scroll_id=scroll_id, scroll=scroll)
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]

        if actions:
            helpers.bulk(es, actions)
            total_processed += len(actions)
            print(f"{total_processed} chunks indexed in total for {original_index}.")

    print(
        f"\nAll indices processed successfully. Total chunks indexed: {total_processed}"
    )

except Exception as e:
    print(f"Error encountered: {e}")
    print(
        f"{total_processed} chunks processed before the error. You can restart to resume."
    )
