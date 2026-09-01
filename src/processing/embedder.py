# embedding.py

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from src.constants.embedding import (
    MODEL_NAME,
    GET_ALL_QUERY,
    EMBEDDING_DIMS,
    BATCH_SIZE,
    SCROLL_TIME,
    SIMILARITY,
    EMBEDDING_INDEX_MAPPING,
    BULK_SIZE
)
import torch
import copy


class Embedder:
    

    def __init__(
        self,
        es_client: Elasticsearch,
        chunk_index: str,
        embedding_index: str,
        model_name: str = MODEL_NAME,
        batch_size: int = BATCH_SIZE,
        embedding_dims: int = EMBEDDING_DIMS,
        scroll_time: str = SCROLL_TIME,
    ):
        self.es = es_client
        self.chunk_index = chunk_index
        self.embedding_index = embedding_index
        self.batch_size = batch_size
        self.scroll_time = scroll_time
        self.embedding_dims = embedding_dims
        if '/' in MODEL_NAME:
            self.model_name = MODEL_NAME.split("/")[-2] + "_" + MODEL_NAME.split("/")[-1] #removes all the path if its local
        else : 
            self.model_name=model_name

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        print("Embedding model loaded successfully.")

        self.create_index_if_not_exists()

    def create_index_if_not_exists(self):
        if not self.es.indices.exists(index=self.embedding_index):
            mapping = copy.deepcopy(EMBEDDING_INDEX_MAPPING)
            mapping["mappings"]["properties"]["embedding"]["dims"] = self.embedding_dims
            mapping["mappings"]["properties"]["embedding"]["similarity"] = SIMILARITY

            self.es.indices.create(index=self.embedding_index, body=mapping)

    def get_already_embedded_ids(self):
        existing_ids = set()

        response = self.es.search(
            index=self.embedding_index,
            scroll=self.scroll_time,
            size=BULK_SIZE,
            body=GET_ALL_QUERY,
        )
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]

        while hits:
            for doc in hits:
                existing_ids.add(doc["_source"]["chunk_id"])

            print(f"Already embedded chunks: {len(existing_ids)}")

            response = self.es.scroll(scroll_id=scroll_id, scroll=self.scroll_time)
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]

        return existing_ids

    def embed_chunks(self,all=False):
        "all pcq parfois jveux tt re rembedder avec un nouveau embedder"
        existing_ids=[]
        if not all:
            existing_ids = self.get_already_embedded_ids()
            print(f"already {len(existing_ids)} documents embedded")
        total_embedded = 0

        response = self.es.search(
            index=self.chunk_index,
            scroll=self.scroll_time,
            size=BULK_SIZE,
            body=GET_ALL_QUERY,
        )
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]

        actions = []

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

                    if len(chunks) >= self.batch_size:
                        self.index_embeddings(chunks, metas, actions)
                        total_embedded += len(actions)
                        print(f"{total_embedded} embeddings indexed so far.")

                        actions = []
                        chunks = []
                        metas = []

                response = self.es.scroll(scroll_id=scroll_id, scroll=self.scroll_time)
                scroll_id = response["_scroll_id"]
                hits = response["hits"]["hits"]

            if chunks:
                self.index_embeddings(chunks, metas, actions)
                total_embedded += len(actions)
                print(f"{total_embedded} embeddings indexed in total.")

        except Exception as e:
            print(f"Error encountered: {e}")
            print(f"{total_embedded} embeddings processed before the error.")

    def index_embeddings(self, chunks, metas, actions, show_progress_bar=True):
        embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )

        for i, embedding in enumerate(embeddings):
            action = {
                "_index": self.embedding_index,
                "_source": {
                    "embedding": embedding.tolist(),
                    "chunk_content": chunks[i],
                    **metas[i],
                },
            }
            actions.append(action)

        helpers.bulk(self.es, actions)
