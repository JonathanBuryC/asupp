# chunking.py

from elasticsearch import Elasticsearch, helpers
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from src.constants.chunking import (MODEL_NAME, TOKENS_PER_CHUNK,OVERLAP,BULK_SIZE,SCROLL_CHUNKING,SCROLL_PROCESSED_ID,TEXT_ATTRIBUTE,GET_ALL_QUERY)
class Chunker:
    def __init__(
        self,
        es_client: Elasticsearch,
        chunk_index: str,
        model_name: str = MODEL_NAME,
        tokens_per_chunk: int = TOKENS_PER_CHUNK,
        chunk_overlap: int = OVERLAP,
        bulk_size: int = BULK_SIZE,
        scroll_time: str = SCROLL_CHUNKING,
    ):
        self.es = es_client
        self.chunk_index = chunk_index
        self.bulk_size = bulk_size
        self.scroll_time = scroll_time
        self.text_splitter = SentenceTransformersTokenTextSplitter(
            model_name=model_name,
            tokens_per_chunk=tokens_per_chunk,
            chunk_overlap=chunk_overlap,
        )

        self.create_index_if_not_exists()

    def create_index_if_not_exists(self):
        if not self.es.indices.exists(index=self.chunk_index):
            self.es.indices.create(index=self.chunk_index)

    def get_already_processed_ids(self):
        processed_ids = set()
        scroll_resp = self.es.search(
            index=self.chunk_index,
            scroll=SCROLL_PROCESSED_ID,
            size=BULK_SIZE,
            body=GET_ALL_QUERY,
        )
        while scroll_resp["hits"]["hits"]:
            processed_ids.update(
                [
                    doc["_source"]["original_doc_id"]
                    for doc in scroll_resp["hits"]["hits"]
                ]
            )
            scroll_resp = self.es.scroll(
                scroll_id=scroll_resp["_scroll_id"], scroll=SCROLL_PROCESSED_ID
            )
        return processed_ids

    def process_index(self, source_index,text_attribute=TEXT_ATTRIBUTE):
        processed_ids = self.get_already_processed_ids()
        #processed_ids = []

        print(f"\nProcessing index: {source_index}")
        total_processed = 0
        query = GET_ALL_QUERY

        response = self.es.search(
            index=source_index, scroll=self.scroll_time, size=BULK_SIZE, body=query
        )
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]

        actions = []

        while hits:
            for doc in hits:
                doc_id = doc["_id"]
                source = doc["_source"]
                content = source.get(text_attribute, "")

                if doc_id in processed_ids:
                    continue

                if not content.strip():
                    continue

                chunks = self.text_splitter.split_text(content)

                for idx, chunk in enumerate(chunks):
                    chunk_doc = {
                        "_index": self.chunk_index,
                        "_source": {
                            "chunk_content": chunk,
                            "original_doc_id": doc_id,
                            "source_index": source_index,
                            "chunk_id": idx,
                            **{k: v for k, v in source.items() if k != text_attribute},
                        },
                    }
                    actions.append(chunk_doc)

                    if len(actions) >= self.bulk_size:
                        helpers.bulk(self.es, actions)
                        total_processed += len(actions)
                        print(f"{total_processed} chunks indexed.")
                        actions = []

            response = self.es.scroll(scroll_id=scroll_id, scroll=self.scroll_time)
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]

        if actions:
            helpers.bulk(self.es, actions)
            total_processed += len(actions)
            print(f"{total_processed} chunks indexed in total for {source_index}.")

        return total_processed

    def process_all_indices(self, source_indices: list):
        grand_total = 0
        try:
            for index in source_indices:
                processed = self.process_index(index)
                grand_total += processed
            print(
                f"\nAll indices processed successfully. Total chunks indexed: {grand_total}"
            )
        except Exception as e:
            print(f"Error encountered: {e}")
            print(
                f"{grand_total} chunks processed before the error. You can restart to resume."
            )
