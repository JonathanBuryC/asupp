GET_ALL_QUERY = {"query": {"match_all": {}}}
#MODEL_NAME = ("/opt/app-root/src/uc202-ipn-rex/src/models/intfloat_multilingual_e5_large_instruct")
MODEL_NAME = ("/opt/app-root/src/uc202-ipn-rex/src/models/models/test")
MODEL_NAME_TEST = (
    "/opt/app-root/src/uc202-ipn-rex/notebooks/intfloat_multilingual_e5_large_instruct/test"
)
BATCH_SIZE=32
EMBEDDING_DIMS=1024
SCROLL_TIME='15m'
SIMILARITY='cosine'
BULK_SIZE=500 #for already processed

EMBEDDING_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "embedding": {
                "type": "dense_vector",
                "dims": 1024,  # dont worry This will be replaced dynamically
                "index": True,
                "similarity": "cosine", #same here
            },
            "chunk_id": {"type": "keyword"},
            "original_doc_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "chunk_content": {"type": "text"},
        }
    }
}