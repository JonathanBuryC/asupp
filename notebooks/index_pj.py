from pathlib import Path
import sys

# Add the project root directory to sys.path
sys.path.append(str(Path().resolve().parent))
from src.constants.paths import SECRET_PATH


from src.elasticsearch.indexer import Indexer
from src.processing.pde_ple import PDE, es
from src.constants.constants import S3_RAW_DOCS_PATH
from src.processing.embedder import Embedder

chunk_index = "uc202-rex-chunks"
embedding_index = "uc202-rex-embeddings"

# Instantiate the Embedder
embedder = Embedder(
    es_client=es,
    chunk_index=chunk_index,
    embedding_index=embedding_index,
)
print(embedder.model_name)
