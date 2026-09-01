from pathlib import Path
import sys

# Add the project root directory to sys.path
sys.path.append(str(Path().resolve().parent))
from src.constants.paths import SECRET_PATH


from src.elasticsearch.indexer import Indexer
from src.processing.pde_ple import PDE, es
from src.constants.constants import S3_RAW_DOCS_PATH
from src.processing.chunker import Chunker

chunker = Chunker(
    es_client=es,
    chunk_index="uc202-rex-chunks",  # reuse your chunk index
)

# Process the new index "uc202-pj"
total_chunks = chunker.process_index("uc202-rex-cameleon")

print(f"Finished indexing {total_chunks} chunks from uc202-rex-cameleon")