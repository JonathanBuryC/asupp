from pathlib import Path
import sys
from elasticsearch import NotFoundError

# Add the project root directory to sys.path
sys.path.append(str(Path().resolve().parent))

from src.constants.paths import SECRET_PATH
from src.processing.pde_ple import PDE, PLE, es
from src.processing.document import Document
from src.constants.constants import (
    USEFUL_EXTENSIONS,
    ENVIRONNEMENT_PATH,
    S3_RAW_DOCS_PATH,
    SERVICES_PATH,
    ENVIRONNEMENT_RAW_PATH,
)
import pandas as pd
from src.processing.utils import *
from src.processing.anonymizer import Anonymizer

# Initialiser l'anonymiseur
a = Anonymizer()

# Définir l'index
index_name = "uc202-rex-chunks"


def convert_anonymization_segment(segment):
    return {
        "text": segment.text,
        "tag": segment.tag,
        "start": segment.start,
        "end": segment.end,
        "score": segment.score,
    }


def update_documents():
    documents_traités = 0
    scroll_timeout = "15m"
    batch_size = 100

    query = {"query": {"bool": {"must_not": {"exists": {"field": "anonymized_text"}}}}}

    try:
        response = es.search(
            index=index_name, body=query, size=batch_size, scroll=scroll_timeout
        )
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]

        while hits:
            for hit in hits:
                doc_id = hit["_id"]
                content = hit["_source"].get("chunk_content", "")

                try:
                    anonymized_text, anonymization_mapping = a.anonymize_text(
                        text=content, inspection=True
                    )
                    serializable_mapping = [
                        convert_anonymization_segment(segment)
                        for segment in anonymization_mapping
                    ]

                    update_body = {
                        "doc": {
                            "anonymized_text": anonymized_text,
                            "anonymization_mapping": serializable_mapping,
                        }
                    }

                    es.update(index=index_name, id=doc_id, body=update_body)

                    documents_traités += 1
                    print(f"Document traité : {documents_traités} | ID: {doc_id}")

                except Exception as e:
                    print(f"Erreur sur le document {doc_id} : {e}")
                    continue  # Skip document if it fails

            try:
                response = es.scroll(scroll_id=scroll_id, scroll=scroll_timeout)
                scroll_id = response["_scroll_id"]
                hits = response["hits"]["hits"]
            except NotFoundError as e:
                print(f"Scroll context expired: {e}")
                break

    except Exception as e:
        print(f"Erreur au démarrage du scroll : {e}")

    print(f"Nombre total de documents traités : {documents_traités}")


# Exécuter la fonction
update_documents()
