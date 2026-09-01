from pathlib import Path
import sys
import os
import pandas as pd

from src.processing.pde_ple import PDE
from src.processing.document import Document
from src.constants.constants import (
    USEFUL_EXTENSIONS,

    ENVIRONNEMENT_RAW_PATH,
)
from src.constants.elastic import   (INDEX_NAME,
    MAX_WORKERS)
from src.processing.utils import clean_text, process_txt_file
from concurrent.futures import ThreadPoolExecutor, as_completed
from elasticsearch.helpers import bulk

from joblib import Parallel, delayed
from tqdm import tqdm


class Indexer:
    def __init__(self, es ,s3_paths, index_name=INDEX_NAME, metadata_csv_path=None,raw_path=None):
        self.s3_paths = s3_paths
        self.index_name = index_name
        self.es = es
        self.metadata_csv_path = metadata_csv_path
        self.metadata_dict = {}
        self.raw_path=raw_path

        if metadata_csv_path:
            self.load_metadata()

    def load_metadata(self):
        """Load and process metadata CSV if provided."""
        df = PDE.s3.read_dataframe(
            path=self.metadata_csv_path,
            file_format="csv",
            encoding="Windows-1252",
            sep=";",
            header=1,
        )
        df = df[df["Type extension de fichier"].isin(USEFUL_EXTENSIONS)]
        df["txt_file_name"] = df["Nom titre du fichier"].apply(
            lambda x: x.lower() + ".txt"
        )
        df.rename(
            columns={
                "Nom titre du fichier": "filename",
                "Type extension de fichier": "extension",
                "Date de création du fichier": "date_creation",
                "Date de dernière modification": "date_modification",
                "Nom du créateur/dernier modificateur": "auteur",
                "Chemin d'accès": "chemin",
                "Taille du fichier": "taille",
            },
            inplace=True,
        )
        df = df.drop_duplicates(subset="filename", keep="last")

        self.metadata_dict = df.set_index("filename")[
            [
                "extension",
                "date_creation",
                "date_modification",
                "auteur",
                "chemin",
                "taille",
            ]
        ].to_dict(orient="index")

    def is_already_indexed(self, doc_id):
        """Check if the document is already indexed."""
        return self.es.exists(index=self.index_name, id=doc_id)
    
    def process_file(self, file_path, raw_path=None):
        try:
            doc = Document(s3_path=file_path)

            if not doc.is_useful() or self.is_already_indexed(doc.raw_filename):
                return None

            doc.download()
            doc.extract_text_by_extension()
            process_txt_file(doc.local_txt_path)
            if raw_path:
                doc.upload_txt(raw_path)
                

            try:
                with open(doc.local_txt_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(doc.local_txt_path, "r", encoding="latin-1") as f:
                    content = f.read()

            filename = doc.filename.lower()
            metadata = self.metadata_dict.get(filename, {})

            body = {
                "title": doc.filename,
                "content": clean_text(content),
                **metadata,
            }
            
            try:
                self.es.index(index=self.index_name, id=doc.raw_filename, document=body)
                print(f"Indexed: {doc.filename}")
            except Exception as e:
                print(f"Failed to index {doc.filename}: {e}")

            doc.delete()
            return body
        except Exception as e:
            print(f"ERROR {e} with file {file_path}")
        


    def index_documents_parallel(self, n_jobs=MAX_WORKERS, raw_path=None):
        print("Starting parallel indexing...")

        # Separate PDF and non-PDF files
        pdf_files = [fp for fp in self.s3_paths if fp.lower().endswith(".pdf")]
        other_files = [fp for fp in self.s3_paths if not fp.lower().endswith(".pdf")]

        # Parallel processing for PDFs
        parallel_results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(self.process_file)(fp, raw_path)
            for fp in tqdm(pdf_files, desc="Indexing PDFs in parallel")
        )

        # Sequential processing for other files
        sequential_results = []
        for fp in tqdm(other_files, desc="Indexing other files sequentially"):
            result = self.process_file(fp, raw_path)
            sequential_results.append(result)

        # Summary
        total_indexed = len(
            [r for r in parallel_results + sequential_results if r is not None]
        )
        print(f"Successfully indexed {total_indexed} documents.")