import torch
import pickle
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.constants.model import S3_PATH_BASE, LOCAL_BASE
from src.processing.pde_ple import  PDE
from typing import Optional
from huggingface_hub import configure_http_backend
import requests 

class Model:
    """
    A wrapper for SentenceTransformer models with helpers for download, finetune,
    save/load etc etc
    """

    def __init__(
        self,
        model_name: str,
        s3_path: Optional[str] = None,
        local_path: str = None,
        model: Optional[SentenceTransformer] = None,
        pickle_path: Optional[str|Path] = None
    ):

        self.model_name = model_name
        if not pickle_path:
            self.pickle_path = f"{LOCAL_BASE + self._sanitize_model_name()}.pickle"
        if not s3_path:
            self.s3_path = S3_PATH_BASE + self._sanitize_model_name()+".pickle"
        if not local_path:
            self.local_path = LOCAL_BASE + self._sanitize_model_name()
        if not os.path.exists(self.local_path) and not model:
            try:
                print (self.s3_path)
                self.model=self.load_from_s3()
            except Exception as e:
                print("couldn't load model from s3, downloading the model ..." )
                self.model=self.download()


    # -----------------
    # Model I/O
    # -----------------
    def download(self, trust_remote_code=True):
        """Download model from HuggingFace hub and store locally."""
        def backend_factory(): 
            session = requests.Session() 
            session.verify = False 
            return session 
        configure_http_backend(backend_factory=backend_factory) 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(
            self.model_name, trust_remote_code=trust_remote_code, device=device
        )
        
        self.model.save(str(self.local_path))
        print(f"Model downloaded and saved at {self.local_path}")
        self.save_pickle()
        self.upload_to_s3()
        return self.model
    
    def _sanitize_model_name(self):
        return self.model_name.replace("/", "_")
    
    def save_pickle(self, pickle_path: Optional[str] = None):
        """Serialize model as pickle file."""
        if self.model is None:
            raise ValueError("No model loaded to save.")
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model pickled at {self.pickle_path}")
    
    def load_pickle(self, pickle_path: str=None):
        """Load model from pickle file."""
        if not os.path.exists(self.pickle_path):
            print(f"Model pickel not found locally {self.local_path}. creating the pickle...")
            self.save_pickle()
        with open(self.pickle_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {self.pickle_path}")
        return self.model

    
    def upload_to_s3(self, bucket_path: str=None):
        """Upload pickle to S3."""
        if not os.path.exists(self.pickle_path):
            print(
                f"Model pickel not found locally {self.local_path}. creating the pickle..."
            )
            self.save_pickle()
        if not bucket_path:
            bucket_path=self.s3_path    
        PDE.s3.upload(local_file=self.pickle_path, path=bucket_path)
        print(f"Model uploaded to {bucket_path}")

    def load_from_s3(self, bucket_path:str=None):
        """Load pickle from S3."""
        if not bucket_path:
            bucket_path=self.s3_path
        self.model = PDE.s3.load_pickle(bucket_path)
        print(f"Model loaded from {bucket_path}")
        return self.model



    