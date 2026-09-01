from pathlib import Path
import sys
import pickle
from io import BytesIO

sys.path.append(str(Path().resolve().parent))
from src.processing.pde_ple import es, PDE
from sentence_transformers import SentenceTransformer
import torch
from src.api.constants.model import MODEL_NAME
from src.models.model import Model

m = Model("sentence-transformers/all-MiniLM-L6-v2")