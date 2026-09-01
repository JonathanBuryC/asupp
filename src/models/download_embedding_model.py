from sentence_transformers import SentenceTransformer
from huggingface_hub import configure_http_backend
import requests
def backend_factory():
    session = requests.Session()
    session.verify = False
    return session
configure_http_backend(backend_factory=backend_factory)
# Reste de ton code
model = SentenceTransformer("intfloat/multilingual-e5-large", trust_remote_code=True)
model.save("./intfloat_multilingual_e5_large_instruct")
