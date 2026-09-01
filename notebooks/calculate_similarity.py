from pathlib import Path
import sys
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Add the project root directory to sys.path
sys.path.append(str(Path().resolve().parent))

# Models list
model_names = [
    "intfloat/multilingual-e5-large-instruct",
    "/opt/app-root/src/uc202-ipn-rex/src/models/models/model_tigran",
    "/opt/app-root/src/uc202-ipn-rex/src/models/models/model_tigran_finetuned",
    "/opt/app-root/src/uc202-ipn-rex/src/models/models/test",
]

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Preload all models once
models = {name: SentenceTransformer(name, device=device) for name in model_names}


def calculate_similarity(model, text1, text2):
    """Compute cosine similarity between two texts using a given model."""
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)

    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    return similarity


def main():
    while True:
        # User input
        text1 = input("Enter the first text (or type 'exit' to quit): ")
        if text1.lower() == "exit":
            break

        text2 = input("Enter the second text (or type 'exit' to quit): ")
        if text2.lower() == "exit":
            break

        print("\n--- Similarity Results ---")
        # Compute and display similarity for each model
        for name, model in models.items():
            try:
                similarity = calculate_similarity(model, text1, text2)
                print(f"{name}: {similarity:.4f}")
            except Exception as e:
                print(f"{name}: ❌ Error - {e}")

        print("--------------------------\n")


if __name__ == "__main__":
    main()
