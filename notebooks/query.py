from pathlib import Path
import sys
import re
import csv 
import os
# Add the project root directory to sys.path
sys.path.append(str(Path().resolve().parent))
from src.processing.pde_ple import es
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from src.processing.utils import clean_text
from collections import defaultdict

import pandas as pd
from datetime import datetime



# Parameters
embedding_index = "uc202-rex-embeddings"
#embedding_index = "uc202-rex-embeddings-trained"
model_name = ("/opt/app-root/src/uc202-ipn-rex/src/models/intfloat_multilingual_e5_large_instruct")
#model_name = ("/opt/app-root/src/uc202-ipn-rex/notebooks/intfloat_multilingual_e5_large_instruct/test")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(model_name, device=device)


def semantic_search(query_text, top_k=3):
    with torch.no_grad():
        query_embedding = model.encode(query_text, convert_to_numpy=True).tolist()
        print("query encoded successfully, len(embedding) : ", len(query_embedding))

    query_body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding_intfloat/multilingual-e5-large') + 1.0",
                    "params": {"query_vector": query_embedding},
                },
            }
        },
    }

    response = es.search(index=embedding_index, body=query_body)

    results = []
    for hit in response["hits"]["hits"]:
        results.append(
            {
                "original_doc_id": hit["_source"]["metadata"],
                "score": hit["_score"],
                "chunk_content": hit["_source"]["chunk_content"],
                "chunk_id": hit["_source"]["chunk_id"],
                "source":"semantic"
            }
        )
    
    return results


def lexical_search(query_text, top_k=3):
    query_body = {
        "size": top_k,
        "query": {"multi_match": {"query": query_text, "fields": ["chunk_content"]}},
    }

    response = es.search(index=embedding_index, body=query_body)

    results = []
    for hit in response["hits"]["hits"]:
        results.append(
            {
                "original_doc_id": hit["_source"]["metadata"],
                "score": hit["_score"],
                "chunk_content": hit["_source"]["chunk_content"],
                "chunk_id": hit["_source"]["chunk_id"],
                "source":"lexical"
            }
        )
    return results

def get_all_sigles():
    """Fetch all sigles from the dictionary index"""
    query_body = {"size": 10000, "_source": ["sigle"], "query": {"match_all": {}}}

    response = es.search(index="lexique_nucleaire", body=query_body)
    sigles = [hit["_source"]["sigle"] for hit in response["hits"]["hits"]]
    return sigles


def detect_sigles(query_text, known_sigles):
    """Detect sigles present in the query"""
    detected = []
    for sigle in known_sigles:
        if re.search(rf"\b{re.escape(sigle)}\b", query_text):
            detected.append(sigle)
    return detected

def fetch_definitions(sigles):
    """Fetch definitions from Elasticsearch for a list of sigles"""
    definitions = {}
    for sigle in sigles:
        query_body = {"size": 10, "query": {"term": {"sigle":{"value": sigle}}}}
        response = es.search(index="lexique_nucleaire", body=query_body)
        definitions[sigle] = [
            hit["_source"]["definition"] for hit in response["hits"]["hits"]
        ]
    return definitions


def expand_query(original_query, definitions_dict):
    """Create expanded query using OR operators for each detected sigle"""
    expanded_parts = []
    for sigle, definitions in definitions_dict.items():
        terms = [f'"{definition}"' for definition in definitions]
        terms.insert(0, sigle)
        expanded_parts.append(f"({' OR '.join(terms)})")

    expanded_query = " AND ".join(expanded_parts)

    if expanded_query:
        print(f"\nQuery expanded to: {expanded_query}\n")
        return f"{original_query} AND {expanded_query}"
    else:
        return original_query
    
def rrf_fusion(semantic_results, lexical_results, k=5, top_k=3):
    scores = defaultdict(float)
    seen_docs = {}

    for i, result_list in enumerate([semantic_results, lexical_results]):
        for rank, doc in enumerate(result_list):
            doc_id = doc["original_doc_id"]["original_doc_id"]
            #print("ICI DEBUG : ", doc_id)
            rrf_score = 1 / (k + rank + 1)
            scores[doc_id] += rrf_score
            if doc_id not in seen_docs or seen_docs[doc_id]["rrf_score"] < scores[doc_id]:
                seen_docs[doc_id] = {**doc, "rrf_score": scores[doc_id]}

    # Sort by RRF score and take top_k
    return sorted(seen_docs.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]

def combined_search(query_text, top_k=3, use_dictionary=False):
    if use_dictionary:
        # Get all known sigles
        known_sigles = get_all_sigles()

        # Detect sigles in the query
        detected_sigles = detect_sigles(query_text, known_sigles)

        if detected_sigles:
            # Fetch their definitions
            definitions_dict = fetch_definitions(detected_sigles)

            # Expand the query
            query_text = expand_query(query_text, definitions_dict)
            print(f"query with dictionary : {query_text}")
    print("\nPerforming semantic search...")
    semantic_results = semantic_search(query_text, top_k)
    print("Performing lexical search...")
    lexical_results = lexical_search(query_text, top_k)
    print("Fusing results with RRF...")
    rrf_results = rrf_fusion(semantic_results, lexical_results, top_k=top_k)

    return {
        f"semantic_top {top_k}": semantic_results,
        f"lexical_top {top_k}": lexical_results,
        f"rrf_top {top_k}": rrf_results,
    }


def save_to_excel(query, results, use_dictionary,top_k):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"search_results_{query.replace(' ', '_')[:30]}_{timestamp}.xlsx"

    semantic_df = pd.DataFrame(
        [
            {
                "Doc ID": res["original_doc_id"],
                "Score": res["score"],
                "Content": clean_text(res["chunk_content"]),
                "Annotation": None
            }
            for res in results[f"semantic_top {top_k}"]
        ]
    )

    lexical_df = pd.DataFrame(
        [
            {
                "Doc ID": res["original_doc_id"],
                "Score": res["score"],
                "Content": clean_text(res["chunk_content"]),
                "Annotation": None,
            }
            for res in results[f"lexical_top {top_k}"]
        ]
    )
    rrf_df = pd.DataFrame(
        [
            {
                "Doc ID": res["original_doc_id"],
                "RRF Score": res["rrf_score"],
                "Content": clean_text(res["chunk_content"]),
                "Annotation": None,
            }
            for res in results[f"rrf_top {top_k}"]
        ]
    )

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        semantic_df.to_excel(writer, sheet_name="Semantic Search", index=False)
        lexical_df.to_excel(writer, sheet_name="Lexical Search", index=False)
        rrf_df.to_excel(writer, sheet_name="Hybrid RRF Search", index=False)

    print(f"Results saved to Excel: {filename}")
def main():
    while True:
        # Ask user for input
        query = input("\nPlease enter your search query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        top_k_input = input("Combien de docs pertinents: ")
        use_dictionary_input = input(
            "Est-ce que tu veux utiliser le dictionnaire ? (oui/non) : "
        )
        use_dictionary = use_dictionary_input.lower() == "oui"

        # Convert top_k to integer
        try:
            top_k = int(top_k_input)
        except ValueError:
            print("Please enter a valid number for the number of documents.")
            continue

        # Assuming combined_search is a function you have defined elsewhere
        results = combined_search(query, top_k, use_dictionary)

        print(f"\nSemantic Search Top {top_k} Results:")
        for res in results[f"semantic_top {top_k}"]:
            print(f"\nDoc ID: {res['original_doc_id']}, Score: {res['score']}\n\n")
            print(f"Content: {clean_text(res['chunk_content'])}\n\n\n")

        print(f"\nLexical Search Top {top_k} Results:")
        for res in results[f"lexical_top {top_k}"]:
            print(f"\nDoc ID: {res['original_doc_id']}, Score: {res['score']}\n\n")
            print(f"Content: {clean_text(res['chunk_content'])}\n\n\n")

        # Save results to CSV
        save_to_excel(query, results, use_dictionary,top_k)

if __name__ == "__main__":
    main()
