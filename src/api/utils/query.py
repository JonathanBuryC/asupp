from pathlib import Path
import sys
import re
import csv
import os
import json
import time
from datetime import datetime
from typing import Any

# Add the project root directory to sys.path
sys.path.append(str(Path().resolve().parent))
from src.processing.pde_ple import es, PDE
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch
import numpy as np
from src.processing.utils import clean_text
from collections import defaultdict
import difflib
import pandas as pd
from datetime import datetime
from src.api.constants.model import (
    MODEL_NAME,
    EMBEDDING_INDEX,
    EXTENSION_FAMILIES,
    MODEL_S3_PATH,
)
import logging

logging.basicConfig(level=logging.INFO)

# Create a logger object
logger = logging.getLogger(__name__)


# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"

#! WARNING:urllib3.connectionpool:Connection pool is full, discarding connection: s3-interne-dsit.edf.fr. Connection pool size: 10
#! donc je le charge le model dans l'image docker directement et tant pis

try:
    logger.info("Trying to load SentenceTransformer model from S3 pickle...")
    model = PDE.s3.load_piddckle(MODEL_S3_PATH)
    model.to(device)

    # ✅ tokenizer comes from the SentenceTransformer itself
    tokenizer = model[0].tokenizer

    logger.info("Successfully loaded model and tokenizer from S3.")

except Exception as e:
    logger.warning(
        "Failed to load model from S3 (%s). Falling back to MODEL_NAME='%s'.",
        e,
        MODEL_NAME,
    )

    try:
        model = SentenceTransformer(MODEL_NAME, device=device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        logger.info("Successfully loaded model and tokenizer from MODEL_NAME.")

    except Exception as e2:
        logger.error(
            "Failed to load model from both S3 and MODEL_NAME. Cannot continue.",
            exc_info=e2,
        )
        raise RuntimeError("Unable to load SentenceTransformer model") from e2


def count_tokens(text: str) -> int:
    """
    Compte le nombre de tokens utilisés par le tokenizer du modèle SentenceTransformer.
    """
    if not isinstance(text, str):
        raise ValueError("L'entrée doit être une chaîne de caractères.")

    encoded = tokenizer.encode(text, add_special_tokens=True)
    return len(encoded)


def build_filter_query(filters):
    """
    Construit le bloc 'filter' (Elasticsearch) à partir des filtres génériques
    et des filtres spécifiques Caméléon.
    """
    must_filters = []
    if not filters:
        return []

    # -----------------------
    # Filtres génériques
    # -----------------------
    extensions = filters.get("extension") or []
    sources = filters.get("source") or []
    date_start = filters.get("date_start")
    date_end = filters.get("date_end")

    if isinstance(extensions, str):
        extensions = [extensions]
    if isinstance(sources, str):
        sources = [sources]

    # ✅ Expand families
    all_extensions = []
    if "__all__" in extensions:
        # Include all variants from all families
        for family_exts in EXTENSION_FAMILIES.values():
            all_extensions.extend(family_exts)
    else:
        for ext in extensions:
            if ext in EXTENSION_FAMILIES:
                all_extensions.extend(EXTENSION_FAMILIES[ext])
            else:
                all_extensions.append(ext)

    if all_extensions:
        must_filters.append(
            {
                "bool": {
                    "should": [
                        {"terms": {"metadata.extension.keyword": all_extensions}},
                        {
                            "bool": {
                                "must_not": {
                                    "exists": {"field": "metadata.extension.keyword"}
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            }
        )

    # Source
    if sources and "__all__" not in sources:
        source_terms = []
        for src in sources:
            if src == "cameleon":
                source_terms.extend(["uc202-pj", "uc202-rex-cameleon"])
            else:
                source_terms.append(src)
        if source_terms:
            must_filters.append(
                {"terms": {"metadata.source_index.keyword": source_terms}}
            )

    # Date (si start & end)

    if date_start and date_end:
        # Determine which date fields to consider based on sources

        date_start = normalize_date(date_start)
        date_end = normalize_date(date_end)

        date_fields = set()

        # If user asked for any cameleon docs (or all), include cameleon date field
        if sources and (("cameleon" in sources) or ("__all__" in sources)):
            date_fields.add("metadata.coeur_date_constat_date")

        # If there are non-cameleon sources (or all), include the generic date_creation
        # Note: if sources is empty, we don't know; but if the search indexes include non-cameleon,
        # you can choose to include date_creation by default OR only when sources explicitly contain non-cameleon.
        # Here we include it when either "__all__" is present OR there is any source not equal to "cameleon".
        if (
            sources
            and ("__all__" in sources or any(src != "cameleon" for src in sources))
        ) or (not sources):
            date_fields.add("metadata.date_creation")

        # For each date field, add a per-field condition:
        # (range matches) OR (field is missing)
        for date_field in date_fields:
            must_filters.append(
                {
                    "bool": {
                        "should": [
                            {
                                "range": {
                                    date_field: {
                                        "gte": date_start,
                                        "lte": date_end,
                                        "format": "dd/MM/yyyy HH:mm:ss",
                                    }
                                }
                            },
                            {"bool": {"must_not": {"exists": {"field": date_field}}}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )

    # -----------------------
    # Filtres spécifiques Caméléon
    # -----------------------
    if sources and (("cameleon" in sources) or ("__all__" in sources)):
        cam_filters = filters.get("cameleon", {}) or {}

        faces = cam_filters.get("faces") or []
        entite = cam_filters.get("entite") or []
        localisation = cam_filters.get("localisation") or []
        etat = cam_filters.get("etat") or []

        # Faces (OR logique)
        if faces and "__all__" not in faces:
            must_filters.append(
                {
                    "bool": {
                        "should": [
                            {
                                "match_phrase": {
                                    "metadata.face_list.liste_nom_faces_text": face
                                }
                            }
                            for face in faces
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )

        # Entité
        if entite and "__all__" not in entite:
            must_filters.append(
                {"terms": {"metadata.coeur_unite_creatrice_text.keyword": entite}}
            )

        # Localisation
        if localisation and "__all__" not in localisation:
            must_filters.append(
                {"terms": {"metadata.coeur_localisation_keyword.keyword": localisation}}
            )

        # etat de validité
        if etat and "__all__" not in etat:
            must_filters.append(
                {"terms": {"metadata.face_list.face_field_etat_text.keyword": etat}}
            )

    return must_filters


def semantic_search(query_text, top_k=3, filters=None):
    t0 = time.perf_counter()
    with torch.no_grad():
        query_embedding = model.encode(query_text, convert_to_numpy=True).tolist()
        print("query encoded successfully, len(embedding) : ", len(query_embedding))

    t1 = time.perf_counter()
    encoding_ms = (t1 - t0) * 1000

    print(
        f"[semantic_script_score] Encodage OK (dim={len(query_embedding)}), {encoding_ms:.1f} ms"
    )

    must_filters = build_filter_query(filters)

    query_body = {
        "size": top_k,
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,  # final hits returned
            "num_candidates": 200,  # adjust for quality/speed
            "filter": must_filters,  # your existing filters (bool/filter clauses)
        },
    }

    t2 = time.perf_counter()
    response = es.search(index=EMBEDDING_INDEX, body=query_body)

    t3 = time.perf_counter()
    search_ms = (t3 - t2) * 1000
    total_ms = (t3 - t0) * 1000

    print(
        f"[semantic_script_score] Recherche ES {search_ms:.1f} ms | Total {total_ms:.1f} ms"
    )

    results = []
    for hit in response["hits"]["hits"]:
        metadata = hit["_source"]["metadata"].copy()
        metadata.pop("chunk_id", None)  # supprime "chunk_id" si elle existe

        results.append(
            {
                "original_doc_id": metadata,
                "score": hit["_score"],
                "chunk_content": hit["_source"]["chunk_content"],
                "chunk_id": hit["_source"]["chunk_id"],
                "source": "semantic",
            }
        )

    return results, encoding_ms, search_ms


def lexical_search(query_text, top_k=3, filters=None):
    must_filters = build_filter_query(filters)
    query_body = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [
                    {"multi_match": {"query": query_text, "fields": ["chunk_content"]}}
                ],
                "filter": must_filters,
            }
        },
    }

    t0 = time.perf_counter()
    response = es.search(index=EMBEDDING_INDEX, body=query_body)
    t1 = time.perf_counter()
    search_ms = (t1 - t0) * 1000
    print(f"[lexical] Recherche ES {search_ms:.1f} ms")

    results = []
    for hit in response["hits"]["hits"]:
        metadata = hit["_source"]["metadata"].copy()
        metadata.pop("chunk_id", None)  # supprime "chunk_id" si elle existe

        results.append(
            {
                "original_doc_id": metadata,
                "score": hit["_score"],
                "chunk_content": hit["_source"]["chunk_content"],
                "chunk_id": hit["_source"]["chunk_id"],
                "source": "lexical",
            }
        )

    return results, search_ms


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
        query_body = {"size": 10, "query": {"term": {"sigle": {"value": sigle}}}}
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
            # print("ICI DEBUG : ", doc_id)
            rrf_score = 1 / (k + rank + 1)
            scores[doc_id] += rrf_score
            if (
                doc_id not in seen_docs
                or seen_docs[doc_id]["rrf_score"] < scores[doc_id]
            ):
                seen_docs[doc_id] = {**doc, "rrf_score": scores[doc_id]}

    # Sort by RRF score and take top_k
    return sorted(seen_docs.values(), key=lambda x: x["rrf_score"], reverse=True)[
        :top_k
    ]


def normalize_date(date_str):
    """
    Convertit 'YYYY-MM-DD' → 'dd/MM/yyyy HH:mm:ss'
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%d/%m/%Y 00:00:00")
    except:
        return date_str


def combined_search(query_text, top_k=3, use_dictionary=False, filters=None):
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
    semantic_results, encoding_ms, semantic_ms = semantic_search(
        query_text, top_k, filters
    )
    print("Performing lexical search...")
    lexical_results, lexical_ms = lexical_search(query_text, top_k, filters)
    print("Fusing results with RRF...")
    rrf_results = rrf_fusion(semantic_results, lexical_results, top_k=top_k)

    return {
        f"semantic_top {top_k}": semantic_results,
        f"lexical_top {top_k}": lexical_results,
        f"rrf_top {top_k}": rrf_results,
        "timings": {
            "semantic_ms": semantic_ms,
            "encoding_ms": encoding_ms,
            "lexical_ms": lexical_ms,
        },
    }


def save_to_excel(query, results, use_dictionary, top_k):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"search_results_{query.replace(' ', '_')[:50]}_{timestamp}.xlsx"

    semantic_df = pd.DataFrame(
        [
            {
                "Doc ID": res["original_doc_id"],
                "Score": res["score"],
                "Content": clean_text(res["chunk_content"]),
                "Annotation": None,
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


def try_parse_json_strict(text: str) -> Any | None:
    """Essaye un json.loads direct."""
    try:
        return json.loads(text)
    except Exception:
        return None


def try_parse_json_relaxed(text: str) -> Any | None:
    """
    Tente d'extraire le premier bloc JSON plausible ({...} ou [...]) puis json.loads.
    Utile si le modèle a ajouté du texte avant/après.
    """
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(1).strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


def normalize_llm_json_output(raw_text: str):
    """
    Nettoie et normalise une sortie LLM en JSON strict.
    Ne conserve que : theme, subthemes, subtheme, content, references.
    Retourne [] en cas d'erreur ou de structure invalide.
    """

    try:
        # 1. Suppression des balises ```json ``` ou ``` ```
        cleaned = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()

        # 2. Parsing JSON
        data = json.loads(cleaned)

        # 3. Validation de la structure racine
        if not isinstance(data, list):
            return []

        normalized_output = []

        for theme_block in data:
            if not isinstance(theme_block, dict):
                continue

            theme = theme_block.get("theme")
            subthemes = theme_block.get("subthemes")

            if not isinstance(theme, str) or not isinstance(subthemes, list):
                continue

            clean_subthemes = []

            for st in subthemes:
                if not isinstance(st, dict):
                    continue

                subtheme = st.get("subtheme")
                content = st.get("content")
                references = st.get("references", [])

                # Sécurité sur types
                if not isinstance(subtheme, str) or not isinstance(content, str):
                    continue

                if not isinstance(references, list):
                    references = []

                # Filtrage des références non string
                references = [ref for ref in references if isinstance(ref, str)]

                clean_subthemes.append(
                    {
                        "subtheme": subtheme.strip(),
                        "content": content.strip(),
                        "references": references,
                    }
                )

            if clean_subthemes:
                normalized_output.append(
                    {"theme": theme.strip(), "subthemes": clean_subthemes}
                )

        return normalized_output

    except Exception:
        # En cas de JSON invalide ou d'erreur imprévue
        return []


def normalize_llm_output(data: Any, reference_key: str = "references") -> Any:
    """
    Nettoie la structure retournée :
      - s'assure que data est une liste de thèmes
      - uniformise les 'references' -> liste[str], dédoublonnée
    """
    if data is None:
        return None

    # Convertir en liste si le modèle renvoie un seul dict
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None

    for theme in data:
        subthemes = theme.get("subthemes", [])
        if not isinstance(subthemes, list):
            theme["subthemes"] = []
            continue

        for sub in subthemes:
            refs = sub.get(reference_key, [])
            # Si le modèle a mis la clé, mais pas au bon type, on corrige
            if isinstance(refs, str):
                refs = [refs]
            elif not isinstance(refs, list):
                refs = []

            # Filtre et dédoublonne
            refs = [str(x) for x in refs if x is not None]
            seen = set()
            refs = [r for r in refs if not (r in seen or seen.add(r))]
            sub[reference_key] = refs

            # Optionnel: trim content
            if "content" in sub and isinstance(sub["content"], str):
                sub["content"] = sub["content"].strip()

    return data


def repair_gemini_json(text: str) -> Any | None:
    """
    Portage backend de fix_json_gemini :
    - Convertit les 'references': "a", "b" --> 'references': ["a","b"]
    - Nettoie les échappements \\" et \n
    - Supprime un éventuel padding en tête/queue (ex: textes hors JSON)
    """
    try:
        # Étape 1 : normalisation basique
        s = text.replace('\\"', '"')
        s = re.sub(r"\\n", " ", s)
        s = re.sub(r"\s+", " ", s).strip()

        # Étape 2 : patch des references si au mauvais format
        pattern = r'"references":\s*"([^"]+)"(?:\s*,\s*"([^"]+)")*'

        def repl(m):
            refs = re.findall(r'"([^"]+)"', m.group(0))
            return '"references": [' + ", ".join(f'"{r}"' for r in refs) + "]"

        s = re.sub(pattern, repl, s)

        # Étape 3 : extraire un bloc JSON propre si présence de texte autour
        m = re.search(r"(\[.*\]|\{.*\})", s, flags=re.DOTALL)
        if not m:
            return None
        candidate = m.group(1).strip()

        return json.loads(candidate)
    except Exception:
        return None


def collect_streaming_content(completion) -> str:
    full_content = ""
    for event in completion:
        if event.choices:
            delta = event.choices[0].delta
            if delta and delta.content:
                full_content += delta.content
    return full_content.strip()


def parse_and_normalize_json(content: str, reference_key: str):
    parsed = (
        try_parse_json_strict(content)
        or try_parse_json_relaxed(content)
        or repair_gemini_json(content)
    )
    if parsed is None:
        return None
    return normalize_llm_output(parsed, reference_key)


def remap_unknown_references(
    structured,
    valid_ids: set[str],
    reference_key: str = "references",
    enable_fuzzy=True,
    threshold=0.92,
):
    """
    Valide et nettoie les références:
      - conserve uniquement celles présentes dans valid_ids
      - si enable_fuzzy=True : remappe une ref inconnue vers l'id le plus proche si ratio >= threshold
      - sinon la supprime
    Retourne (structured, report) où report indique ce qui a été remappé/supprimé.
    """
    if isinstance(structured, dict):
        structured = [structured]

    report = {"removed": [], "remapped": []}

    for theme in structured or []:
        subthemes = theme.get("subthemes", []) or []
        for sub in subthemes:
            refs = sub.get(reference_key, []) or []
            new_refs = []
            for r in refs:
                if r in valid_ids:
                    new_refs.append(r)
                else:
                    # tentative fuzzy
                    if enable_fuzzy and isinstance(r, str) and len(r) >= 6:
                        # get_close_matches retourne une liste triée par similarité
                        candidates = difflib.get_close_matches(
                            r, valid_ids, n=1, cutoff=threshold
                        )
                        if candidates:
                            closest = candidates[0]
                            new_refs.append(closest)
                            report["remapped"].append({"from": r, "to": closest})
                            continue
                    # sinon, on supprime
                    report["removed"].append(r)
            # dédoublonnage
            seen = set()
            new_refs = [x for x in new_refs if not (x in seen or seen.add(x))]
            sub[reference_key] = new_refs

    return structured, report


import time
import json


def lexical_search_cpu_timed(query_text, top_k=3, filters=None):
    print(1)
    must_filters = build_filter_query(filters)
    print(2)
    query_body = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [
                    {"multi_match": {"query": query_text, "fields": ["chunk_content"]}}
                ],
                "filter": must_filters,
            }
        },
    }

    t0 = time.perf_counter()
    print(3)
    response = es.search(index=EMBEDDING_INDEX, body=query_body)
    print(4)
    t1 = time.perf_counter()
    search_ms = (t1 - t0) * 1000

    # print(f"[lexical_cpu] Recherche ES {search_ms:.1f} ms")

    results = []
    for hit in response["hits"]["hits"]:
        metadata = hit["_source"].get("metadata", {}).copy()
        metadata.pop("chunk_id", None)
        results.append(
            {
                "original_doc_id": metadata,
                "score": hit["_score"],
                "chunk_content": hit["_source"]["chunk_content"],
                "chunk_id": hit["_source"]["chunk_id"],
                "source": "lexical",
            }
        )

    timings = {"lexical_search_ms": search_ms}
    return results, timings


def semantic_search_cpu_timed(query_text, top_k=3, filters=None):
    import time

    # --- Encodage FULL CPU (et normalisation recommandée) ---
    t0 = time.perf_counter()

    with torch.no_grad():
        query_embedding = model.encode(
            query_text,
            device=device,  # renforce le CPU même si un GPU existe
            convert_to_numpy=True,
            normalize_embeddings=True,  # stabilise cosine/dot_product
        ).tolist()

    t1 = time.perf_counter()
    encoding_ms = (t1 - t0) * 1000
    # print(f"[semantic_cpu] Encodage OK (dim={len(query_embedding)}), {encoding_ms:.1f} ms")

    # --- Recherche ES via script_score (ton implémentation actuelle) ---
    must_filters = build_filter_query(filters)
    query_body = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": query_embedding},
                            },
                        }
                    },
                ],
                "filter": must_filters,
            }
        },
    }

    t2 = time.perf_counter()
    response = es.search(index=EMBEDDING_INDEX, body=query_body)
    t3 = time.perf_counter()
    search_ms = (t3 - t2) * 1000
    total_ms = (t3 - t0) * 1000
    # print(f"[semantic_cpu] ES={search_ms:.1f} ms | Total={total_ms:.1f} ms")

    results = []
    for hit in response["hits"]["hits"]:
        metadata = hit["_source"].get("metadata", {}).copy()
        metadata.pop("chunk_id", None)
        results.append(
            {
                "original_doc_id": metadata,
                "score": hit["_score"],
                "chunk_content": hit["_source"]["chunk_content"],
                "chunk_id": hit["_source"]["chunk_id"],
                "source": "semantic",
            }
        )

    timings = {
        "encoding_ms": encoding_ms,
        "semantic_search_ms": search_ms,
        "semantic_total_ms": total_ms,
    }
    return results, timings


def combined_search_cpu_timed(query_text, top_k=3, use_dictionary=False, filters=None):
    # Option d’expansion (sigles), inchangée
    if use_dictionary:
        known_sigles = get_all_sigles()
        detected_sigles = detect_sigles(query_text, known_sigles)
        if detected_sigles:
            definitions_dict = fetch_definitions(detected_sigles)
            query_text = expand_query(query_text, definitions_dict)
            # print(f"[combined_cpu] Query expandie: {query_text}")

    # --- Semantic (CPU + timers)
    semantic_results, sem_timings = semantic_search_cpu_timed(
        query_text, top_k, filters
    )

    # --- Lexical (CPU + timers)
    lexical_results, lex_timings = lexical_search_cpu_timed(query_text, top_k, filters)

    # --- Fusion RRF (inchangée)
    rrf_results = rrf_fusion(semantic_results, lexical_results, top_k=top_k)

    timings = {
        "encoding_ms": sem_timings["encoding_ms"],
        "semantic_search_ms": sem_timings["semantic_search_ms"],
        "lexical_search_ms": lex_timings["lexical_search_ms"],
        # total = encodage + recherche sémantique + recherche lexicale (sans prendre en compte la fusion)
        "total_ms": sem_timings["encoding_ms"]
        + sem_timings["semantic_search_ms"]
        + lex_timings["lexical_search_ms"],
    }

    return {
        f"semantic_top_{top_k}": semantic_results,
        f"lexical_top_{top_k}": lexical_results,
        f"rrf_top_{top_k}": rrf_results,
        "timings_ms": timings,
    }


# --- benchmark.py ---
from statistics import mean, stdev


def benchmark_queries(
    queries: list[str],
    repeats: int = 5,
    top_k: int = 100,
    use_dictionary: bool = False,
    filters=None,
):
    """
    Exécute combined_search_cpu_timed sur chaque requête 'repeats' fois,
    agrège les temps et imprime un résumé propre.
    """
    report = []

    for q in queries:
        enc_times = []
        sem_times = []
        lex_times = []
        tot_times = []

        print(
            f"\n=== Benchmark sur requête ===\n{q}\n(repeats={repeats}, top_k={top_k})"
        )

        for i in range(repeats):
            out = combined_search_cpu_timed(
                q, top_k=top_k, use_dictionary=use_dictionary, filters=filters
            )
            t = out["timings_ms"]
            enc_times.append(t["encoding_ms"])
            sem_times.append(t["semantic_search_ms"])
            lex_times.append(t["lexical_search_ms"])
            tot_times.append(t["total_ms"])

        def fmt_stats(name, values):
            avg = mean(values)
            sd = stdev(values) if len(values) >= 2 else 0.0
            p50 = sorted(values)[len(values) // 2]
            print(
                f"  {name:<20} avg={avg:8.1f} ms | std={sd:7.1f} ms | p50≈{p50:7.1f} ms"
            )

        fmt_stats("Encodage", enc_times)
        fmt_stats("Recherche sém.", sem_times)
        fmt_stats("Recherche lex.", lex_times)
        fmt_stats("Total (approx.)", tot_times)

        report.append(
            {
                "query": q,
                "repeats": repeats,
                "top_k": top_k,
                "avg_ms": {
                    "encoding": mean(enc_times),
                    "semantic_search": mean(sem_times),
                    "lexical_search": mean(lex_times),
                    "total": mean(tot_times),
                },
                "series_ms": {
                    "encoding": enc_times,
                    "semantic_search": sem_times,
                    "lexical_search": lex_times,
                    "total": tot_times,
                },
            }
        )

    return report


# --- Utilitaires ---
SOURCE_FIELDS = ["chunk_content", "chunk_id", "metadata"]  # réduire la réponse


def lexical_search_timed(query_text, top_k=3, filters=None):
    must_filters = build_filter_query(filters)
    query_body = {
        "size": top_k,
        "_source": SOURCE_FIELDS,
        "query": {
            "bool": {
                "must": [
                    {"multi_match": {"query": query_text, "fields": ["chunk_content"]}}
                ],
                "filter": must_filters,
            }
        },
    }

    t0 = time.perf_counter()
    response = es.search(index=EMBEDDING_INDEX, body=query_body)
    t1 = time.perf_counter()
    search_ms = (t1 - t0) * 1000

    results = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        metadata = src.get("metadata", {}).copy()
        metadata.pop("chunk_id", None)
        results.append(
            {
                "original_doc_id": metadata,
                "score": hit["_score"],
                "chunk_content": src["chunk_content"],
                "chunk_id": src["chunk_id"],
                "source": "lexical",
            }
        )

    timings = {"lexical_search_ms": search_ms}
    return results, timings


def semantic_search_knn_timed(query_text, top_k=3, filters=None, num_candidates=None):
    # --- Encodage sur GPU (si dispo) + normalisation ---
    t0 = time.perf_counter()

    with torch.no_grad():
        query_embedding = model.encode(
            query_text,
            device=device,  # "cuda" si dispo
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

    t1 = time.perf_counter()
    encoding_ms = (t1 - t0) * 1000

    must_filters = build_filter_query(filters)
    if num_candidates is None:
        # règle simple : 50 × top_k (cap 2000)
        num_candidates = min(top_k * 50, 2000)

    # --- kNN natif + filtres ---
    query_body = {
        "size": top_k,
        "_source": SOURCE_FIELDS,
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": num_candidates,
        },
        "query": {  # boolean filters
            "bool": {"filter": must_filters}
        },
    }

    t2 = time.perf_counter()
    response = es.search(index=EMBEDDING_INDEX, body=query_body)
    t3 = time.perf_counter()
    search_ms = (t3 - t2) * 1000
    total_ms = (t3 - t0) * 1000  # encodage + ES

    results = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        metadata = src.get("metadata", {}).copy()
        metadata.pop("chunk_id", None)
        results.append(
            {
                "original_doc_id": metadata,
                "score": hit["_score"],
                "chunk_content": src["chunk_content"],
                "chunk_id": src["chunk_id"],
                "source": "semantic_knn",
            }
        )

    timings = {
        "encoding_ms": encoding_ms,
        "semantic_search_ms": search_ms,
        "semantic_total_ms": total_ms,
    }
    return results, timings


def combined_search_gpu_timed(
    query_text, top_k=3, use_dictionary=False, filters=None, num_candidates=None
):
    # Expansion optionnelle (sigles)
    if use_dictionary:
        known_sigles = get_all_sigles()
        detected_sigles = detect_sigles(query_text, known_sigles)
        if detected_sigles:
            definitions_dict = fetch_definitions(detected_sigles)
            query_text = expand_query(query_text, definitions_dict)

    # --- Sémantique kNN (encodage GPU) ---
    semantic_results, sem_timings = semantic_search_knn_timed(
        query_text, top_k=top_k, filters=filters, num_candidates=num_candidates
    )

    # --- Lexicale ---
    lexical_results, lex_timings = lexical_search_timed(
        query_text, top_k=top_k, filters=filters
    )

    # --- Fusion RRF ---
    rrf_results = rrf_fusion(semantic_results, lexical_results, top_k=top_k)

    timings = {
        "encoding_ms": sem_timings["encoding_ms"],
        "semantic_search_ms": sem_timings["semantic_search_ms"],
        "lexical_search_ms": lex_timings["lexical_search_ms"],
        "total_ms": sem_timings["encoding_ms"]
        + sem_timings["semantic_search_ms"]
        + lex_timings["lexical_search_ms"],
    }

    return {
        f"semantic_top_{top_k}": semantic_results,
        f"lexical_top_{top_k}": lexical_results,
        f"rrf_top_{top_k}": rrf_results,
        "timings_ms": timings,
    }


from statistics import mean, stdev


def percentile(values, q):
    s = sorted(values)
    idx = int(round((q / 100) * (len(s) - 1)))
    return s[idx]


def benchmark_queries_gpu(
    queries: list[str],
    repeats: int = 5,
    top_k: int = 100,
    use_dictionary: bool = False,
    filters=None,
    num_candidates: int | None = 1000,
):
    report = []

    for q in queries:
        enc_times, sem_times, lex_times, tot_times = [], [], [], []

        print(
            f"\n=== Benchmark (GPU encodage, kNN ES) ===\n{q}\n(repeats={repeats}, top_k={top_k})"
        )

        for _ in range(repeats):
            out = combined_search_gpu_timed(
                q,
                top_k=top_k,
                use_dictionary=use_dictionary,
                filters=filters,
                num_candidates=num_candidates,
            )
            t = out["timings_ms"]
            enc_times.append(t["encoding_ms"])
            sem_times.append(t["semantic_search_ms"])
            lex_times.append(t["lexical_search_ms"])
            tot_times.append(t["total_ms"])

        def show_stats(name, vals):
            print(
                f"  {name:<20} avg={mean(vals):8.1f} ms | std={stdev(vals) if len(vals) > 1 else 0.0:7.1f} ms | p50≈{percentile(vals, 50):7.1f} ms | p95≈{percentile(vals, 95):7.1f} ms"
            )

        show_stats("Encodage (GPU)", enc_times)
        show_stats("Recherche sém.", sem_times)
        show_stats("Recherche lex.", lex_times)
        show_stats("Total (approx.)", tot_times)

        report.append(
            {
                "query": q,
                "repeats": repeats,
                "top_k": top_k,
                "avg_ms": {
                    "encoding": mean(enc_times),
                    "semantic_search": mean(sem_times),
                    "lexical_search": mean(lex_times),
                    "total": mean(tot_times),
                },
                "p50_ms": {
                    "encoding": percentile(enc_times, 50),
                    "semantic_search": percentile(sem_times, 50),
                    "lexical_search": percentile(lex_times, 50),
                    "total": percentile(tot_times, 50),
                },
                "p95_ms": {
                    "encoding": percentile(enc_times, 95),
                    "semantic_search": percentile(sem_times, 95),
                    "lexical_search": percentile(lex_times, 95),
                    "total": percentile(tot_times, 95),
                },
            }
        )

    return report


def collect_stream_text(stream) -> str:
    """
    Consume a chat completion stream and return the concatenated text content.

    Expects the stream to yield chunks where:
      - chunk.choices[0].delta.content contains incremental text pieces (may be None on some chunks)
      - chunk.choices[0].finish_reason appears on the final chunk (ignored here)

    Returns:
      The full assistant message as a single string.
    """
    parts = []

    for chunk in stream:
        # Be defensive: SDKs may vary
        if getattr(chunk, "choices", None):
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if delta:
                content_piece = getattr(delta, "content", None)
                if content_piece:
                    parts.append(content_piece)

    return "".join(parts)


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
        save_to_excel(query, results, use_dictionary, top_k)


if __name__ == "__main__":
    main()
