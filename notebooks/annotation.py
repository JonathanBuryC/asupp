import argparse
from pathlib import Path
import sys
import os
import random
import pandas as pd
from datetime import datetime

# Add the project root directory to sys.path
sys.path.append(str(Path().resolve().parent))

from query import combined_search
from src.processing.utils import clean_text


import random
from datetime import datetime
import pandas as pd
from openpyxl import Workbook



import math



def generate_annotation_excel_with_overlap(
    top_k,
    num_annotators,
    query_txt_path,
    use_dictionary=False,
    cross_annotation=True,
    cross_annotation_percentage=0.1,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"annotation_atelier_{timestamp}.xlsx"

    # Read queries
    with open(query_txt_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    # Annotator -> list of all their assignments
    annotator_data = {f"Annotator_{i + 1}": [] for i in range(num_annotators)}

    print("Running combined search for all questions...\n")
    for question in questions:
        print(f"Processing: {question}")
        results = combined_search(question, top_k, use_dictionary)
        rrf_results = results[f"rrf_top {top_k}"]

        # Clean and standardize each answer
        cleaned = [
            {
                "Question": question,
                "Doc ID": res["original_doc_id"],
                "RRF Score": res["rrf_score"],
                "Content": clean_text(res["chunk_content"]),
                "Annotation": None,
            }
            for res in rrf_results
        ]

        # Temporary per-query distribution
        per_query = {f"Annotator_{i + 1}": [] for i in range(num_annotators)}

        # Step 1: Round-robin assignment
        for idx, item in enumerate(cleaned):
            annotator_index = idx % num_annotators
            annotator_key = f"Annotator_{annotator_index + 1}"
            per_query[annotator_key].append(item)

        # Step 2: Optional cross-annotation
        if cross_annotation:
            for i in range(num_annotators):
                current = f"Annotator_{i + 1}"
                prev = f"Annotator_{(i - 1) % num_annotators + 1}"  # wrap around
                prev_items = per_query[prev]
                overlap_count = max(
                    1, math.floor(len(prev_items) * cross_annotation_percentage)
                )
                overlap_samples = random.sample(
                    prev_items, min(overlap_count, len(prev_items))
                )
                per_query[current].extend(overlap_samples)

        # Step 3: Save this query's assignments globally
        for annotator in annotator_data:
            annotator_data[annotator].extend(per_query[annotator])

    # Export Excel with one sheet per annotator
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for annotator, rows in annotator_data.items():
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=annotator, index=False)

    print(f"\nAnnotation Excel file generated: {filename}")


import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, help="Number of top documents per query")
    parser.add_argument("--num_annotators", type=int, help="Number of annotators")
    parser.add_argument(
        "--query_txt_path", type=str, help="Path to text file with one query per line"
    )
    parser.add_argument(
        "--use_dictionary", action="store_true", help="Use dictionary expansion"
    )
    parser.add_argument(
        "--cross_annotation",
        action="store_true",
        default=True,
        help="Enable overlapping annotation between annotators",
    )
    parser.add_argument(
        "--cross_annotation_percentage",
        type=float,
        default=0.1,
        help="Percentage of previous annotator's answers to be duplicated (default: 0.05)",
    )

    args = parser.parse_args()

    # Prompt interactively if required arguments are missing
    if args.top_k is None:
        args.top_k = int(input("Enter top_k (e.g., 100): "))

    if args.num_annotators is None:
        args.num_annotators = int(input("Enter number of annotators (e.g., 5): "))

    if args.query_txt_path is None:
        args.query_txt_path = input(
            "Enter path to query text file (e.g., queries.txt): "
        )

    return args


# Usage
if __name__ == "__main__":
    args = get_args()

    print(f"Top K: {args.top_k}")
    print(f"Number of Annotators: {args.num_annotators}")
    print(f"Query File Path: {args.query_txt_path}")
    print(f"Use Dictionary: {args.use_dictionary}")
    print(f"Cross Annotation Enabled: {args.cross_annotation}")
    print(f"Cross Annotation Percentage: {args.cross_annotation_percentage}")

    generate_annotation_excel_with_overlap(
        top_k=args.top_k,
        num_annotators=args.num_annotators,
        query_txt_path=args.query_txt_path,
        use_dictionary=args.use_dictionary,
        cross_annotation=args.cross_annotation,
        cross_annotation_percentage=args.cross_annotation_percentage,
    )