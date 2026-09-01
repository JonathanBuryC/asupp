from pathlib import Path
import sys

# Add the project root directory to sys.path
sys.path.append(str(Path().resolve().parent))
from src.constants.paths import SECRET_PATH
from src.processing.pde_ple import PDE, es


import re
from sentence_transformers import SentenceTransformer
import csv
dic = PDE.s3.download(
    path="s3://bkt-pud-uc/uc202-rex/dictionary1.csv",
    local_file="/opt/app-root/src/uc202-ipn-rex/dictionary1.csv",
)




csv.field_size_limit(1000000)  # Set a higher limit, e.g., 1,000,000


def csv_to_string(file_path):
    with open(file_path, mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        # Read each row and join them into a single string
        csv_string = "\n".join([", ".join(row) for row in csv_reader])
    return csv_string





text = csv_to_string("/opt/app-root/src/uc202-ipn-rex/dictionary1.csv")
# Regex pour découper le texte
pattern = r'(Le sigle|Le code|Le trigramme|Le multigramme)\s*"([^"]+)"\s*signifie\s*(.*?)(?=(Le sigle|Le code|Le trigramme|Le multigramme|$))'

matches = re.finditer(pattern, text, re.DOTALL)

parsed_entries = []
for match in matches:
    sigle = match.group(2).strip()
    definition = match.group(3).strip().rstrip(".")
    parsed_entries.append(
        {
            "sigle": sigle,
            "definition": definition,
            "full_text": f'Le sigle "{sigle}" signifie {definition}.',
        }
    )



model = SentenceTransformer("intfloat/multilingual-e5-large")

for entry in parsed_entries:
    vector = model.encode(entry["definition"]).tolist()
    entry["embedding"] = vector
    es.index(index="lexique_nucleaire", body=entry)

print(len(parsed_entries))