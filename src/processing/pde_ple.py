"""Module aiming at having an instance of PandasDataEngine and PolarsDataEngine usable in the constants"""
from daptools import (
    AWSSettings, 
    PandasDataEngine,
    PolarsDataEngine,
)
from ..constants.paths import SECRET_PATH
import json
from elasticsearch import Elasticsearch

PDE = PandasDataEngine()
PLE = PolarsDataEngine()

aws_settings = AWSSettings(config_file_path=SECRET_PATH)


PDE.connect_to_s3(settings=aws_settings, name="s3")
PLE.connect_to_s3(settings=aws_settings, name="s3")

if SECRET_PATH.exists():
    with open(SECRET_PATH, "r") as f:
        secrets = json.load(f)
    print("Secrets chargés avec succès !")
else:
    raise FileNotFoundError(f"Le fichier de secret {SECRET_PATH} n'existe pas.")

# Access the ELASTIC_USER variable
ENDPOINT = secrets["ELASTIC"]["ELASTIC_HOSTS"].split(",")[0]
USERNAME_elastic = secrets["ELASTIC"]["ELASTIC_USER"]
MDP_elastic = secrets["ELASTIC"]["ELASTIC_PASSWORD"]

es = Elasticsearch(
    hosts=ENDPOINT, verify_certs=False, basic_auth=(USERNAME_elastic, MDP_elastic),timeout=120
)

