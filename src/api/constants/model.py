#MODEL_NAME = ("/opt/app-root/src/uc202-ipn-rex/src/models/intfloat_multilingual_e5_large_instruct")
MODEL_NAME = "/opt/app-root/src/uc202-ipn-rex/src/models/models/test"
MODEL_S3_PATH = "s3://bkt-pud-uc/uc202-rex/models/model_e5_finetuned.pickle"

#EMBEDDING_INDEX = "uc202-rex-embeddings-e5"
#EMBEDDING_INDEX = "uc202-rex-embeddings"
EMBEDDING_INDEX = "uc202-rex-embeddings-e5-trained-v2"


EXTENSION_FAMILIES = {
    "excel": [".xlsx", ".xls", ".csv", ".XLSX", ".XLS"],
    "word": [".doc", ".docx", ".txt", ".DOC", ".DOCX"],
    "powerpoint": [".ppt", ".pptx", ".PPT", ".PPTX"],
    "pdf": [".pdf", ".PDF"],
    "rtf": [".rtf"],
}
