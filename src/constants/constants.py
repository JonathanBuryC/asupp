USEFUL_EXTENSIONS = [
    ".pdf",
    ".doc",
    ".docx",
    ".odt",
    ".txt",
    ".rtf",
    ".odt",
    ".ppt",
    ".pptx",
    ".pps",
    ".ppsx",
    ".odp",
    ".xls",
    ".xlsx",
    ".xlsb",
    ".ods",
    ".csv",
]
EXAMPLES_PATH="/opt/app-root/src/uc202-ipn-rex/example_files/"
S3_BUCKET_PATH="s3://bkt-pud-uc/uc202-rex/"
S3_DOCS_PATH=S3_BUCKET_PATH+"FA3_useful_files_final/"
S3_RAW_DOCS_PATH = S3_BUCKET_PATH + "FA3_raw_files_txt/"

SERVICES_PATH=S3_DOCS_PATH+"Services.004/"
ENVIRONNEMENT_PATH = SERVICES_PATH  + "G-Environnement.007/" 
ENVIRONNEMENT_RAW_PATH = S3_RAW_DOCS_PATH + "G-Environnement.007/"
LOT_INGENIERIE_LOCALE_PATH = SERVICES_PATH + "W-Lot-Ingenierie-Locale.023/"
TEMP_DIR="/opt/app-root/src/uc202-ipn-rex/temp_dir"
SMALL_ANONYMIZER_MODEL="sm_generic"
DEFAULT_ANONYMIZER_MODEL = "md_generic"
DEFAULT_ENTITIES_TO_DEACTIVATE = {
    "__AMOUNT__",
    "__CONTRACT__",
    "__DATE__",
    "__JOB_POSITION__",
    "__LOCALISATION__",
    "__NUMBER__",
    "__ORG_DISTRIB__",
    "__ORG_EDF__",
    "__ORG_ENERGY__",
    "__ORG__",
    "__REFERENCE__",
    "__URL__",
    "__SIREN__",
    "__SIRET__",
    "__PATH__",
    "__NUM_BUSINESS__",
    "__PDL__",
    "__PCE__",
    "__INITIALS_CC__",
    "__FACTURATION_ACCOUNT__",
    "__INFO_BANK__",
    "__ELEC_METER__",
    "__EDF_PHONE_NUMBER_3004__",
    "__EDF_PHONE_NUMBER_321515__",
    "__COMMERCIAL_ACCOUNT__",
    "__CODE__",
    "__BP__",
}
DEFAULT_ENTITIES_TO_ACTIVATE = {
    "__PERSON__",
    "__PHONE_NUMBER__",
    "__NNI__", #voir si possible de ne pas anonymiser vu que c en interne, pareil pour les mail @edf.fr
    "__EMAIL__",
    "__DIGITAL_ID__"   

}