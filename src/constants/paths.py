from pathlib import Path

from dotenv import load_dotenv
import os

_CURRENT_FILE_PATH = Path(__file__)
PATH_TO_DOTENV = _CURRENT_FILE_PATH.parents[2] / "config" / "local.env"
# * clunky : this condition (non-existence of .env) is true if the back-end part of the code is wheeled --> this is prod environment
if not PATH_TO_DOTENV.exists():
    IS_PROD_ENVIRONMENT = True
    APP_HOME_PATH = Path("/opt/app-root/src")
    PROJECT_NAME = _CURRENT_FILE_PATH.parents[1].parts[-1].replace("_", "-")
    PATH_TO_DOTENV = APP_HOME_PATH / PROJECT_NAME / "config" / ".env"
else:
    # * here prod conditions are false
    IS_PROD_ENVIRONMENT = False
    APP_HOME_PATH = _CURRENT_FILE_PATH.parents[3]
    PROJECT_NAME = _CURRENT_FILE_PATH.parents[2].parts[-1]


load_dotenv(dotenv_path=str(PATH_TO_DOTENV))

LOCAL_SECRETS_PATH = "./config/secret.json" 
LOCAL_SECRETS_PATH = "/opt/app-root/src/uc202-ipn-rex/config/secret.json"
CONTAINER_SECRETS_PATH = "/opt/app-root/src/config/secret.json"  # pour Docker simple
DEPLOYMENT_SECRETS_PATH = "/etc/secret-volume/secret.json"

def get_secret_file_path() -> Path:
    """Détermine automatiquement le bon chemin du secret.json."""
    if os.path.exists(DEPLOYMENT_SECRETS_PATH):
        print(" Secrets lus depuis le déploiement (OpenShift).")
        return Path(DEPLOYMENT_SECRETS_PATH)
    elif os.path.exists(CONTAINER_SECRETS_PATH):
        print("Secrets lus depuis le conteneur Docker.")
        return Path(CONTAINER_SECRETS_PATH)
    else:
        print("Secrets lus depuis le dossier local.")
        return Path(LOCAL_SECRETS_PATH)

SECRET_PATH = get_secret_file_path()

# * Path relative to local container, in order to be able to write
PACKAGE_NAME = PROJECT_NAME.replace("-", "_")
PROJECT_PATH = APP_HOME_PATH / PROJECT_NAME
SRC_PATH = PROJECT_PATH / "src"
TESTS_PATH = PROJECT_PATH / "tests"
PACKAGE_PATH = SRC_PATH / PACKAGE_NAME
CONSTANTS_PATH = PACKAGE_PATH / "constants"

# * inputs
INPUTS_PATH = PROJECT_PATH / "inputs"
INPUTS_RAW_PATH = INPUTS_PATH / "raw"
INPUTS_PROCESSED_PATH = INPUTS_PATH / "processed"
INPUTS_EXTERNAL_PATH = INPUTS_PATH / "external"

# * docs
DOCS_PATH = PROJECT_PATH / "docs"

# * tests
TESTS_PATH = PROJECT_PATH / "tests"

# * target
TARGET_PATH = PROJECT_PATH / "target"
TARGET_RESULTS_PATH = TARGET_PATH / "results"


def _create_s3_path(folder: str, file: str) -> str:
    return r"/".join([folder, file]).replace(r"//", r"/")


# * config s3, postgres, oracle
CONFIGURATION_REPOSITORY_PATH = PROJECT_PATH / "config"
#SECRET_PATH = CONFIGURATION_REPOSITORY_PATH / "secret.json"
CONFIG_PATH = CONFIGURATION_REPOSITORY_PATH/"config.json"


URL_ONE_API= "https://oneapi.edf.fr/dteo/it/It_EdfPortailMultiIAG_OpenAI_Bearer/1.0/v2/workspaces/HcA-puQ/webhooks/v1"
#URL_ONE_API= "https://oneapi-run-dteo-dev.edf.fr/it/It_EdfPortailMultiIAG_OpenAI_Bearer/1.0/v2/workspaces/HcA-puQ/webhooks/v1"
#URL_ONE_API= "https://oneapi-run-dteo-dev.edf.fr/it/It_EdfPortailMultiIAG_OpenAI_Bearer/1.0/v2/workspaces/HcA-puQ/webhooks/v1/chat/completions"
