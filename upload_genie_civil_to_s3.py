"""
Upload récursif du dossier réseau
\\atlas.edf.fr\\CO\\dpc-fla3\\Services.004\\J-Lot-Genie-Civil.010
vers S3, en préservant l'arborescence, avec log de reprise
(à exécuter sur un poste Windows ayant accès au partage réseau).
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.processing.pde_ple import PDE

# --- Config à adapter ---
LOCAL_ROOT = Path(r"\\atlas.edf.fr\CO\dpc-fla3\Services.004\J-Lot-Genie-Civil.010")
S3_BASE_PATH = "s3://bkt-pud-uc/uc202-rex/J-Lot-Genie-Civil.010"
UPLOAD_LOG_PATH = Path(__file__).parent / "uploaded_log_genie_civil.txt"
MAX_WORKERS = 32  # commence prudent (réseau SMB), monte si stable


def already_uploaded_set() -> set[str]:
    if UPLOAD_LOG_PATH.exists():
        with open(UPLOAD_LOG_PATH, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f)
    return set()


def list_local_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def upload_one(file_path: Path):
    try:
        rel_path = file_path.relative_to(LOCAL_ROOT).as_posix()
        s3_target_path = f"{S3_BASE_PATH}/{rel_path}"
        PDE.s3.upload(local_file=str(file_path), path=s3_target_path)
        with open(UPLOAD_LOG_PATH, "a", encoding="utf-8") as log:
            log.write(str(file_path) + "\n")
        return f"OK: {file_path}"
    except Exception as e:
        return f"ERROR with {file_path}: {e}"


def main():
    done = already_uploaded_set()
    all_files = [p for p in list_local_files(LOCAL_ROOT) if str(p) not in done]
    print(f"Fichiers restant à uploader : {len(all_files)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(upload_one, fp): fp for fp in all_files}
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
