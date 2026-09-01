# uc202-rex-ipn

L’analyse du Retour d’Expérience (REX) est une démarche fondamentale pour l’amélioration continue des projets d’ingénierie et de construction chez EDF. Cependant, le processus actuel présente plusieurs défis. Les équipes sont souvent contraintes d’examiner manuellement un volume important de constats, dont une partie est peu pertinente pour des recherches spécifiques. Les méthodes de recherche par mots-clés sont imprécises et chronophages.

De plus, l’exploitation des sources de REX non structurées, telles que les documents du projet FA3, reste difficile et sous‑utilisée. Cette situation conduit à des analyses disparates et à une perte potentielle d’informations précieuses.

---

## Interlocuteurs

### Auteurs du projet

- **Rostom Zitoun** (Data Scientist) : <rostom.zitoun@edf.fr>  
- **Daniel Hobbs** (Chef de Projet) : <daniel.hobbs@edf.fr>

### Client

- **Nicolas Lemoine** : <nicolas.lemoine@edf.fr>

---

## Installation (local)

Cette section décrit les étapes nécessaires à l’installation du projet en local.

### Pré‑requis

- Avoir installé **UV**

### Étapes d’installation

#### Linux

```bash
curl -sSL https://get.astral.sh/uv | bash
```

#### Windows

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

+ Cloner ce repo git:

```bash
git clone https://si-devops-gitlab.edf.fr/pud-usine/use-cases/uc202-ipn-rex.git
```

+ Se placer à l'interieur du répertoire:

```bash
cd uc202-ipn-rex
```

+ Pour créer un environnement virtuel et installer les dépendances, utilisez :

```bash
uv venv 
uv sync
```

+ Pour activer l'environnement virtuel créé :

```bash
source .venv/bin/activate 
```

+ Pour executer le main :

```bash
bash run.sh
```
## Données sur S3:

Le bucket bkt-pud-uc/uc202-rex contient les données utilisées lors du UC : 
+ FA3_useful_files_final : contient les fichiers Environnement utiles (extensions exploitables)
+ FA3_raw_files_txt : contient l'extraction brute des fichiers Environnement en des .txt
+ ZZZ - Test REX FLA3 : contient les fichiers de la Dir. technique FA3 utiles (extensions exploitables)
+ ZZZ - Test REX FLA3_raw_files_txt : contient l'extraction brute des fichiers de la Dir. technique FA3 en des .txt
+ pieces_jointes : contient les pieces jointes des constats Caméléon.
+ pieces_jointes_raw_file_txt : contient l'extraction brute des pièces jointes en des .txt
+ models : contient les modèles testés et le modèle finetuné retenu.
+ dictionary.csv : un dictionnaire contenant > 10.000 bigrammes/trigrammes/sigles et leur définitions.
+ FA3_useful_files.csv : un fichier contenant les métadonnées de tous les fichiers du réseau partagé FA3

## Données sur Elastic:

Les indexs Elastic contenant les données utilisées pendant le UC :
+ uc202-rex-camaleon : les constats caméléon du périmètre de UC + les métadonnées pertinentes + création du texte requêtable en concaténant les attributs pertinents.
+ uc202-pj : les pièces jointes ratachés aux constats
+ uc202-rex-environnement : les documents du répertoire Environnement
+ uc202-rex-gsimon : les documents de la Dir. technique FA3
+ uc202-rex-chunks: les 4 indexs mentionnées précédemment découpés en chunks
+ uc202-rex-embeddings-e5-trained-v2 : l'index précedent + les embeddings avec le modèle E5-multilingual finetuné
+ uc202-rex-chunks-elecbert-256 : les 4 index précédemment mentionnées tokenizé avec ElecBert.
