# Databricks notebook source
# MAGIC %md
# MAGIC # 🥉 Étape 2 — Ingestion Bronze : AutoLoader + Extraction texte PDF
# MAGIC
# MAGIC **Projet : Plateforme RAG à l'échelle**
# MAGIC
# MAGIC Ce notebook couvre :
# MAGIC - Installation de `pdfplumber` pour l'extraction de texte
# MAGIC - Lecture des PDFs depuis le Volume Unity Catalog
# MAGIC - Extraction du texte page par page
# MAGIC - Stockage dans la **Delta Table Bronze** (`bronze.pdf_raw`)
# MAGIC - Exploration de la table avec quelques requêtes SQL

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Paramètres globaux
# MAGIC
# MAGIC > Doit être cohérent avec le notebook 01

# COMMAND ----------

CATALOG_NAME  = "rag_project"
SCHEMA_BRONZE = "bronze"
VOLUME_NAME   = "raw_pdfs"
TABLE_BRONZE  = f"{CATALOG_NAME}.{SCHEMA_BRONZE}.pdf_raw"
VOLUME_PATH   = f"/Volumes/{CATALOG_NAME}/{SCHEMA_BRONZE}/{VOLUME_NAME}"

# Dossier de checkpoint pour AutoLoader (gère l'état du streaming)
CHECKPOINT_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_BRONZE}/checkpoints/pdf_raw"

print(f"""
📂 Volume source  : {VOLUME_PATH}
📋 Table Bronze   : {TABLE_BRONZE}
🔖 Checkpoint     : {CHECKPOINT_PATH}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Installation des dépendances
# MAGIC
# MAGIC `pdfplumber` est la lib Python la plus robuste pour extraire du texte
# MAGIC depuis des PDFs (meilleur que PyPDF2 sur les PDFs complexes avec tableaux).

# COMMAND ----------

# MAGIC %pip install pdfplumber --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Re-déclare les paramètres après restartPython
CATALOG_NAME  = "rag_project"
SCHEMA_BRONZE = "bronze"
VOLUME_NAME   = "raw_pdfs"
TABLE_BRONZE  = f"{CATALOG_NAME}.{SCHEMA_BRONZE}.pdf_raw"
VOLUME_PATH   = f"/Volumes/{CATALOG_NAME}/{SCHEMA_BRONZE}/{VOLUME_NAME}"
CHECKPOINT_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_BRONZE}/checkpoints/pdf_raw"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Vérification des PDFs dans le Volume

# COMMAND ----------

files = dbutils.fs.ls(VOLUME_PATH)
print(f"📂 {len(files)} fichier(s) détecté(s) dans le Volume :\n")
for f in files:
    print(f"  📄 {f.name:<50} {f.size/1024:.1f} KB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fonction d'extraction de texte PDF
# MAGIC
# MAGIC On définit une fonction qui :
# MAGIC - Prend le chemin d'un PDF en entrée
# MAGIC - Retourne une liste de dicts (un par page) avec le texte extrait
# MAGIC
# MAGIC > 💡 **Pourquoi page par page ?**
# MAGIC > Pour le RAG, on veut garder la granularité de la source.
# MAGIC > Cela permet de citer la page exacte dans les réponses du chatbot.

# COMMAND ----------

import pdfplumber
import os
from datetime import datetime
from typing import List, Dict, Any


def extract_text_from_pdf(volume_path: str) -> List[Dict[str, Any]]:
    """
    Extrait le texte d'un PDF page par page.
    
    Args:
        volume_path: Chemin du PDF dans le Volume Unity Catalog
                     Ex: /Volumes/rag_project/bronze/raw_pdfs/rapport.pdf
    
    Returns:
        Liste de dicts, un par page :
        {
            "file_name": str,
            "file_path": str,
            "page_number": int,
            "raw_text": str,
            "char_count": int,
            "ingestion_date": datetime,
            "file_size_bytes": int
        }
    """
    # Conversion du chemin Volume → chemin local accessible par Python
    # Les Volumes sont montés sous /Volumes/ directement sur le driver node
    local_path = volume_path  # sur Databricks, /Volumes/ est accessible nativement

    results = []
    file_name = os.path.basename(local_path)
    file_size = os.path.getsize(local_path)
    ingestion_date = datetime.now()

    try:
        with pdfplumber.open(local_path) as pdf:
            print(f"  📄 {file_name} — {len(pdf.pages)} page(s) détectée(s)")
            
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""  # None → "" si page sans texte
                text = text.strip()
                
                results.append({
                    "file_name":        file_name,
                    "file_path":        volume_path,
                    "page_number":      page_num,
                    "raw_text":         text,
                    "char_count":       len(text),
                    "ingestion_date":   ingestion_date,
                    "file_size_bytes":  file_size
                })
    except Exception as e:
        print(f"  ❌ Erreur sur {file_name} : {e}")

    return results


# Test rapide sur le premier fichier pour valider
print("🧪 Test d'extraction sur le premier PDF :\n")
first_pdf = files[0]
# Chemin local pour pdfplumber (sans le préfixe dbfs:)
local_pdf_path = first_pdf.path.replace("dbfs:", "")

pages = extract_text_from_pdf(local_pdf_path)
print(f"\n✅ {len(pages)} page(s) extraite(s)")
print(f"\n--- Aperçu page 1 (200 premiers caractères) ---")
print(pages[0]["raw_text"][:200] if pages else "Aucun texte extrait")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Extraction de tous les PDFs du Volume
# MAGIC
# MAGIC On itère sur tous les fichiers `.pdf` du Volume et on consolide
# MAGIC les résultats dans une liste Python avant de créer le DataFrame Spark.

# COMMAND ----------

all_pages = []

print("🔄 Extraction de tous les PDFs...\n")
for f in files:
    if f.name.lower().endswith(".pdf"):
        local_path = f.path.replace("dbfs:", "")
        pages = extract_text_from_pdf(local_path)
        all_pages.extend(pages)
        print(f"  → {len(pages)} page(s) extraite(s)\n")

print(f"✅ Total : {len(all_pages)} pages extraites depuis {len(files)} PDF(s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Création du DataFrame Spark et écriture en Delta Table
# MAGIC
# MAGIC On convertit la liste Python en **DataFrame Spark** puis on écrit
# MAGIC en **Delta Lake** avec le mode `overwrite` (idempotent pour les tests).
# MAGIC
# MAGIC > 💡 En production, on utiliserait `append` + déduplication,
# MAGIC > ou AutoLoader en streaming pour n'ingérer que les nouveaux fichiers.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    LongType, TimestampType
)

spark = SparkSession.builder.getOrCreate()

# Schéma explicite — bonne pratique en production
schema = StructType([
    StructField("file_name",        StringType(),    nullable=False),
    StructField("file_path",        StringType(),    nullable=False),
    StructField("page_number",      IntegerType(),   nullable=False),
    StructField("raw_text",         StringType(),    nullable=True),
    StructField("char_count",       IntegerType(),   nullable=True),
    StructField("ingestion_date",   TimestampType(), nullable=False),
    StructField("file_size_bytes",  LongType(),      nullable=True),
])

# Création du DataFrame
df_bronze = spark.createDataFrame(all_pages, schema=schema)

print(f"📊 DataFrame créé : {df_bronze.count()} lignes, {len(df_bronze.columns)} colonnes")
df_bronze.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Écriture en Delta Table Bronze
# MAGIC
# MAGIC On utilise `spark.sql("USE CATALOG ...")` pour s'assurer
# MAGIC qu'on écrit dans le bon catalog Unity Catalog.

# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG_NAME}")

(
    df_bronze
    .write
    .format("delta")
    .mode("overwrite")                    # Idempotent : OK pour les tests
    .option("overwriteSchema", "true")    # Permet de modifier le schéma si besoin
    .saveAsTable(TABLE_BRONZE)
)

print(f"✅ Table Bronze écrite : {TABLE_BRONZE}")
print(f"   → {df_bronze.count()} lignes ingérées")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Validation & Exploration de la table Bronze

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a. Aperçu des données

# COMMAND ----------

print("📋 Aperçu de la table Bronze :\n")
spark.table(TABLE_BRONZE).show(10, truncate=80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6b. Statistiques par fichier

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     file_name,
# MAGIC     COUNT(*)            AS nb_pages,
# MAGIC     SUM(char_count)     AS total_chars,
# MAGIC     AVG(char_count)     AS avg_chars_per_page,
# MAGIC     MIN(char_count)     AS min_chars,
# MAGIC     MAX(char_count)     AS max_chars,
# MAGIC     MAX(ingestion_date) AS ingested_at
# MAGIC FROM rag_project.bronze.pdf_raw
# MAGIC GROUP BY file_name
# MAGIC ORDER BY file_name

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6c. Pages avec peu ou pas de texte
# MAGIC
# MAGIC > Ces pages sont souvent des images, graphiques ou pages de garde.
# MAGIC > On les gardera en Bronze mais on les filtrera en Silver.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT file_name, page_number, char_count, raw_text
# MAGIC FROM rag_project.bronze.pdf_raw
# MAGIC WHERE char_count < 100
# MAGIC ORDER BY char_count ASC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6d. Vérification Delta Lake — Time Travel & Historique

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Historique des écritures sur la table (fonctionnalité Delta Lake)
# MAGIC DESCRIBE HISTORY rag_project.bronze.pdf_raw

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. (Bonus) AutoLoader — Pour la production
# MAGIC
# MAGIC > Le code ci-dessous montre comment on ferait en **production**
# MAGIC > avec AutoLoader pour ingérer automatiquement les nouveaux PDFs
# MAGIC > dès qu'ils arrivent dans le Volume, sans re-traiter les anciens.
# MAGIC >
# MAGIC > ⚠️ **Ne pas exécuter maintenant** — nécessite un cluster en mode streaming.
# MAGIC > C'est fourni à titre pédagogique pour l'entretien.

# COMMAND ----------

# ── PRODUCTION PATTERN (ne pas exécuter sur le trial) ──────────────────────
#
# from pyspark.sql.functions import udf, col
# from pyspark.sql.types import ArrayType, StructType, ...
#
# # UDF qui wrap notre fonction d'extraction
# extract_udf = udf(extract_text_from_pdf, ArrayType(schema))
#
# # AutoLoader : surveille le Volume et ingère les nouveaux fichiers
# df_stream = (
#     spark.readStream
#         .format("cloudFiles")
#         .option("cloudFiles.format", "binaryFile")   # Lit les fichiers binaires (PDF)
#         .option("cloudFiles.schemaLocation", CHECKPOINT_PATH + "/schema")
#         .load(VOLUME_PATH)
# )
#
# # Pour chaque micro-batch, extraction du texte et écriture en Delta
# (
#     df_stream
#     .writeStream
#     .format("delta")
#     .outputMode("append")
#     .option("checkpointLocation", CHECKPOINT_PATH)
#     .trigger(availableNow=True)   # One-shot : traite tout puis s'arrête
#     .toTable(TABLE_BRONZE)
# )
#
# ── Pourquoi AutoLoader est puissant ? ─────────────────────────────────────
# - Détecte automatiquement les nouveaux fichiers (S3/ADLS/GCS/Volume)
# - Gère le checkpoint : ne re-traite jamais un fichier déjà ingéré
# - Scale automatiquement sur des millions de fichiers
# - Compatible avec le mode "Trigger.AvailableNow" (batch incrémental)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Récapitulatif

# COMMAND ----------

count = spark.table(TABLE_BRONZE).count()
files_count = spark.table(TABLE_BRONZE).select("file_name").distinct().count()

print("=" * 60)
print("✅ ÉTAPE 2 TERMINÉE — Récapitulatif")
print("=" * 60)
print(f"""
🥉 Table Bronze : {TABLE_BRONZE}
   → {files_count} fichier(s) PDF ingéré(s)
   → {count} page(s) au total

✅ Ce qui a été démontré :
   - Extraction texte PDF avec pdfplumber
   - Schéma explicite Spark (bonne pratique prod)
   - Écriture Delta Lake avec Unity Catalog
   - Delta Time Travel (DESCRIBE HISTORY)
   - Pattern AutoLoader pour la production

🔜 Prochaine étape :
   Notebook 03_silver_text_cleaning.py
   → Nettoyage du texte (suppression entêtes/pieds de page)
   → Détection des pages vides
   → Chunking du texte pour les embeddings
""")
print("=" * 60)
