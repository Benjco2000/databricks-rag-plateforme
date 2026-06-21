# Databricks notebook source
# MAGIC %md
# MAGIC # 🥈 Étape 3 — Silver : Nettoyage du texte & Chunking
# MAGIC
# MAGIC **Projet : Plateforme RAG à l'échelle**
# MAGIC
# MAGIC Ce notebook couvre :
# MAGIC - Lecture de la Delta Table Bronze (`bronze.pdf_raw`)
# MAGIC - Nettoyage du texte (caractères parasites, espaces, lignes vides)
# MAGIC - Filtrage des pages sans contenu utile
# MAGIC - Chunking du texte avec `RecursiveCharacterTextSplitter` (LangChain)
# MAGIC - Écriture dans la **Delta Table Silver** (`silver.pdf_chunks`)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Paramètres globaux

# COMMAND ----------

CATALOG_NAME  = "rag_project"
SCHEMA_BRONZE = "bronze"
SCHEMA_SILVER = "silver"
TABLE_BRONZE  = f"{CATALOG_NAME}.{SCHEMA_BRONZE}.pdf_raw"
TABLE_SILVER  = f"{CATALOG_NAME}.{SCHEMA_SILVER}.pdf_chunks"

# Paramètres de chunking
CHUNK_SIZE    = 500   # Taille max d'un chunk en caractères
CHUNK_OVERLAP = 100   # Overlap entre chunks consécutifs (évite de couper le contexte)
MIN_CHARS     = 50    # Pages avec moins de 50 chars → considérées vides

print(f"""
📋 Table source  : {TABLE_BRONZE}
📋 Table cible   : {TABLE_SILVER}

⚙️  Paramètres chunking :
   chunk_size    = {CHUNK_SIZE} caractères
   chunk_overlap = {CHUNK_OVERLAP} caractères
   min_chars     = {MIN_CHARS} (filtre pages vides)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Installation des dépendances

# COMMAND ----------

# MAGIC %pip install langchain langchain-text-splitters --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Re-déclare les paramètres après restartPython
CATALOG_NAME  = "rag_project"
SCHEMA_BRONZE = "bronze"
SCHEMA_SILVER = "silver"
TABLE_BRONZE  = f"{CATALOG_NAME}.{SCHEMA_BRONZE}.pdf_raw"
TABLE_SILVER  = f"{CATALOG_NAME}.{SCHEMA_SILVER}.pdf_chunks"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
MIN_CHARS     = 50

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Lecture de la table Bronze

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, trim

spark = SparkSession.builder.getOrCreate()
spark.sql(f"USE CATALOG {CATALOG_NAME}")

df_bronze = spark.table(TABLE_BRONZE)

print(f"📊 Table Bronze chargée : {df_bronze.count()} lignes\n")
df_bronze.show(5, truncate=60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Nettoyage du texte
# MAGIC
# MAGIC On applique une série de transformations pour nettoyer le texte brut :
# MAGIC - Suppression des caractères de contrôle (\\x00, \\t excessifs, etc.)
# MAGIC - Normalisation des espaces multiples et sauts de ligne
# MAGIC - Strip des espaces en début/fin
# MAGIC
# MAGIC > 💡 En production sur des millions de docs, on enrichirait ce nettoyage
# MAGIC > avec des règles métier spécifiques (ex: supprimer les en-têtes
# MAGIC > répétitifs détectés par pattern matching).

# COMMAND ----------

import re


def clean_text(text: str) -> str:
    """
    Nettoie le texte extrait d'un PDF.
    
    Transformations appliquées :
    1. Supprime les caractères de contrôle non imprimables
    2. Remplace les tabulations par des espaces
    3. Réduit les sauts de ligne multiples (>2) à 2 max
    4. Réduit les espaces multiples à un seul
    5. Strip global
    """
    if not text:
        return ""
    
    # 1. Supprime caractères de contrôle (sauf \n et \t)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", text)
    
    # 2. Remplace les tabulations par un espace
    text = text.replace("\t", " ")
    
    # 3. Réduit les sauts de ligne multiples
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # 4. Réduit les espaces multiples
    text = re.sub(r" {2,}", " ", text)
    
    # 5. Strip
    text = text.strip()
    
    return text


# Test de la fonction sur un exemple
sample_text = "  Rapport   Annuel\t\t2023\n\n\n\nChiffre d'affaires : 4,2 Mds€  \n\n\n  "
cleaned = clean_text(sample_text)
print("Avant :", repr(sample_text))
print("Après :", repr(cleaned))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Application du nettoyage + Filtrage des pages vides

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Enregistrement de la fonction de nettoyage comme UDF Spark
clean_text_udf = udf(clean_text, StringType())

# Application + calcul de la longueur après nettoyage
df_cleaned = (
    df_bronze
    .withColumn("cleaned_text", clean_text_udf(col("raw_text")))
    .withColumn("cleaned_char_count", length(col("cleaned_text")))
)

# Stats avant filtrage
total_before = df_cleaned.count()
empty_pages  = df_cleaned.filter(col("cleaned_char_count") < MIN_CHARS).count()

print(f"📊 Pages totales       : {total_before}")
print(f"🗑️  Pages vides (<{MIN_CHARS} chars) : {empty_pages}")
print(f"✅ Pages conservées    : {total_before - empty_pages}")

# Filtrage des pages trop courtes
df_filtered = df_cleaned.filter(col("cleaned_char_count") >= MIN_CHARS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Chunking avec RecursiveCharacterTextSplitter
# MAGIC
# MAGIC ### Pourquoi chunker ?
# MAGIC
# MAGIC Les modèles d'embeddings ont une limite de tokens (~512 tokens pour la plupart).
# MAGIC On découpe donc chaque page en morceaux plus petits avec un **overlap**
# MAGIC pour ne pas perdre le contexte entre deux chunks consécutifs.
# MAGIC
# MAGIC ```
# MAGIC Page complète (2000 chars)
# MAGIC ├── Chunk 1 : chars 0    → 500
# MAGIC ├── Chunk 2 : chars 400  → 900   ← overlap de 100 chars avec chunk 1
# MAGIC ├── Chunk 3 : chars 800  → 1300  ← overlap de 100 chars avec chunk 2
# MAGIC └── Chunk 4 : chars 1200 → 1700
# MAGIC ```
# MAGIC
# MAGIC `RecursiveCharacterTextSplitter` essaie de couper sur `\n\n`, puis `\n`,
# MAGIC puis `.`, puis ` ` — jamais au milieu d'un mot.

# COMMAND ----------

import json
import uuid
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialisation du splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],  # Ordre de priorité de découpe
    length_function=len,
)


def chunk_text(file_name: str, page_number: int, text: str) -> List[Dict[str, Any]]:
    """
    Découpe le texte d'une page en chunks.
    
    Returns:
        Liste de dicts avec les métadonnées de chaque chunk.
    """
    if not text or len(text) < 10:
        return []
    
    chunks = splitter.split_text(text)
    results = []
    
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{file_name}_p{page_number:03d}_c{idx:03d}"
        
        # Métadonnées stockées en JSON pour faciliter la récupération lors du RAG
        metadata = json.dumps({
            "file_name":   file_name,
            "page_number": page_number,
            "chunk_index": idx,
            "chunk_id":    chunk_id
        })
        
        results.append({
            "chunk_id":        chunk_id,
            "file_name":       file_name,
            "page_number":     page_number,
            "chunk_index":     idx,
            "chunk_text":      chunk,
            "chunk_size":      len(chunk),
            "source_metadata": metadata
        })
    
    return results


# Test sur un exemple
test_chunks = chunk_text("test.pdf", 1, "Ceci est un texte de test. " * 30)
print(f"✅ Test chunking : {len(test_chunks)} chunk(s) générés")
for c in test_chunks:
    print(f"   Chunk {c['chunk_index']} : {c['chunk_size']} chars — '{c['chunk_text'][:60]}...'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Application du chunking sur tout le dataset
# MAGIC
# MAGIC On collecte les données filtrées côté driver (petit volume pour le trial),
# MAGIC on applique le chunking en Python, puis on recrée un DataFrame Spark.
# MAGIC
# MAGIC > 💡 **En production** avec des millions de pages, on utiliserait une
# MAGIC > **Pandas UDF** (vectorized UDF) pour distribuer le chunking sur les workers
# MAGIC > et éviter de collecter toutes les données sur le driver.

# COMMAND ----------

# Collecte sur le driver (acceptable sur le trial avec quelques PDFs)
rows = df_filtered.select("file_name", "page_number", "cleaned_text").collect()

print(f"🔄 Chunking de {len(rows)} page(s)...\n")

all_chunks = []
for row in rows:
    chunks = chunk_text(row["file_name"], row["page_number"], row["cleaned_text"])
    all_chunks.extend(chunks)
    print(f"  📄 {row['file_name']} — page {row['page_number']} → {len(chunks)} chunk(s)")

print(f"\n✅ Total : {len(all_chunks)} chunks générés")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Création du DataFrame Silver et écriture en Delta Table

# COMMAND ----------

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType
)

schema_silver = StructType([
    StructField("chunk_id",        StringType(),  nullable=False),
    StructField("file_name",       StringType(),  nullable=False),
    StructField("page_number",     IntegerType(), nullable=False),
    StructField("chunk_index",     IntegerType(), nullable=False),
    StructField("chunk_text",      StringType(),  nullable=False),
    StructField("chunk_size",      IntegerType(), nullable=True),
    StructField("source_metadata", StringType(),  nullable=True),
])

df_silver = spark.createDataFrame(all_chunks, schema=schema_silver)

print(f"📊 DataFrame Silver : {df_silver.count()} chunks, {len(df_silver.columns)} colonnes")
df_silver.printSchema()

# COMMAND ----------

# Écriture en Delta Table Silver
(
    df_silver
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_SILVER)
)

print(f"✅ Table Silver écrite : {TABLE_SILVER}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Validation & Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8a. Aperçu des chunks

# COMMAND ----------

spark.table(TABLE_SILVER).show(10, truncate=80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8b. Statistiques de chunking par fichier

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     file_name,
# MAGIC     COUNT(*)            AS nb_chunks,
# MAGIC     AVG(chunk_size)     AS avg_chunk_size,
# MAGIC     MIN(chunk_size)     AS min_chunk_size,
# MAGIC     MAX(chunk_size)     AS max_chunk_size,
# MAGIC     COUNT(DISTINCT page_number) AS nb_pages
# MAGIC FROM rag_project.silver.pdf_chunks
# MAGIC GROUP BY file_name
# MAGIC ORDER BY file_name

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8c. Distribution des tailles de chunks
# MAGIC
# MAGIC > Idéalement on veut une distribution centrée autour de CHUNK_SIZE.
# MAGIC > Des chunks très petits (<100 chars) peuvent indiquer des pages mal extraites.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     CASE
# MAGIC         WHEN chunk_size < 100  THEN '< 100'
# MAGIC         WHEN chunk_size < 200  THEN '100-200'
# MAGIC         WHEN chunk_size < 350  THEN '200-350'
# MAGIC         WHEN chunk_size < 500  THEN '350-500'
# MAGIC         ELSE '= 500 (max)'
# MAGIC     END AS taille_bucket,
# MAGIC     COUNT(*) AS nb_chunks
# MAGIC FROM rag_project.silver.pdf_chunks
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8d. Exemple de chunks consécutifs (vérification de l'overlap)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Vérifie que l'overlap fonctionne bien entre chunks consécutifs
# MAGIC SELECT
# MAGIC     chunk_id,
# MAGIC     page_number,
# MAGIC     chunk_index,
# MAGIC     chunk_size,
# MAGIC     SUBSTRING(chunk_text, 1, 120) AS debut_chunk
# MAGIC FROM rag_project.silver.pdf_chunks
# MAGIC WHERE file_name = (SELECT file_name FROM rag_project.silver.pdf_chunks LIMIT 1)
# MAGIC ORDER BY page_number, chunk_index

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Récapitulatif

# COMMAND ----------

nb_chunks = spark.table(TABLE_SILVER).count()
nb_files  = spark.table(TABLE_SILVER).select("file_name").distinct().count()

print("=" * 60)
print("✅ ÉTAPE 3 TERMINÉE — Récapitulatif")
print("=" * 60)
print(f"""
🥈 Table Silver : {TABLE_SILVER}
   → {nb_files} fichier(s) traité(s)
   → {nb_chunks} chunks prêts pour les embeddings

⚙️  Paramètres utilisés :
   chunk_size    = {CHUNK_SIZE} chars
   chunk_overlap = {CHUNK_OVERLAP} chars

✅ Ce qui a été démontré :
   - Nettoyage texte avec UDF Spark
   - Filtrage des pages vides
   - Chunking avec RecursiveCharacterTextSplitter
   - Overlap pour préserver le contexte
   - Schéma Silver avec source_metadata (clé pour le RAG)

🔜 Prochaine étape :
   Notebook 04_gold_embeddings.py
   → Génération des embeddings avec BGE (modèle open source)
   → Stockage dans la Delta Table Gold
   → Création de l'index Databricks Vector Search
""")
print("=" * 60)
