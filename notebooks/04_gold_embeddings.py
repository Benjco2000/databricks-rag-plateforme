# Databricks notebook source
# MAGIC %md
# MAGIC # 🥇 Étape 4 — Gold : Embeddings & Databricks Vector Search
# MAGIC
# MAGIC **Projet : Plateforme RAG à l'échelle**
# MAGIC
# MAGIC Ce notebook couvre :
# MAGIC - Génération des embeddings avec **BGE-small** (modèle open source, via sentence-transformers)
# MAGIC - Stockage dans la **Delta Table Gold** (`gold.pdf_embeddings`)
# MAGIC - Création d'un **Databricks Vector Search endpoint**
# MAGIC - Création d'un **Delta Sync Index** sur la table Gold
# MAGIC - Test de similarité sémantique avec quelques requêtes

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Paramètres globaux

# COMMAND ----------

CATALOG_NAME  = "rag_project"
SCHEMA_SILVER = "silver"
SCHEMA_GOLD   = "gold"
TABLE_SILVER  = f"{CATALOG_NAME}.{SCHEMA_SILVER}.pdf_chunks"
TABLE_GOLD    = f"{CATALOG_NAME}.{SCHEMA_GOLD}.pdf_embeddings"

# Vector Search
VS_ENDPOINT_NAME = "rag_vs_endpoint"
VS_INDEX_NAME    = f"{CATALOG_NAME}.{SCHEMA_GOLD}.pdf_embeddings_index"

# Modèle d'embedding (open source, tourne sur le driver du cluster)
# BGE-small : 384 dimensions, rapide, très bon rapport qualité/perf
EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM    = 384

print(f"""
📋 Table source    : {TABLE_SILVER}
📋 Table Gold      : {TABLE_GOLD}
🔍 VS Endpoint     : {VS_ENDPOINT_NAME}
🔍 VS Index        : {VS_INDEX_NAME}

🤖 Modèle embedding : {EMBEDDING_MODEL}
📐 Dimensions       : {EMBEDDING_DIM}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Installation des dépendances

# COMMAND ----------

# MAGIC %pip install sentence-transformers databricks-vectorsearch --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Re-déclare les paramètres après restartPython
CATALOG_NAME     = "rag_project"
SCHEMA_SILVER    = "silver"
SCHEMA_GOLD      = "gold"
TABLE_SILVER     = f"{CATALOG_NAME}.{SCHEMA_SILVER}.pdf_chunks"
TABLE_GOLD       = f"{CATALOG_NAME}.{SCHEMA_GOLD}.pdf_embeddings"
VS_ENDPOINT_NAME = "rag_vs_endpoint"
VS_INDEX_NAME    = f"{CATALOG_NAME}.{SCHEMA_GOLD}.pdf_embeddings_index"
EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM    = 384

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Chargement du modèle d'embedding
# MAGIC
# MAGIC On utilise **BGE-small** de BAAI (Beijing Academy of AI) :
# MAGIC - Modèle open source, aucun coût API
# MAGIC - 384 dimensions — compact et rapide
# MAGIC - Excellent sur les textes techniques et financiers
# MAGIC - Compatible avec Databricks Vector Search (cosine similarity)
# MAGIC
# MAGIC > 💡 En production, on utiliserait **Databricks BGE via Foundation Model APIs**
# MAGIC > pour ne pas dépendre du driver et scaler sur des GPU workers.
# MAGIC > Sur le trial, on le fait tourner localement sur le driver.

# COMMAND ----------

from sentence_transformers import SentenceTransformer
import numpy as np

print(f"🔄 Chargement du modèle {EMBEDDING_MODEL}...")
model = SentenceTransformer(EMBEDDING_MODEL)
print("✅ Modèle chargé\n")

# Test rapide
test_sentences = [
    "Le chiffre d'affaires d'ACME est de 4,2 milliards d'euros.",
    "Quels sont les revenus annuels de l'entreprise ?",
    "La météo est agréable aujourd'hui à Paris."
]
test_embeddings = model.encode(test_sentences)

print(f"📐 Shape des embeddings : {test_embeddings.shape}")
print(f"   → {len(test_sentences)} phrases × {test_embeddings.shape[1]} dimensions\n")

# Similarité cosine entre les phrases (pour valider le modèle)
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(test_embeddings)
print("🔍 Matrice de similarité cosine :")
print(f"   Phrase 1 ↔ Phrase 2 (même sujet)    : {sim_matrix[0][1]:.3f}  ← doit être élevé")
print(f"   Phrase 1 ↔ Phrase 3 (sujets différents): {sim_matrix[0][2]:.3f}  ← doit être faible")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Lecture de la table Silver

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sql(f"USE CATALOG {CATALOG_NAME}")

df_silver = spark.table(TABLE_SILVER)
total_chunks = df_silver.count()

print(f"📊 {total_chunks} chunks à embedder\n")
df_silver.show(5, truncate=80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Génération des embeddings
# MAGIC
# MAGIC On collecte les chunks côté driver et on génère les embeddings en batch.
# MAGIC
# MAGIC > 💡 **Pattern production** : on utiliserait une **Pandas UDF** avec
# MAGIC > `@pandas_udf` et un broadcast du modèle sur tous les workers,
# MAGIC > ou les **Foundation Model APIs** de Databricks pour un endpoint
# MAGIC > d'embedding managé et scalable à l'infini.

# COMMAND ----------

import pandas as pd
from typing import List

# Collecte des chunks (acceptable sur le trial)
chunks_df = df_silver.select(
    "chunk_id", "file_name", "page_number",
    "chunk_index", "chunk_text", "source_metadata"
).toPandas()

print(f"🔄 Génération des embeddings pour {len(chunks_df)} chunks...")
print(f"   Modèle : {EMBEDDING_MODEL}\n")

# Génération en batch (plus efficace que un par un)
# BGE recommande un préfixe "Represent this sentence: " pour les passages
texts_to_embed = [
    f"Represent this sentence: {text}"
    for text in chunks_df["chunk_text"].tolist()
]

embeddings = model.encode(
    texts_to_embed,
    batch_size=32,          # 32 chunks à la fois
    show_progress_bar=True,
    normalize_embeddings=True  # Normalisation L2 → cosine similarity = dot product
)

print(f"\n✅ Embeddings générés : shape {embeddings.shape}")
chunks_df["embedding"] = embeddings.tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Création de la Delta Table Gold
# MAGIC
# MAGIC La table Gold contient les chunks + leurs embeddings vectoriels.
# MAGIC C'est cette table qui sera synchronisée avec le Vector Search Index.
# MAGIC
# MAGIC > ⚠️ **Contrainte Databricks Vector Search** :
# MAGIC > La table Gold doit avoir **Change Data Feed (CDF) activé**.
# MAGIC > CDF permet à Vector Search de détecter les nouvelles lignes
# MAGIC > et de mettre à jour l'index de façon incrémentale.

# COMMAND ----------

from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, ArrayType, FloatType
)

schema_gold = StructType([
    StructField("chunk_id",        StringType(),              nullable=False),
    StructField("file_name",       StringType(),              nullable=False),
    StructField("page_number",     IntegerType(),             nullable=False),
    StructField("chunk_index",     IntegerType(),             nullable=False),
    StructField("chunk_text",      StringType(),              nullable=False),
    StructField("source_metadata", StringType(),              nullable=True),
    StructField("embedding",       ArrayType(FloatType()),    nullable=False),
    # Colonne RLS-ready (non utilisée sur le trial mais présente pour la prod)
    # StructField("document_owner",  StringType(),              nullable=True),
])

df_gold = spark.createDataFrame(chunks_df, schema=schema_gold)

# Ajout d'un document_owner fictif (RLS-ready)
from pyspark.sql.functions import lit
df_gold = df_gold.withColumn("document_owner", lit("trial_user"))

print(f"📊 DataFrame Gold : {df_gold.count()} lignes")
df_gold.printSchema()

# COMMAND ----------

# Écriture en Delta Table Gold
# IMPORTANT : delta.enableChangeDataFeed = true requis pour Vector Search
(
    df_gold
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")   # ← Obligatoire pour VS
    .saveAsTable(TABLE_GOLD)
)

print(f"✅ Table Gold écrite : {TABLE_GOLD}")
print(f"   → Change Data Feed activé (requis pour Vector Search)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vérification CDF activé

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Vérifie que le CDF est bien activé sur la table Gold
# MAGIC DESCRIBE DETAIL rag_project.gold.pdf_embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Création du Vector Search Endpoint
# MAGIC
# MAGIC Un **endpoint Vector Search** est un service managé Databricks qui héberge
# MAGIC un ou plusieurs index vectoriels. Il expose une API REST pour la recherche
# MAGIC par similarité.
# MAGIC
# MAGIC > ⏱️ La création d'un endpoint prend **5 à 10 minutes** sur le trial.
# MAGIC > On attend qu'il soit en état `ONLINE` avant de créer l'index.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
import time

vsc = VectorSearchClient()

# Vérifie si l'endpoint existe déjà
existing_endpoints = [ep["name"] for ep in vsc.list_endpoints().get("endpoints", [])]

if VS_ENDPOINT_NAME in existing_endpoints:
    print(f"✅ Endpoint '{VS_ENDPOINT_NAME}' existe déjà")
else:
    print(f"🔄 Création de l'endpoint '{VS_ENDPOINT_NAME}'...")
    vsc.create_endpoint(
        name=VS_ENDPOINT_NAME,
        endpoint_type="STANDARD"   # STANDARD = serverless, pas de GPU dédié
    )
    print("   → Endpoint en cours de création (5-10 min)...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Attente que l'endpoint soit ONLINE

# COMMAND ----------

def wait_for_endpoint(vsc, endpoint_name: str, timeout_minutes: int = 15):
    """Attend que l'endpoint VS soit en état ONLINE."""
    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        status = vsc.get_endpoint(endpoint_name)["endpoint_status"]["state"]
        print(f"   [{time.strftime('%H:%M:%S')}] Endpoint status : {status}")
        if status == "ONLINE":
            print(f"\n✅ Endpoint ONLINE !")
            return True
        elif status in ["OFFLINE", "ERROR"]:
            raise Exception(f"Endpoint en erreur : {status}")
        time.sleep(30)
    raise TimeoutError(f"Endpoint pas ONLINE après {timeout_minutes} minutes")

wait_for_endpoint(vsc, VS_ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Création du Delta Sync Index
# MAGIC
# MAGIC Le **Delta Sync Index** synchronise automatiquement la table Gold
# MAGIC avec l'index vectoriel. Chaque modification de la table (nouveau chunk,
# MAGIC mise à jour) est répercutée dans l'index via le Change Data Feed.
# MAGIC
# MAGIC Types d'index Databricks Vector Search :
# MAGIC - **Delta Sync** ← on utilise ça : sync auto depuis une Delta Table
# MAGIC - **Direct Vector Access** : on pousse les vecteurs manuellement via API

# COMMAND ----------

# Vérifie si l'index existe déjà
try:
    existing_index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    print(f"✅ Index '{VS_INDEX_NAME}' existe déjà")
except Exception:
    print(f"🔄 Création de l'index '{VS_INDEX_NAME}'...")
    vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME,
        source_table_name=TABLE_GOLD,           # Table Delta source
        pipeline_type="TRIGGERED",              # TRIGGERED = sync manuelle
                                                # CONTINUOUS = sync temps réel (prod)
        primary_key="chunk_id",                 # Clé primaire unique
        embedding_dimension=EMBEDDING_DIM,      # 384 pour BGE-small
        embedding_vector_column="embedding",    # Colonne contenant les vecteurs
    )
    print("   → Index en cours de création...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Attente que l'index soit ONLINE

# COMMAND ----------

def wait_for_index(vsc, endpoint_name: str, index_name: str, timeout_minutes: int = 20):
    """Attend que l'index VS soit en état ONLINE."""
    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        index_info = vsc.get_index(endpoint_name, index_name)
        
        # L'objet AISearchIndex expose un dict via .describe()
        index_dict = index_info.describe()
        status = index_dict.get("status", {}).get("detailed_state", "UNKNOWN")
        
        print(f"   [{time.strftime('%H:%M:%S')}] Index status : {status}")
        if status == "ONLINE":
            print(f"\n✅ Index ONLINE et prêt !")
            return True
        elif "ERROR" in status:
            raise Exception(f"Index en erreur : {status}")
        time.sleep(30)
    raise TimeoutError(f"Index pas ONLINE après {timeout_minutes} minutes")

wait_for_index(vsc, VS_ENDPOINT_NAME, VS_INDEX_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Test de recherche par similarité
# MAGIC
# MAGIC On teste le Vector Search avec quelques questions métier
# MAGIC pour valider que le retrieval fonctionne correctement.

# COMMAND ----------

def search_similar_chunks(query: str, num_results: int = 3) -> pd.DataFrame:
    """
    Recherche les chunks les plus similaires à une query.
    
    1. Encode la query avec le même modèle d'embedding
    2. Interroge le Vector Search Index
    3. Retourne les chunks les plus proches (cosine similarity)
    """
    # Même préfixe BGE que lors de l'indexation
    query_embedding = model.encode(
        f"Represent this sentence: {query}",
        normalize_embeddings=True
    ).tolist()

    index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    results = index.similarity_search(
        query_vector=query_embedding,
        columns=["chunk_id", "file_name", "page_number", "chunk_text", "source_metadata"],
        num_results=num_results
    )

    hits = results.get("result", {}).get("data_array", [])
    cols = ["chunk_id", "file_name", "page_number", "chunk_text", "source_metadata", "score"]
    return pd.DataFrame(hits, columns=cols)


# Tests avec des questions métier
test_queries = [
    "Quel est le chiffre d'affaires annuel ?",
    "Quels sont les risques identifiés ?",
    "Quelles sont les obligations du prestataire ?",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"🔍 Query : {query}")
    print(f"{'='*60}")
    results_df = search_similar_chunks(query, num_results=2)
    for _, row in results_df.iterrows():
        print(f"\n  📄 {row['file_name']} — page {row['page_number']}  (score: {row['score']:.4f})")
        print(f"  {row['chunk_text'][:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Récapitulatif

# COMMAND ----------

nb_gold = spark.table(TABLE_GOLD).count()

print("=" * 60)
print("✅ ÉTAPE 4 TERMINÉE — Récapitulatif")
print("=" * 60)
print(f"""
🥇 Table Gold    : {TABLE_GOLD}
   → {nb_gold} embeddings stockés
   → CDF activé (sync incrémentale)
   → Colonne document_owner (RLS-ready)

🔍 Vector Search
   → Endpoint : {VS_ENDPOINT_NAME}  (ONLINE)
   → Index    : {VS_INDEX_NAME}     (ONLINE)
   → Type     : Delta Sync / TRIGGERED
   → Similarité : cosine

✅ Ce qui a été démontré :
   - Embeddings open source (BGE-small, 0 coût API)
   - Change Data Feed pour sync incrémentale
   - Delta Sync Index (Vector Search natif Databricks)
   - Retrieval sémantique validé sur questions métier
   - Architecture RLS-ready (document_owner)

🔜 Prochaine étape :
   Notebook 05_rag_chatbot.py
   → Chaîne RAG avec LangChain
   → LLM via Databricks Foundation Model APIs (Llama 3 / DBRX)
   → Logging MLflow + Model Serving
""")
print("=" * 60)

# COMMAND ----------


