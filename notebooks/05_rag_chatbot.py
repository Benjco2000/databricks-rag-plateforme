# Databricks notebook source
# MAGIC %md
# MAGIC # 🤖 Étape 5 — RAG Chatbot : LangChain + Foundation Model APIs + MLflow
# MAGIC
# MAGIC **Projet : Plateforme RAG à l'échelle**
# MAGIC
# MAGIC Ce notebook couvre :
# MAGIC - Construction de la chaîne RAG avec **LangChain**
# MAGIC - LLM via **Databricks Foundation Model APIs** (Llama 3.3 70B)
# MAGIC - Embedding de la query via **BGE-small** (même modèle que l'indexation)
# MAGIC - Retrieval depuis le **Vector Search Index**
# MAGIC - Logging du modèle RAG dans **MLflow**
# MAGIC - Déploiement via **Databricks Model Serving**
# MAGIC - Test end-to-end du chatbot

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Paramètres globaux

# COMMAND ----------

CATALOG_NAME      = "rag_project"
SCHEMA_GOLD       = "gold"
TABLE_GOLD        = f"{CATALOG_NAME}.{SCHEMA_GOLD}.pdf_embeddings"
VS_ENDPOINT_NAME  = "rag_vs_endpoint"
VS_INDEX_NAME     = f"{CATALOG_NAME}.{SCHEMA_GOLD}.pdf_embeddings_index"
EMBEDDING_MODEL   = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM     = 384
LLM_ENDPOINT      = "databricks-meta-llama-3-3-70b-instruct"
MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_GOLD}.rag_chatbot"
SERVING_ENDPOINT  = "rag-chatbot-endpoint"
# Récupère automatiquement ton email Databricks, pour le MLFLOW_EXPERIMENT
username = spark.sql("SELECT current_user()").first()[0]
MLFLOW_EXPERIMENT = f"/Users/{username}/rag_chatbot"
print(f"📊 Experiment path : {MLFLOW_EXPERIMENT}")

print(f"""
📋 Table Gold    : {TABLE_GOLD}
🔍 VS Endpoint   : {VS_ENDPOINT_NAME}
🔍 VS Index      : {VS_INDEX_NAME}
🤖 LLM Endpoint  : {LLM_ENDPOINT}
📊 MLflow Exp    : {MLFLOW_EXPERIMENT}
""")

# COMMAND ----------

# MAGIC %pip install langchain langchain-community sentence-transformers databricks-vectorsearch databricks-sdk mlflow --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Re-déclare les paramètres après restartPython
CATALOG_NAME      = "rag_project"
SCHEMA_GOLD       = "gold"
TABLE_GOLD        = f"{CATALOG_NAME}.{SCHEMA_GOLD}.pdf_embeddings"
VS_ENDPOINT_NAME  = "rag_vs_endpoint"
VS_INDEX_NAME     = f"{CATALOG_NAME}.{SCHEMA_GOLD}.pdf_embeddings_index"
EMBEDDING_MODEL   = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM     = 384
LLM_ENDPOINT      = "databricks-meta-llama-3-3-70b-instruct"
MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_GOLD}.rag_chatbot"
SERVING_ENDPOINT  = "rag-chatbot-endpoint"

# Récupère automatiquement ton email Databricks, pour le MLFLOW_EXPERIMENT
username = spark.sql("SELECT current_user()").first()[0]
MLFLOW_EXPERIMENT = f"/Users/{username}/rag_chatbot"
print(f"📊 Experiment path : {MLFLOW_EXPERIMENT}")



# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Vérification des prérequis
# MAGIC
# MAGIC Avant de lancer le chatbot, on vérifie que toutes les étapes
# MAGIC précédentes ont bien été exécutées.

# COMMAND ----------

from pyspark.sql import SparkSession
from databricks.vector_search.client import VectorSearchClient

spark = SparkSession.builder.getOrCreate()
spark.sql(f"USE CATALOG {CATALOG_NAME}")
vsc = VectorSearchClient()

# Vérification table Gold
gold_count = spark.table(TABLE_GOLD).count()
assert gold_count > 0, "❌ Table Gold vide — relance le notebook 04 !"
print(f"✅ Table Gold : {gold_count} embeddings")

# Vérification Vector Search Index
index_info = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME).describe()
vs_status  = index_info.get("status", {}).get("detailed_state", "UNKNOWN")
assert vs_status == "ONLINE", f"❌ VS Index pas ONLINE : {vs_status}"
print(f"✅ Vector Search Index : {vs_status}")

print("\n✅ Tous les prérequis sont OK — on peut lancer le chatbot !")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Chargement du modèle d'embedding
# MAGIC
# MAGIC Règle absolue en RAG : **même modèle** à l'indexation et au retrieval.
# MAGIC Changer de modèle = re-indexer toute la table Gold.

# COMMAND ----------

from sentence_transformers import SentenceTransformer

print(f"🔄 Chargement du modèle : {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("✅ Modèle chargé\n")

def embed_query(query: str) -> list:
    """Encode une query avec le même préfixe BGE que lors de l'indexation."""
    return embedding_model.encode(
        f"Represent this sentence: {query}",
        normalize_embeddings=True
    ).tolist()

# Test
test_vec = embed_query("test")
print(f"✅ Embedding test : {len(test_vec)} dimensions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fonction de retrieval — Vector Search

# COMMAND ----------

def retrieve_chunks(query: str, num_results: int = 4) -> list:
    """
    Retrouve les K chunks les plus proches sémantiquement d'une query.
    C'est l'étape RETRIEVE du RAG.
    """
    query_vector = embed_query(query)
    index        = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)

    results = index.similarity_search(
        query_vector=query_vector,
        columns=["chunk_id", "file_name", "page_number", "chunk_text"],
        num_results=num_results
    )

    hits = results.get("result", {}).get("data_array", [])
    cols = ["chunk_id", "file_name", "page_number", "chunk_text", "score"]

    return [dict(zip(cols, hit)) for hit in hits]


# Test du retrieval
print("🔍 Test retrieval :\n")
test_chunks = retrieve_chunks("Quel est le chiffre d'affaires ?", num_results=2)
for i, c in enumerate(test_chunks):
    print(f"Chunk {i+1} | {c['file_name']} p.{c['page_number']} | score: {c['score']:.4f}")
    print(f"  {c['chunk_text'][:150]}...\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Construction de la chaîne RAG
# MAGIC
# MAGIC Pattern **Retrieve → Augment → Generate** :
# MAGIC
# MAGIC ```
# MAGIC Question utilisateur
# MAGIC       ↓
# MAGIC   [RETRIEVE]  Vector Search → top-K chunks
# MAGIC       ↓
# MAGIC   [AUGMENT]   Injection des chunks dans le prompt
# MAGIC       ↓
# MAGIC   [GENERATE]  LLM génère la réponse basée UNIQUEMENT sur le contexte
# MAGIC       ↓
# MAGIC   Réponse + sources citées
# MAGIC ```

# COMMAND ----------

import mlflow.deployments

fm_client = mlflow.deployments.get_deploy_client("databricks")

SYSTEM_PROMPT = """Tu es un assistant d'entreprise expert en analyse de documents.
Tu réponds aux questions des employés en te basant UNIQUEMENT sur les documents fournis en contexte.

Règles strictes :
1. Si la réponse n'est pas dans le contexte, réponds "Je ne trouve pas cette information dans les documents disponibles."
2. Cite toujours la source (nom du fichier et numéro de page) de chaque information.
3. Réponds en français, de façon claire et structurée.
4. Ne fabrique jamais d'information absente du contexte."""


def build_prompt(query: str, chunks: list) -> str:
    """Construit le prompt RAG — étape AUGMENT."""
    context_blocks = [
        f"[Source {i+1} : {c['file_name']}, page {c['page_number']}]\n{c['chunk_text']}"
        for i, c in enumerate(chunks)
    ]
    context = "\n\n---\n\n".join(context_blocks)

    return (
        f"Voici les extraits de documents pertinents :\n\n{context}\n\n---\n\n"
        f"Question : {query}\n\n"
        f"Réponds en citant les sources [Source X] pour chaque information."
    )


def rag_answer(query: str, num_chunks: int = 4) -> dict:
    """
    Pipeline RAG complet : Retrieve → Augment → Generate.
    Retourne la réponse, les sources et les chunks utilisés.
    """
    # 1. RETRIEVE
    chunks = retrieve_chunks(query, num_results=num_chunks)
    if not chunks:
        return {"answer": "Aucun document pertinent trouvé.", "sources": [], "chunks": []}

    # 2. AUGMENT
    prompt = build_prompt(query, chunks)

    # 3. GENERATE — Foundation Model APIs
    response = fm_client.predict(
        endpoint=LLM_ENDPOINT,
        inputs={
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            "max_tokens":  1024,
            "temperature": 0.1  # Faible = réponses factuelles
        }
    )

    answer  = response["choices"][0]["message"]["content"]
    sources = list({f"{c['file_name']} (page {c['page_number']})" for c in chunks})

    return {"answer": answer, "sources": sources, "chunks": chunks}


print("✅ Chaîne RAG construite")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Tests end-to-end

# COMMAND ----------

def display_rag_result(query: str):
    print(f"\n{'='*65}")
    print(f"❓ Question : {query}")
    print(f"{'='*65}")
    result = rag_answer(query)
    print(f"\n💬 Réponse :\n{result['answer']}")
    print(f"\n📚 Sources :")
    for src in result["sources"]:
        print(f"   • {src}")
    print(f"\n🔍 Chunks : {len(result['chunks'])} retrievés")
    for c in result["chunks"]:
        print(f"   [{c['score']:.3f}] {c['file_name']} p.{c['page_number']}")


questions = [
    "Quel est le chiffre d'affaires et la marge opérationnelle ?",
    "Est-ce que l'entreprise ADBE fait du rachat d'action ?",
    "Quelle est la météo à Paris aujourd'hui ?",  # hors contexte → doit refuser
]

for q in questions:
    display_rag_result(q)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Logging dans MLflow
# MAGIC
# MAGIC On encapsule la chaîne RAG dans une **classe MLflow PyFunc**
# MAGIC pour la logger, la versionner et la déployer via Model Serving.
# MAGIC
# MAGIC > 💡 `mlflow.pyfunc` est le format universel MLflow — il permet de wrapper
# MAGIC > n'importe quel modèle Python et de l'exposer comme endpoint REST standard.

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import pandas as pd


class RAGChatbot(mlflow.pyfunc.PythonModel):
    """
    Wrapper MLflow PyFunc pour le chatbot RAG.

    Deux méthodes imposées par l'interface PyFunc :
    - load_context : appelé UNE FOIS au démarrage du serving (chargement du modèle)
    - predict      : appelé à CHAQUE requête
    """

    def load_context(self, context):
        from sentence_transformers import SentenceTransformer
        from databricks.vector_search.client import VectorSearchClient
        import mlflow.deployments
        import json, os

        # Lecture config
        config_path = context.artifacts["config"]
        with open(config_path) as f:
            p = json.load(f)

        self.embedding_model = SentenceTransformer(p["embedding_model"])
        self.vs_endpoint     = p["vs_endpoint"]
        self.vs_index        = p["vs_index"]
        self.llm_endpoint    = p["llm_endpoint"]
        self.system_prompt   = p["system_prompt"]

        # Auth explicite pour le serving container
        # Le container Serving utilise le token de l'environnement
        self.vsc       = VectorSearchClient(disable_notice=True)
        self.fm_client = mlflow.deployments.get_deploy_client("databricks")

    def _embed(self, query: str) -> list:
        return self.embedding_model.encode(
            f"Represent this sentence: {query}",
            normalize_embeddings=True
        ).tolist()

    def _retrieve(self, query: str, k: int = 4) -> list:
        try:
            index   = self.vsc.get_index(self.vs_endpoint, self.vs_index)
            results = index.similarity_search(
                query_vector=self._embed(query),
                columns=["file_name", "page_number", "chunk_text"],
                num_results=k
            )
            hits = results.get("result", {}).get("data_array", [])
            return [
                {"file_name": h[0], "page_number": h[1], "chunk_text": h[2]}
                for h in hits
            ]
        except Exception as e:
            # On ne crashe pas si le VS est indisponible
            print(f"⚠️ Retrieval error: {e}")
            return []

    def predict(self, context, model_input):
        import pandas as pd

        # Health check Databricks envoie un DataFrame vide — on le gère
        if model_input is None or len(model_input) == 0:
            return pd.Series(["OK"])

        results = []
        for query in model_input["query"]:
            try:
                chunks = self._retrieve(str(query))
                if not chunks:
                    results.append("Aucun document pertinent trouvé.")
                    continue

                context_text = "\n\n---\n\n".join([
                    f"[{c['file_name']} p.{c['page_number']}]\n{c['chunk_text']}"
                    for c in chunks
                ])
                prompt = (
                    f"Contexte :\n{context_text}\n\n"
                    f"Question : {query}\n\n"
                    f"Réponds en citant les sources."
                )
                response = self.fm_client.predict(
                    endpoint=self.llm_endpoint,
                    inputs={
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user",   "content": prompt}
                        ],
                        "max_tokens":  1024,
                        "temperature": 0.1
                    }
                )
                results.append(response["choices"][0]["message"]["content"])

            except Exception as e:
                results.append(f"Erreur : {str(e)}")

        return pd.Series(results)

print("✅ Classe RAGChatbot MLflow PyFunc définie")

# COMMAND ----------

from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint
)

mlflow.set_experiment(MLFLOW_EXPERIMENT)

artifacts = {
    "embedding_model": EMBEDDING_MODEL,
    "vs_endpoint":     VS_ENDPOINT_NAME,
    "vs_index":        VS_INDEX_NAME,
    "llm_endpoint":    LLM_ENDPOINT,
    "system_prompt":   SYSTEM_PROMPT,
}

with mlflow.start_run(run_name="rag_chatbot_v3") as run:

    mlflow.log_params({
        "embedding_model": EMBEDDING_MODEL,
        "llm_endpoint":    LLM_ENDPOINT,
        "num_chunks":      4,
        "temperature":     0.1,
        "chunk_size":      500,
        "chunk_overlap":   100,
    })

    mlflow.log_metrics({
        "nb_docs_indexed":   spark.table(TABLE_GOLD).select("file_name").distinct().count(),
        "nb_chunks_indexed": spark.table(TABLE_GOLD).count(),
    })

    import json, tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "config.json")
        with open(config_path, "w") as f:
            json.dump(artifacts, f)

        mlflow.pyfunc.log_model(
            artifact_path="rag_chatbot",
            python_model=RAGChatbot(),
            artifacts={"config": config_path},
            registered_model_name=MODEL_NAME,
            input_example=pd.DataFrame({"query": ["Quel est le chiffre d'affaires ?"]}),
            pip_requirements=[
                "sentence-transformers",
                "databricks-vectorsearch",
                "mlflow",
            ],
            # ← La clé : déclarer les ressources Databricks accessibles
            resources=[
                DatabricksVectorSearchIndex(index_name=VS_INDEX_NAME),
                DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
                # DatabricksServingEndpoint(endpoint_name=VS_ENDPOINT_NAME),
            ]
        )

    run_id = run.info.run_id
    print(f"✅ Run MLflow    : {run_id}")
    print(f"✅ Modèle loggé : {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Déploiement via Databricks Model Serving
# MAGIC
# MAGIC On déploie le modèle RAG comme un **endpoint REST** managé.
# MAGIC Une fois déployé, il expose :
# MAGIC `POST /serving-endpoints/rag-chatbot-endpoint/invocations`
# MAGIC
# MAGIC > ⏱️ La création prend 5 à 10 minutes.
# MAGIC > `scale_to_zero_enabled=True` → l'endpoint s'éteint si inactif,
# MAGIC > ce qui préserve tes crédits trial.

# COMMAND ----------

from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from mlflow.tracking import MlflowClient

w      = WorkspaceClient()
client = MlflowClient()

# Récupération de la dernière version
versions       = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0].version
print(f"🚀 Mise à jour : {MODEL_NAME} v{latest_version} → {SERVING_ENDPOINT}\n")

new_config = EndpointCoreConfigInput(
    served_entities=[
        ServedEntityInput(
            entity_name=MODEL_NAME,
            entity_version=latest_version,
            workload_size="Small",
            scale_to_zero_enabled=True,
        )
    ]
)

try:
    # Update si existe, create sinon
    existing = [ep.name for ep in w.serving_endpoints.list()]
    if SERVING_ENDPOINT in existing:
        w.serving_endpoints.update_config_and_wait(
            name=SERVING_ENDPOINT,
            served_entities=new_config.served_entities,
            timeout=timedelta(minutes=20)
        )
        print(f"✅ Endpoint '{SERVING_ENDPOINT}' mis à jour vers v{latest_version} !")
    else:
        w.serving_endpoints.create_and_wait(
            name=SERVING_ENDPOINT,
            config=new_config,
            timeout=timedelta(minutes=10)
        )
        print(f"✅ Endpoint '{SERVING_ENDPOINT}' créé !")

except Exception as e:
    print(f"⚠️  {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Test de l'endpoint REST
# MAGIC
# MAGIC On appelle l'endpoint comme le ferait une application externe —
# MAGIC via HTTP avec un Personal Access Token.

# COMMAND ----------

# Verification du déploiement de l'endpoint
# Cellule de diagnostic — exécute ça avant le test
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

endpoints = w.serving_endpoints.list()
print("📋 Endpoints disponibles :\n")
for ep in endpoints:
    print(f"  • {ep.name} — state: {ep.state}")

# COMMAND ----------

import requests

token    = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host     = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
endpoint = f"{host}/serving-endpoints/{SERVING_ENDPOINT}/invocations"

def call_chatbot(query: str) -> str:
    """Appelle le chatbot RAG via son endpoint REST."""
    response = requests.post(
        endpoint,
        json={"dataframe_records": [{"query": query}]},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json"
        }
    )
    response.raise_for_status()
    return response.json()["predictions"][0]


# Test final via API REST
query  = "Donne moi des bonnes nouvelles annoncées par Adobe ?"
answer = call_chatbot(query)

print(f"🌐 Appel REST vers : {endpoint}\n")
print(f"❓ Question : {query}")
print(f"\n💬 Réponse :\n{answer}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Récapitulatif final du projet

# COMMAND ----------

print("=" * 65)
print("🎉 PROJET RAG COMPLET — Récapitulatif")
print("=" * 65)
print(f"""
Architecture Medallion
  🥉 Bronze : {CATALOG_NAME}.bronze.pdf_raw
  🥈 Silver : {CATALOG_NAME}.silver.pdf_chunks
  🥇 Gold   : {CATALOG_NAME}.gold.pdf_embeddings

Vector Search
  🔍 Endpoint : {VS_ENDPOINT_NAME}
  🔍 Index    : {VS_INDEX_NAME}  (Delta Sync, CDF)

MLflow
  📊 Experiment : {MLFLOW_EXPERIMENT}
  📦 Model      : {MODEL_NAME}
  🚀 Serving    : {SERVING_ENDPOINT}  (scale-to-zero)

Stack technique démontrée
  ✅ Unity Catalog — Catalog / Schemas / Volumes / RLS-ready
  ✅ Delta Lake + Medallion Architecture
  ✅ AutoLoader pattern (commenté, production-ready)
  ✅ pdfplumber — extraction PDF page par page
  ✅ LangChain RecursiveCharacterTextSplitter + overlap
  ✅ BGE-small embeddings open source (0 coût API)
  ✅ Databricks Vector Search — Delta Sync Index + CDF
  ✅ Foundation Model APIs — Llama 3.3 70B
  ✅ MLflow PyFunc + Model Registry
  ✅ Databricks Model Serving — endpoint REST
  ✅ Scale-to-zero — optimisation coûts trial
""")
print("=" * 65)
