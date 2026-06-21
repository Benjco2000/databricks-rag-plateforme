# 🤖 Plateforme RAG à l'échelle sur Databricks
### Analyse de documents PDF & Chatbot d'entreprise avec GenAI & Vector Search

---

## 📋 Présentation du projet

Ce projet implémente une **plateforme RAG (Retrieval-Augmented Generation)** complète sur Databricks, capable de :
- Ingérer des documents PDF (rapports financiers, contrats, notes stratégiques)
- Extraire et structurer le texte via un pipeline Medallion (Bronze → Silver → Gold)
- Générer des embeddings vectoriels et les indexer dans Databricks Vector Search
- Répondre aux questions des employés via un chatbot d'entreprise basé sur les documents

> **Contexte métier** : Un employé peut poser la question *"Quels sont les engagements SLA du contrat fournisseur ?"* et obtenir une réponse précise citant le document source et la page exacte — sans que le LLM n'invente d'information.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        UNITY CATALOG                                │
│  rag_project                                                        │
│  ├── bronze.pdf_raw          (texte brut par page)                  │
│  ├── silver.pdf_chunks       (chunks nettoyés)                      │
│  └── gold.pdf_embeddings     (chunks + vecteurs)  ←── CDF activé   │
└─────────────────────────────────────────────────────────────────────┘
         ↑                          ↑                      ↑
    Notebook 02               Notebook 03            Notebook 04
   AutoLoader +              Nettoyage +           BGE-small +
   pdfplumber               Chunking              Embeddings
                                                       ↓
                                          ┌────────────────────────┐
                                          │  Databricks            │
                                          │  Vector Search         │
                                          │  (Delta Sync Index)    │
                                          └────────────┬───────────┘
                                                       │ similarity search
┌──────────────────┐    HTTP POST     ┌────────────────▼───────────┐
│  Application     │ ───────────────► │  Model Serving Endpoint    │
│  (UI, Bot, API)  │                  │  "rag-chatbot-endpoint"     │
└──────────────────┘                  │                            │
                                      │  RAGChatbot (PyFunc)       │
                                      │  ├── BGE-small (embed)     │
                                      │  ├── Vector Search         │
                                      │  └── Llama 3.3 70B (LLM)  │
                                      └────────────────────────────┘
                                                       ↑
                                          ┌────────────┴───────────┐
                                          │  MLflow Model Registry  │
                                          │  rag_project.gold.      │
                                          │  rag_chatbot (v1, v2…) │
                                          └────────────────────────┘
```

### Pipeline d'ingestion (Medallion Architecture)

```
PDFs (Volume UC)
      ↓  AutoLoader (cloudFiles)
🥉 BRONZE — pdf_raw
   file_name | page_number | raw_text | ingestion_date | file_size_bytes
      ↓  Nettoyage + RecursiveCharacterTextSplitter
🥈 SILVER — pdf_chunks
   chunk_id | file_name | page_number | chunk_index | chunk_text | source_metadata
      ↓  BGE-small embeddings (384 dimensions)
🥇 GOLD — pdf_embeddings
   chunk_id | chunk_text | embedding (Array<Float>) | document_owner [RLS-ready]
      ↓  Delta Sync Index (Change Data Feed)
🔍 VECTOR SEARCH INDEX
   Similarité cosine | Top-K retrieval
```

---

## 🛠️ Stack technique

| Composant | Technologie | Rôle |
|---|---|---|
| **Stockage** | Delta Lake + Unity Catalog | Tables Medallion, gouvernance |
| **Ingestion** | AutoLoader (`cloudFiles`) | Détection nouveaux fichiers |
| **Extraction PDF** | pdfplumber | Texte brut page par page |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Découpage avec overlap |
| **Embeddings** | BGE-small-en-v1.5 (HuggingFace) | Vecteurs 384 dimensions |
| **Index vectoriel** | Databricks Vector Search | Similarité cosine, Delta Sync |
| **LLM** | Llama 3.3 70B (Foundation Model APIs) | Génération de réponses |
| **Serving modèle** | MLflow PyFunc + Model Serving | Endpoint REST du chatbot |
| **Traçabilité** | MLflow Experiments + Model Registry | Versioning, reproductibilité |
| **Gouvernance** | Unity Catalog + Row Filters | RLS, audit, lignage |

---

## 📁 Structure du projet

```
databricks-rag-platform/
├── README.md
└── notebooks/
    ├── 01_setup_unity_catalog.py      # Catalog, Schemas, Volume, Secrets
    ├── 02_bronze_ingestion.py         # AutoLoader + pdfplumber → Bronze
    ├── 03_silver_text_cleaning.py     # Nettoyage + Chunking → Silver
    ├── 04_gold_embeddings.py          # Embeddings + Vector Search → Gold
    └── 05_rag_chatbot.py              # RAG Chain + MLflow + Model Serving
```

---

## 🚀 Lancement du projet

### Prérequis
- Databricks workspace (trial ou enterprise) sur AWS
- Cluster DBR 14.0+ (ML Runtime recommandé)
- Accès Unity Catalog (account admin)

### Étape 1 — Setup
```python
# Exécuter le notebook 01
# Crée : Catalog rag_project / Schemas bronze-silver-gold / Volume raw_pdfs
```

### Étape 2 — Upload des PDFs
```
Catalog → rag_project → bronze → raw_pdfs → Upload to this Volume
```

### Étapes 3 à 5 — Pipeline
```python
# Exécuter dans l'ordre :
02_bronze_ingestion.py        # ~2 min
03_silver_text_cleaning.py    # ~2 min
04_gold_embeddings.py         # ~15 min (Vector Search endpoint)
05_rag_chatbot.py             # ~15 min (Model Serving endpoint)
```

### Test de l'endpoint
```python
import requests

response = requests.post(
    "https://<workspace>.cloud.databricks.com/serving-endpoints/rag-chatbot-endpoint/invocations",
    json={"dataframe_records": [{"query": "Quel est le chiffre d'affaires ?"}]},
    headers={"Authorization": "Bearer <token>", "Content-Type": "application/json"}
)
print(response.json()["predictions"][0])
```

---

## 🧠 Concepts clés & Questions d'entretien

Cette section documente les décisions d'architecture et les concepts importants discutés pendant le projet.

---

### 1. Medallion Architecture — Pourquoi Bronze/Silver/Gold ?

Chaque couche a un rôle précis et une philosophie différente :

- **Bronze** : données brutes, **immuables**. On ne modifie jamais le Bronze — si une extraction échoue, on peut toujours re-traiter depuis la source.
- **Silver** : données nettoyées et structurées. C'est ici qu'on filtre les pages vides, qu'on normalise le texte, qu'on chunke.
- **Gold** : données prêtes pour la consommation métier (embeddings, features ML). C'est la seule couche exposée au serving.

> **Point entretien** : *"Le Bronze est append-only et immuable — c'est notre filet de sécurité. Si le nettoyage Silver est mal configuré, on peut tout re-traiter sans re-ingérer les sources."*

---

### 2. AutoLoader — Pourquoi pas un simple `spark.read` ?

`spark.read` est un batch one-shot. **AutoLoader** (`cloudFiles`) surveille en continu un dossier (S3, ADLS, Volume) et ingère automatiquement les nouveaux fichiers sans re-traiter les anciens.

```python
# Production pattern
df_stream = (
    spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("cloudFiles.schemaLocation", CHECKPOINT_PATH)
        .load(VOLUME_PATH)
)
```

Le **checkpoint** garantit l'idempotence : même si le job crashe et redémarre, aucun fichier n'est traité deux fois.

> **Point entretien** : *"AutoLoader est event-driven et idempotent grâce au checkpoint. En prod on utilise `Trigger.AvailableNow` pour du micro-batch ou `Trigger.ProcessingTime` pour du streaming continu."*

---

### 3. Chunking — Pourquoi découper avec un overlap ?

Les modèles d'embeddings ont une limite de tokens (~512). On découpe le texte en morceaux de 500 caractères avec **100 caractères d'overlap** entre chunks consécutifs.

```
Page (2000 chars)
├── Chunk 1 : chars 0   → 500
├── Chunk 2 : chars 400 → 900   ← overlap 100 chars avec chunk 1
├── Chunk 3 : chars 800 → 1300
└── Chunk 4 : chars 1200→ 1700
```

Sans overlap, une phrase coupée entre deux chunks ne serait jamais retrouvée par le retrieval.

`RecursiveCharacterTextSplitter` coupe en priorité sur `\n\n`, puis `\n`, puis `.`, puis ` ` — jamais au milieu d'un mot.

> **Point entretien** : *"L'overlap évite de perdre le contexte sémantique à la coupure. Sans overlap, une information à cheval sur deux chunks est introuvable par le Vector Search."*

---

### 4. Embeddings — Trial vs Production

**Sur le trial** : le modèle BGE-small tourne sur le **driver node** du cluster.
```python
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = model.encode(texts)
```

**En production**, deux options selon le contexte :

| Pattern | Usage | Avantage |
|---|---|---|
| **Pandas UDF + broadcast** | Batch (indexation de PDFs) | Parallèle sur tous les workers, moins cher |
| **Foundation Model APIs** | Temps réel (query utilisateur) | Toujours disponible, indépendant du cluster |

```python
# Pandas UDF — batch distribué
bc_model = spark.sparkContext.broadcast(SentenceTransformer("BAAI/bge-small-en-v1.5"))

@pandas_udf(ArrayType(FloatType()))
def embed_udf(texts: pd.Series) -> pd.Series:
    return pd.Series(bc_model.value.encode(texts.tolist()).tolist())

# Foundation Model APIs — temps réel
client = mlflow.deployments.get_deploy_client("databricks")
response = client.predict(endpoint="databricks-bge-large-en", inputs={"input": [query]})
```

> **Point entretien** : *"Pandas UDF pour le batch car le modèle est distribué sur les workers. Foundation Model APIs pour le serving temps réel car c'est un service managé indépendant du cluster — zéro cold start, SLA garanti."*

---

### 5. Change Data Feed — Sync incrémentale Vector Search

Le **CDF (Change Data Feed)** est une fonctionnalité Delta Lake qui capture chaque INSERT/UPDATE/DELETE sur une table. Le Vector Search Index utilise le CDF pour se mettre à jour de façon **incrémentale** — seuls les nouveaux chunks sont indexés, pas toute la table.

```sql
-- Activation obligatoire sur la table Gold
CREATE TABLE gold.pdf_embeddings
TBLPROPERTIES (delta.enableChangeDataFeed = true);
```

Sans CDF, le Vector Search devrait re-indexer toute la table Gold à chaque nouveau PDF — prohibitif à l'échelle.

> **Point entretien** : *"CDF est la condition sine qua non pour un Delta Sync Index en mode TRIGGERED ou CONTINUOUS. Sans ça, pas de sync incrémentale possible."*

---

### 6. Row-Level Security (RLS) — Architecture complète

Le RLS doit être cohérent à **deux niveaux** — c'est le point que la plupart des candidats oublient.

**Niveau 1 — Unity Catalog Row Filters (Delta Table)**
```sql
CREATE FUNCTION rag_project.security.document_filter(owner STRING)
RETURN owner = CURRENT_USER() OR IS_ACCOUNT_GROUP_MEMBER('admin');

ALTER TABLE rag_project.silver.pdf_chunks
SET ROW FILTER rag_project.security.document_filter ON (document_owner);
```

**Niveau 2 — Vector Search (filtres à la requête)**
```python
results = index.similarity_search(
    query_vector=embedding,
    filters={"document_owner": current_user},  # ← RLS au retrieval
    num_results=4
)
```

Sans le filtre Vector Search, un utilisateur pourrait retrouver des chunks de documents qu'il ne devrait pas voir, même si la Delta Table est protégée.

> **Point entretien** : *"Le RLS doit être appliqué à la fois sur la Delta Table ET sur le Vector Search Index à la requête. Oublier l'un des deux crée une fuite de données potentielle."*

---

### 7. MLflow — Ce qu'on a vraiment fait

On a réalisé **deux choses en une** opération :

```python
with mlflow.start_run(run_name="rag_chatbot_v1"):       # ← Experiment Run
    mlflow.log_params(...)                               # ← Traçabilité
    mlflow.log_metrics(...)
    mlflow.pyfunc.log_model(
        registered_model_name=MODEL_NAME                 # ← Model Registry
    )
```

| | MLflow Experiments | MLflow Model Registry |
|---|---|---|
| **Contient** | Runs, params, métriques | Modèles versionnés |
| **Usage** | Comparaison, traçabilité | Déploiement, cycle de vie |

En prod on ajouterait des **métriques RAGAS** pour évaluer objectivement chaque version :
- `faithfulness` : la réponse est-elle fidèle aux chunks ?
- `answer_relevancy` : la réponse répond-elle à la question ?
- `context_recall` : les bons chunks ont-ils été retrievés ?

---

### 8. Model Serving — Authentification OAuth M2M

Le container Model Serving ne peut pas utiliser ton token personnel. Il utilise un **token OAuth M2M (Machine-to-Machine)** injecté automatiquement par Databricks au démarrage.

Pour que Databricks sache quelles ressources injecter, il faut déclarer les dépendances lors du logging :

```python
from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint
)

mlflow.pyfunc.log_model(
    ...
    resources=[
        DatabricksVectorSearchIndex(index_name=VS_INDEX_NAME),
        DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
    ]
)
```

Sans cette déclaration → `MlflowException: Reading Databricks credential configuration failed`.

> **Point entretien** : *"On ne gère pas de credentials manuellement. En déclarant les ressources dans `log_model`, Databricks injecte automatiquement un token OAuth M2M dans le container — c'est le pattern sécurisé recommandé pour le serving."*

---

### 9. Scale-to-Zero — Impact sur le serving

`scale_to_zero_enabled=True` permet d'éteindre le container quand il est inactif.

| | Scale-to-Zero ON | Scale-to-Zero OFF |
|---|---|---|
| **Coût idle** | 0 | Fixe (container allumé H24) |
| **Cold start** | 1-3 min après inactivité | Aucun |
| **Usage** | Trial, trafic faible | Production critique, SLA strict |

Le cold start est **transparent pour l'application** — elle appelle toujours le même endpoint, Databricks redémarre le container automatiquement.

---

### 10. Ce qu'on ferait différemment à l'échelle (millions de PDFs)

| Étape | Trial | Production |
|---|---|---|
| **Ingestion** | Batch one-shot | AutoLoader streaming continu |
| **PDFs scannés** | Non géré | Azure AI Document Intelligence (OCR) |
| **Extraction** | Driver node | Pandas UDF distribuée |
| **Chunking** | Driver node | Pandas UDF + chunking sémantique par type de doc |
| **Embeddings** | Driver node | Pandas UDF broadcast ou Foundation Model APIs |
| **Qualité données** | Filtrage simple | Delta Live Tables + contraintes + quarantine table |
| **RLS** | Non implémenté | Unity Catalog Row Filters + VS filters |
| **Orchestration** | Notebooks manuels | Databricks Workflows schedulés |
| **Évaluation** | Manuelle | RAGAS automatisé à chaque version |
| **Monitoring** | Aucun | Logging questions/réponses → Delta Table → dashboard |

---

### 11. Online Tables vs Vector Search — Ne pas confondre

| | Vector Search Index | Online Table |
|---|---|---|
| **Recherche par** | Similarité sémantique (cosine) | Clé primaire exacte (lookup) |
| **Usage RAG** | Retrieval des chunks ✅ | Profil utilisateur pour personnalisation |
| **Latence** | ~100ms | <5ms |
| **Mise à jour** | Delta Sync (CDF) | Sync depuis Delta Table |

Dans notre RAG, **pas besoin d'Online Table**. Elle serait utile pour un RAG personnalisé qui adapte les réponses selon le profil de l'utilisateur (département, langue, séniorité).

---

### 12. Quand re-indexer toute la table Gold ?

| Changement | Re-indexation nécessaire ? |
|---|---|
| Nouveaux PDFs | ❌ Append + sync CDF automatique |
| Nouveau modèle d'embedding | ✅ Complet (vecteurs incompatibles) |
| Nouveau chunk_size | ✅ Complet (chunks différents) |
| Nouveau LLM ou prompt | ❌ Seul le Model Serving change |

> **Règle absolue** : le modèle d'embedding à l'indexation et au retrieval doit être **strictement identique**, y compris le préfixe de prompt (`"Represent this sentence: "`).

---

## 📊 Ce que ce projet démontre pour un entretien SA Databricks

| Compétence | Démonstration |
|---|---|
| **Delta Lake** | Medallion Architecture, CDF, Time Travel, DESCRIBE HISTORY |
| **Unity Catalog** | Catalog/Schema/Volume, RLS-ready, gouvernance |
| **AutoLoader** | Pattern production commenté et expliqué |
| **Vector Search** | Delta Sync Index, endpoint managé, similarité cosine |
| **Foundation Model APIs** | LLM + embeddings managés, zéro infrastructure |
| **MLflow** | Experiments + Model Registry + PyFunc wrapper |
| **Model Serving** | Endpoint REST, scale-to-zero, OAuth M2M |
| **GenAI / RAG** | Pipeline complet Retrieve → Augment → Generate |
| **Scalabilité** | Pandas UDF, broadcast, DLT, Workflows |
| **Sécurité** | RLS Delta + RLS Vector Search, OAuth M2M |
