# Databricks notebook source
# MAGIC %md
# MAGIC # 🏗️ Étape 1 — Setup Unity Catalog & Ingestion PDFs
# MAGIC
# MAGIC **Projet : Plateforme RAG à l'échelle — Analyse de documents PDF**
# MAGIC
# MAGIC Ce notebook couvre :
# MAGIC - Création du Catalog / Schemas (Bronze, Silver, Gold) dans Unity Catalog
# MAGIC - Création d'un Volume pour stocker les PDFs bruts
# MAGIC - Upload de PDFs samples (rapports financiers simulés)
# MAGIC - Vérification de l'environnement (libs, cluster config)
# MAGIC - Stockage de la clé OpenAI dans Databricks Secrets

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Vérification de l'environnement

# COMMAND ----------

import sys
print(f"Python version : {sys.version}")

# Vérification des libs critiques disponibles sur le cluster
libs_to_check = ["pyspark", "delta", "mlflow"]
for lib in libs_to_check:
    try:
        __import__(lib)
        print(f"✅ {lib} — OK")
    except ImportError:
        print(f"❌ {lib} — MANQUANT")

# Affiche la version de Databricks Runtime
try:
    dbr_version = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
    print(f"\n🔷 Databricks Runtime : {dbr_version}")
except:
    print("\n⚠️ Impossible de récupérer la version DBR")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Paramètres du projet
# MAGIC
# MAGIC > **Adapter** `CATALOG_NAME` à ton workspace si besoin.
# MAGIC > Sur un trial Databricks, tu as généralement un catalog `main` pré-existant.

# COMMAND ----------

# ─── Paramètres globaux ───────────────────────────────────────────────────────
CATALOG_NAME   = "rag_project"       # Catalog dédié au projet
SCHEMA_BRONZE  = "bronze"            # Données brutes (PDFs metadata, texte brut)
SCHEMA_SILVER  = "silver"            # Texte extrait et nettoyé
SCHEMA_GOLD    = "gold"              # Chunks + embeddings
VOLUME_NAME    = "raw_pdfs"          # Volume Unity Catalog pour stocker les PDFs

# Chemin Unity Catalog du Volume
VOLUME_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_BRONZE}/{VOLUME_NAME}"

print(f"""
📦 Catalog   : {CATALOG_NAME}
🥉 Bronze    : {SCHEMA_BRONZE}
🥈 Silver    : {SCHEMA_SILVER}
🥇 Gold      : {SCHEMA_GOLD}
📂 Volume    : {VOLUME_PATH}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Création du Catalog Unity Catalog
# MAGIC
# MAGIC > ℹ️ Sur Databricks trial, tu es `account admin` donc tu peux créer un catalog.
# MAGIC > Si tu as une erreur de permissions, utilise `main` comme catalog et saute cette cellule.

# COMMAND ----------

# Création du Catalog (idempotent grâce à IF NOT EXISTS)
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")
spark.sql(f"USE CATALOG {CATALOG_NAME}")
print(f"✅ Catalog '{CATALOG_NAME}' prêt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Création des Schemas (Bronze / Silver / Gold)
# MAGIC
# MAGIC On suit l'architecture **Medallion** :
# MAGIC - **Bronze** : données brutes, immuables (PDFs metadata + texte brut non nettoyé)
# MAGIC - **Silver** : texte extrait, nettoyé, structuré par page
# MAGIC - **Gold** : chunks découpés + embeddings vectoriels prêts pour le Vector Search

# COMMAND ----------

for schema in [SCHEMA_BRONZE, SCHEMA_SILVER, SCHEMA_GOLD]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{schema}")
    print(f"✅ Schema '{CATALOG_NAME}.{schema}' créé")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Création du Volume pour stocker les PDFs bruts
# MAGIC
# MAGIC Un **Volume Unity Catalog** est un espace de stockage de fichiers géré par UC.
# MAGIC Il remplace DBFS pour les nouveaux projets et offre une gouvernance fine.

# COMMAND ----------

spark.sql(f"""
    CREATE VOLUME IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_BRONZE}.{VOLUME_NAME}
""")
print(f"✅ Volume créé : {VOLUME_PATH}")

# Vérification
dbutils.fs.ls(VOLUME_PATH)
print("📂 Volume accessible et vide — prêt pour l'upload des PDFs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Génération de PDFs samples
# MAGIC
# MAGIC On va créer **5 PDFs simulant des rapports financiers** avec `fpdf2`.
# MAGIC Dans un projet réel, tu uploaderais tes vrais PDFs dans le Volume via :
# MAGIC - L'UI Databricks (Catalog > Volume > Upload)
# MAGIC - `dbutils.fs.cp("file:/local/path", VOLUME_PATH + "/fichier.pdf")`
# MAGIC - Une pipeline de streaming depuis S3/ADLS

# COMMAND ----------

# MAGIC %pip install fpdf2 --quiet

# COMMAND ----------

# Restart le Python interpreter après l'install pip
dbutils.library.restartPython()

# COMMAND ----------

# Re-déclare les paramètres (nécessaire après restartPython)
CATALOG_NAME   = "rag_project"
SCHEMA_BRONZE  = "bronze"
SCHEMA_SILVER  = "silver"
SCHEMA_GOLD    = "gold"
VOLUME_NAME    = "raw_pdfs"
VOLUME_PATH    = f"/Volumes/{CATALOG_NAME}/{SCHEMA_BRONZE}/{VOLUME_NAME}"

# COMMAND ----------

from fpdf import FPDF
import os

# Contenu simulé de rapports financiers
SAMPLE_DOCS = [
    {
        "filename": "rapport_annuel_2023_acme.pdf",
        "title": "Rapport Annuel 2023 — ACME Corporation",
        "pages": [
            {
                "title": "Résumé Exécutif",
                "content": (
                    "ACME Corporation a réalisé un chiffre d'affaires de 4,2 milliards d'euros en 2023, "
                    "en hausse de 12% par rapport à 2022. La marge opérationnelle s'établit à 18,5%. "
                    "Les activités cloud ont représenté 35% des revenus totaux. "
                    "Le résultat net atteint 620 millions d'euros, soit une progression de 8% sur un an."
                )
            },
            {
                "title": "Analyse des Risques",
                "content": (
                    "Les principaux risques identifiés pour 2024 incluent la volatilité des taux de change, "
                    "l'incertitude réglementaire en Europe, et la concurrence accrue sur le segment cloud. "
                    "Le ratio dette/EBITDA est de 1,8x, jugé soutenable par le management. "
                    "Une provision de 150M€ a été constituée pour faire face aux risques cyber."
                )
            }
        ]
    },
    {
        "filename": "contrat_fournisseur_techsup_2024.pdf",
        "title": "Contrat Cadre Fournisseur — TechSup Solutions 2024",
        "pages": [
            {
                "title": "Objet du Contrat",
                "content": (
                    "Le présent contrat cadre est conclu entre ACME Corporation (le Client) et TechSup Solutions "
                    "(le Prestataire) pour la fourniture de services de maintenance informatique. "
                    "La durée du contrat est de 3 ans à compter du 1er janvier 2024. "
                    "Le montant annuel forfaitaire s'élève à 2,4 millions d'euros HT."
                )
            },
            {
                "title": "Niveaux de Service (SLA)",
                "content": (
                    "Le Prestataire s'engage à une disponibilité de 99,5% des systèmes couverts. "
                    "Le temps de réponse pour les incidents critiques (P1) est de 2 heures maximum. "
                    "Pour les incidents majeurs (P2), le délai de prise en charge est de 4 heures. "
                    "Des pénalités de 5% du mensuel sont appliquées en cas de non-respect des SLA."
                )
            }
        ]
    },
    {
        "filename": "note_strategie_ia_2024.pdf",
        "title": "Note Stratégique — Intelligence Artificielle 2024",
        "pages": [
            {
                "title": "Vision et Ambitions IA",
                "content": (
                    "ACME Corporation se fixe l'objectif d'automatiser 30% de ses processus back-office "
                    "grâce à l'IA générative d'ici fin 2025. Un budget de 80 millions d'euros est alloué "
                    "sur 2 ans pour ce programme de transformation. Les cas d'usage prioritaires sont : "
                    "l'analyse de contrats, le support client, et l'analyse financière prédictive."
                )
            },
            {
                "title": "Roadmap Technique",
                "content": (
                    "Phase 1 (Q1-Q2 2024) : déploiement d'un chatbot RAG sur les documents internes. "
                    "Phase 2 (Q3 2024) : fine-tuning d'un LLM propriétaire sur les données métier. "
                    "Phase 3 (Q4 2024 - 2025) : industrialisation et passage à l'échelle. "
                    "L'infrastructure retenue est Databricks sur AWS, avec MLflow pour le tracking."
                )
            }
        ]
    },
    {
        "filename": "rapport_audit_conformite_2023.pdf",
        "title": "Rapport d'Audit Conformité RGPD 2023",
        "pages": [
            {
                "title": "Résultats de l'Audit",
                "content": (
                    "L'audit annuel RGPD réalisé en novembre 2023 a identifié 3 non-conformités mineures "
                    "et 0 non-conformité majeure. Les données personnelles de 2,1 millions de clients "
                    "sont traitées dans le respect du règlement européen. "
                    "Un DPO (Data Protection Officer) est en poste depuis 2021."
                )
            },
            {
                "title": "Plan de Remédiation",
                "content": (
                    "Les non-conformités mineures portent sur la durée de rétention des logs (12 mois au lieu de 6), "
                    "l'absence de chiffrement sur 2 bases de données de test, et des mentions légales "
                    "à mettre à jour sur le site web. Le plan de remédiation prévoit une correction "
                    "avant le 31 mars 2024 avec budget alloué de 120 000 euros."
                )
            }
        ]
    },
    {
        "filename": "budget_previsionnel_2024.pdf",
        "title": "Budget Prévisionnel 2024",
        "pages": [
            {
                "title": "Hypothèses Budgétaires",
                "content": (
                    "Le budget 2024 repose sur une croissance des revenus de 10%, portée par l'expansion "
                    "en Amérique du Nord et le lancement de 3 nouveaux produits SaaS. "
                    "Les charges d'exploitation sont attendues en hausse de 7%, principalement sous l'effet "
                    "de la masse salariale (+5%) et des investissements cloud (+15%)."
                )
            },
            {
                "title": "Répartition par Division",
                "content": (
                    "Division Cloud & Data : budget de 1,8 milliard d'euros (+18%). "
                    "Division Services Professionnels : budget de 900 millions d'euros (+5%). "
                    "Division Hardware : budget de 600 millions d'euros (-3%, marché mature). "
                    "Les investissements R&D représentent 12% du budget total, soit 396 millions d'euros."
                )
            }
        ]
    }
]

def create_sample_pdf(doc_info: dict, output_path: str):
    """Génère un PDF multi-pages simulant un document d'entreprise."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for i, page in enumerate(doc_info["pages"]):
        pdf.add_page()
        # En-tête
        pdf.set_font("Helvetica", style="B", size=16)
        pdf.cell(0, 12, doc_info["title"], ln=True, align="C")
        pdf.ln(4)
        # Titre de section
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(0, 10, f"{i+1}. {page['title']}", ln=True)
        pdf.ln(2)
        # Corps du texte
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 7, page["content"])
        pdf.ln(6)
        # Pied de page
        pdf.set_y(-20)
        pdf.set_font("Helvetica", style="I", size=8)
        pdf.cell(0, 6, f"Page {i+1} | Confidentiel — Usage interne uniquement", align="C")

    pdf.output(output_path)

# Génération des PDFs dans un dossier tmp local puis copie dans le Volume
local_tmp = "/tmp/sample_pdfs"
os.makedirs(local_tmp, exist_ok=True)

for doc in SAMPLE_DOCS:
    local_path = f"{local_tmp}/{doc['filename']}"
    create_sample_pdf(doc, local_path)
    # Copie vers le Volume Unity Catalog
    volume_dest = f"{VOLUME_PATH}/{doc['filename']}"
    dbutils.fs.cp(f"file:{local_path}", volume_dest)
    print(f"✅ Uploadé : {doc['filename']}")

print(f"\n📂 Tous les PDFs sont dans le Volume : {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Vérification du Volume

# COMMAND ----------

files = dbutils.fs.ls(VOLUME_PATH)
print(f"📂 {len(files)} fichiers dans le Volume :\n")
for f in files:
    size_kb = f.size / 1024
    print(f"  📄 {f.name:<45} {size_kb:.1f} KB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Stockage de la clé OpenAI dans Databricks Secrets
# MAGIC
# MAGIC > ⚠️ **Ne jamais hardcoder une clé API dans un notebook !**
# MAGIC >
# MAGIC > Databricks Secrets permet de stocker des secrets chiffrés accessibles uniquement à l'exécution.
# MAGIC >
# MAGIC > ### Setup (à faire UNE FOIS dans le terminal / Databricks CLI) :
# MAGIC >
# MAGIC > ```bash
# MAGIC > # 1. Installer la CLI Databricks
# MAGIC > pip install databricks-cli
# MAGIC >
# MAGIC > # 2. Authentification
# MAGIC > databricks configure --token
# MAGIC > # → Entrer l'URL de ton workspace et ton Personal Access Token
# MAGIC >
# MAGIC > # 3. Créer un secret scope
# MAGIC > databricks secrets create-scope --scope rag_project
# MAGIC >
# MAGIC > # 4. Stocker la clé OpenAI
# MAGIC > databricks secrets put --scope rag_project --key openai_api_key
# MAGIC > # → Coller ta clé OpenAI quand demandé
# MAGIC > ```
# MAGIC >
# MAGIC > Une fois fait, la clé est accessible dans les notebooks avec :
# MAGIC > ```python
# MAGIC > import os
# MAGIC > os.environ["OPENAI_API_KEY"] = dbutils.secrets.get("rag_project", "openai_api_key")
# MAGIC > ```

# COMMAND ----------

# !pip install databricks-cli

# COMMAND ----------

!databricks configure --token --host https://dbc-8986c94a-0283.cloud.databricks.com

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# Test de récupération du secret (fonctionnera après avoir exécuté le setup CLI ci-dessus)
try:
    import os
    os.environ["OPENAI_API_KEY"] = dbutils.secrets.get("rag_project", "openai_api_key")
    print("✅ Clé OpenAI récupérée depuis Databricks Secrets")
    print(f"   Valeur masquée : {os.environ['OPENAI_API_KEY'][:8]}...")
except Exception as e:
    print("⚠️  Secret non trouvé — exécute d'abord le setup CLI décrit ci-dessus")
    print(f"   Erreur : {e}")
    print("\n   Pour les tests rapides uniquement (⚠️ NE PAS committer) :")
    print("   os.environ['OPENAI_API_KEY'] = 'sk-...'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Récapitulatif & Validation finale

# COMMAND ----------

print("=" * 60)
print("✅ ÉTAPE 1 TERMINÉE — Récapitulatif")
print("=" * 60)

print(f"\n📦 Unity Catalog")
spark.sql(f"SHOW SCHEMAS IN {CATALOG_NAME}").show()

print(f"\n📂 Volume {VOLUME_PATH}")
files = dbutils.fs.ls(VOLUME_PATH)
print(f"   → {len(files)} PDFs prêts pour l'ingestion\n")

print("""
🔜 Prochaine étape :
   Notebook 02_bronze_ingestion.py
   → AutoLoader + extraction texte PDF (pdfplumber)
   → Création de la Delta Table Bronze
""")
print("=" * 60)
