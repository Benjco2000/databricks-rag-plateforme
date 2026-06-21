"""
Databricks App — Chatbot RAG d'entreprise
Interface Streamlit connectée au Model Serving Endpoint "rag-chatbot-endpoint"
"""

import streamlit as st
import requests
import os

# ─── Configuration ────────────────────────────────────────────────────────────
SERVING_ENDPOINT_NAME = "rag-chatbot-endpoint"
DATABRICKS_HOST       = os.environ.get("DATABRICKS_HOST", "")

# Databricks Apps injecte CLIENT_ID + CLIENT_SECRET via le Service Principal
CLIENT_ID     = os.environ.get("DATABRICKS_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("DATABRICKS_CLIENT_SECRET", "")

ENDPOINT_URL = f"{DATABRICKS_HOST}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations"


# ─── Auth OAuth M2M ───────────────────────────────────────────────────────────
def get_token() -> str:
    """
    Obtient un token OAuth M2M via le Service Principal de l'App.
    Databricks Apps injecte DATABRICKS_CLIENT_ID et DATABRICKS_CLIENT_SECRET
    automatiquement — pas besoin de les hardcoder.
    """
    response = requests.post(
        f"{DATABRICKS_HOST}/oidc/v1/token",
        data={
            "grant_type":    "client_credentials",
            "scope":         "all-apis",
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        }
    )
    response.raise_for_status()
    return response.json()["access_token"]


# ─── Appel au Model Serving Endpoint ──────────────────────────────────────────
def call_rag_endpoint(query: str) -> dict:
    """
    Appelle le Model Serving Endpoint RAG via HTTP.
    1. Obtient un token OAuth M2M frais
    2. POST vers /serving-endpoints/rag-chatbot-endpoint/invocations
    3. Retourne la réponse du chatbot RAG
    """
    try:
        token = get_token()
        response = requests.post(
            ENDPOINT_URL,
            json={"dataframe_records": [{"query": query}]},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json"
            },
            timeout=120  # 2 min — nécessaire si l'endpoint est en cold start
        )
        response.raise_for_status()
        answer = response.json()["predictions"][0]
        return {"success": True, "answer": answer}

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "answer": "⏳ L'endpoint met du temps à répondre (cold start). Réessaie dans 30 secondes."
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return {
                "success": False,
                "answer": "❌ Endpoint introuvable. Vérifie que `rag-chatbot-endpoint` est bien déployé."
            }
        return {"success": False, "answer": f"❌ Erreur HTTP {e.response.status_code} : {str(e)}"}
    except Exception as e:
        return {"success": False, "answer": f"❌ Erreur : {str(e)}"}


# ─── Extraction des sources depuis la réponse ─────────────────────────────────
def extract_sources(answer: str) -> list:
    """
    Extrait les références de sources depuis la réponse du LLM.
    Le LLM est instruit de citer les sources au format [Source X].
    """
    import re
    sources = []
    matches  = re.findall(r'\[Source \d+[^\]]*\]', answer)
    sources.extend(matches)
    matches2 = re.findall(r'\([^)]*\.pdf[^)]*\)', answer)
    sources.extend(matches2)
    return list(set(sources)) if sources else []


# ─── Configuration de la page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Chatbot RAG — Documents d'entreprise",
    page_icon="🤖",
    layout="centered"
)

# ─── Style CSS custom ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .source-badge {
        background-color: #f0f2f6;
        border-left: 3px solid #4CAF50;
        padding: 0.4rem 0.8rem;
        border-radius: 4px;
        font-size: 0.85rem;
        margin: 0.2rem 0;
        color: #333;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 3px solid #ffc107;
        padding: 0.6rem 0.8rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 Chatbot RAG d'entreprise</h1>
    <p style="color: #666;">Posez vos questions sur les documents internes</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─── Initialisation de l'historique ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources_history" not in st.session_state:
    st.session_state.sources_history = {}

# ─── Affichage de l'historique des messages ───────────────────────────────────
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Affiche les sources sous les messages de l'assistant
        if message["role"] == "assistant" and i in st.session_state.sources_history:
            sources = st.session_state.sources_history[i]
            if sources:
                st.markdown("**📚 Sources citées :**")
                for src in sources:
                    st.markdown(
                        f'<div class="source-badge">📄 {src}</div>',
                        unsafe_allow_html=True
                    )

# ─── Zone de saisie ───────────────────────────────────────────────────────────
if query := st.chat_input("Posez votre question sur les documents..."):

    # Ajout du message utilisateur
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Appel à l'endpoint RAG
    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche dans les documents..."):
            result = call_rag_endpoint(query)

        answer  = result["answer"]
        sources = extract_sources(answer)

        st.markdown(answer)

        if sources:
            st.markdown("**📚 Sources citées :**")
            for src in sources:
                st.markdown(
                    f'<div class="source-badge">📄 {src}</div>',
                    unsafe_allow_html=True
                )
        elif result["success"]:
            st.markdown(
                '<div class="warning-box">ℹ️ Aucune source détectée dans la réponse</div>',
                unsafe_allow_html=True
            )

    # Sauvegarde dans l'historique
    msg_index = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.sources_history[msg_index] = sources


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Informations")

    st.markdown(f"""
    **Model Serving Endpoint**
    `{SERVING_ENDPOINT_NAME}`

    **LLM**
    Llama 3.3 70B (Foundation Model APIs)

    **Embeddings**
    BGE-small-en-v1.5 (384 dims)

    **Retrieval**
    Databricks Vector Search
    """)

    st.divider()

    # Statut de l'endpoint
    st.markdown("### 🔌 Statut de l'endpoint")
    if st.button("Vérifier le statut"):
        try:
            token    = get_token()
            response = requests.get(
                f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/{SERVING_ENDPOINT_NAME}",
                headers={"Authorization": f"Bearer {token}"}
            )
            state = response.json().get("state", {}).get("ready", "UNKNOWN")
            if state == "READY":
                st.success("✅ READY")
            else:
                st.warning(f"⚠️ {state}")
        except Exception as e:
            st.error(f"❌ {str(e)}")

    st.divider()

    # Vider la conversation
    if st.button("🗑️ Vider la conversation"):
        st.session_state.messages        = []
        st.session_state.sources_history = {}
        st.rerun()

    st.divider()

    # Questions exemples
    st.markdown("### 💡 Questions exemples")
    example_questions = [
        "Quel est le chiffre d'affaires annuel ?",
        "Quels sont les risques identifiés ?",
        "Quelles sont les obligations SLA ?",
        "Quelle est la stratégie IA pour 2024 ?",
    ]
    for q in example_questions:
        if st.button(q, key=f"ex_{q}"):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()