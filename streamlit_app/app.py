"""Premium dark-themed Streamlit dashboard for Corporate Intelligence & RAGA (Financial Analyst Agentic RAG)."""

import json
import streamlit as st
import httpx
import pandas as pd

import os
# Base backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAGA Analyst Terminal",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply premium dark theme styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: #0e1117;
        color: #c9d1d9;
    }
    .thought-box {
        background-color: #1f242c;
        border-left: 4px solid #1f6feb;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 4px;
        font-size: 0.95em;
    }
    .tool-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 8px 12px;
        margin: 5px 0;
        font-family: monospace;
        border-radius: 4px;
        color: #58a6ff;
    }
    .observation-box {
        background-color: #0d1117;
        border: 1px dashed #30363d;
        padding: 10px;
        margin: 5px 0 12px 0;
        font-family: monospace;
        font-size: 0.85em;
        border-radius: 4px;
        max-height: 250px;
        overflow-y: auto;
        color: #8b949e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💼 RAGA: Corporate Intelligence Terminal")
st.caption("RAGA (Retrieval-Augmented Generation Analyst) | SQL Registry + SEC Hybrid Vector RAG + Llama Prompt Guard")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Terminal Setup")
    
    # Persona selection
    persona_options = {
        "finance": "💼 RAGA (Finance)",
        "coder": "💻 DevHelper (Coding)",
        "researcher": "🔍 Scholar (Research)",
        "general": "🤖 General Assistant"
    }
    selected_persona = st.selectbox(
        "Agent Persona",
        options=list(persona_options.keys()),
        format_func=lambda x: persona_options[x]
    )
    
    # Dynamic API key entry
    custom_key = st.text_input(
        "API Key (Optional Override)", 
        type="password",
        help="Input your OpenAI or Groq API key to run live LLM agents. If empty, the system defaults to Mock Mode."
    )

    # Check backend health with optional API key
    try:
        health_params = {}
        if custom_key:
            health_params["api_key"] = custom_key
        health = httpx.get(f"{BACKEND_URL}/health", params=health_params).json()
        is_mock = health.get("mock_mode", True)
        active_model = health.get("model", "gpt OS 120 B")
    except Exception:
        st.error("Cannot connect to FastAPI backend on port 8000. Please start the backend server.")
        st.stop()

    if is_mock:
        st.warning("⚠️ Mock Mode: No API key configured. Executing simulated ReAct planning traces.")
    else:
        st.success(f"🟢 Active Model: `{active_model}`")

    st.subheader("Chat Sessions")
    
    # List sessions
    try:
        sessions = httpx.get(f"{BACKEND_URL}/sessions").json()
    except Exception:
        sessions = []

    # Session Selector
    session_titles = {s["session_id"]: s["title"] for s in sessions}
    session_ids = list(session_titles.keys())
    
    if not session_ids:
        # Create default session
        try:
            new_sess = httpx.post(f"{BACKEND_URL}/sessions", json={"title": "Analysis 1"}).json()
            session_ids = [new_sess["session_id"]]
            session_titles[new_sess["session_id"]] = new_sess["title"]
        except Exception:
            session_ids = ["default"]
            session_titles["default"] = "Default Chat"

    selected_session = st.selectbox(
        "Select Session",
        options=session_ids,
        format_func=lambda x: session_titles.get(x, x)
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Chat", use_container_width=True):
            try:
                new_title = f"Analysis {len(session_ids) + 1}"
                new_sess = httpx.post(f"{BACKEND_URL}/sessions", json={"title": new_title}).json()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    with col2:
        if st.button("Delete Chat", use_container_width=True):
            if selected_session:
                try:
                    httpx.delete(f"{BACKEND_URL}/sessions/{selected_session}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.markdown("""
    ### Suggested Queries:
    - *\"Calculate Tesla's Gross margins in FY25 and outline vehicle pricing risk metrics.\"*
    - *\"Compare Apple and Microsoft operating efficiency ratios for FY25 and write an investment memo.\"*
    - *\"What is Nvidia's current revenue, and what did they say about Blackwell supply chain allocations in transcripts?\"*
    """)


# --- Tabs Layout ---
tab_terminal, tab_registry, tab_docs = st.tabs([
    "📈 Analyst Terminal",
    "📊 Structured Registry",
    "📁 Document Explorer"
])

# ----------------- TAB 1: ANALYST TERMINAL -----------------
with tab_terminal:
    st.subheader("Autonomous Analyst Conversation")
    
    # Load session message history
    chat_history = []
    if selected_session:
        try:
            chat_history = httpx.get(f"{BACKEND_URL}/chat/{selected_session}/history").json()
        except Exception:
            pass

    # Display chat history
    for msg in chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if question := st.chat_input("Ask RAGA to search filings, run metrics, remember preferences, or compile memos..."):
        # Write user question
        with st.chat_message("user"):
            st.markdown(question)

        # Agent thinking and streaming container
        with st.chat_message("assistant"):
            st.write("Initializing agent execution loop...")
            
            # Placeholders
            trace_container = st.container()
            answer_title = st.empty()
            answer_container = st.empty()

            thoughts = []
            final_answer = ""
            active_expander = None
            
            # SSE streaming parser
            try:
                params = {"question": question, "session_id": selected_session, "persona": selected_persona}
                if custom_key:
                    params["api_key"] = custom_key
                with httpx.stream("GET", f"{BACKEND_URL}/query_stream", params=params, timeout=120.0) as r:
                    current_event = None
                    for line in r.iter_lines():
                        if not line:
                            continue
                        
                        if line.startswith("event:"):
                            current_event = line.replace("event:", "").strip()
                        elif line.startswith("data:") and current_event:
                            data_str = line.replace("data:", "").strip()
                            try:
                                data = json.loads(data_str)
                            except ValueError:
                                data = {}

                            # Handle Event Types
                            if current_event == "thought":
                                with trace_container:
                                    st.markdown(f"<div class='thought-box'><b>Thought</b>: {data.get('text')}</div>", unsafe_allow_html=True)
                            
                            elif current_event == "tool_call":
                                with trace_container:
                                    st.markdown(f"<div class='tool-box'>⚙️ Executing Tool: <b>{data.get('name')}</b>({data.get('args')})</div>", unsafe_allow_html=True)
                            
                            elif current_event == "observation":
                                with trace_container:
                                    with st.expander("🔍 View Tool Observation Output", expanded=False):
                                        st.markdown(f"<div class='observation-box'>{data.get('output')}</div>", unsafe_allow_html=True)
                            
                            elif current_event == "token":
                                if not final_answer:
                                    answer_title.markdown("### Final Report Summary")
                                final_answer += data.get("token", "")
                                answer_container.markdown(final_answer)
                                
                            elif current_event == "error":
                                st.error(f"Security/Agent Error: {data.get('message')}")
                                
                            elif current_event == "done":
                                st.success("Analysis complete!")
                                
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")


# ----------------- TAB 2: STRUCTURED REGISTRY -----------------
with tab_registry:
    st.subheader("Tabular Company Metrics Registry")
    st.caption("Relational database stats used by the financial ratio calculator tool.")

    # Fetch company metrics
    try:
        companies = httpx.get(f"{BACKEND_URL}/companies").json()
    except Exception:
        companies = []

    if companies:
        df = pd.DataFrame(companies)
        # ReOrder columns for readability
        cols = ["ticker", "company_name", "sector", "revenue", "net_income", "operating_income", "cash", "total_assets", "total_liabilities", "competitors"]
        existing_cols = [c for c in cols if c in df.columns]
        df_display = df[existing_cols].copy()
        df_display.columns = [c.replace("_", " ").title() for c in df_display.columns]
        
        st.dataframe(df_display, hide_index=True, use_container_width=True)
    else:
        st.info("No company profiles found in structured registry.")

    st.divider()
    
    # Manual Upsert Form
    st.subheader("Upsert Company Profile Metrics")
    with st.form("upsert_profile_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker = st.text_input("Stock Ticker (e.g. AMZN)", max_chars=5).upper().strip()
            name = st.text_input("Company Name (e.g. Amazon Inc.)")
            sector = st.text_input("Sector (e.g. Consumer Cyclical)")
        with col2:
            revenue = st.number_input("Revenue (Millions USD)", min_value=0.0, format="%.2f")
            net_income = st.number_input("Net Income (Millions USD)", format="%.2f")
            operating_income = st.number_input("Operating Income (Millions USD)", format="%.2f")
        with col3:
            cash = st.number_input("Cash & Equivalents (Millions USD)", min_value=0.0, format="%.2f")
            assets = st.number_input("Total Assets (Millions USD)", min_value=0.0, format="%.2f")
            liabilities = st.number_input("Total Liabilities (Millions USD)", min_value=0.0, format="%.2f")
            
        competitors_str = st.text_input("Competitors (Comma-separated tickers, e.g. MSFT, AAPL)")
        
        submit = st.form_submit_button("Upsert Profile")
        if submit:
            if not ticker or not name:
                st.error("Ticker and Company Name are required.")
            else:
                payload = {
                    "ticker": ticker,
                    "company_name": name,
                    "sector": sector,
                    "competitors": [c.strip().upper() for c in competitors_str.split(",") if c.strip()],
                    "revenue": revenue,
                    "net_income": net_income,
                    "operating_income": operating_income,
                    "total_assets": assets,
                    "total_liabilities": liabilities,
                    "cash": cash
                }
                try:
                    res = httpx.post(f"{BACKEND_URL}/companies", json=payload)
                    if res.status_code == 200:
                        st.success(f"Profile for {ticker} successfully saved/updated!")
                        st.rerun()
                    else:
                        st.error(f"Error upserting profile: {res.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")


# ----------------- TAB 3: DOCUMENT EXPLORER -----------------
with tab_docs:
    st.subheader("SEC Filing & Transcript Index")
    
    # List documents
    try:
        documents = httpx.get(f"{BACKEND_URL}/documents").json()
    except Exception:
        documents = []

    if documents:
        df_docs = pd.DataFrame(documents)
        df_docs = df_docs[["name", "chunks", "chars", "metadata"]]
        st.dataframe(df_docs, use_container_width=True)
    else:
        st.info("No documents uploaded yet.")

    st.divider()
    
    st.subheader("Ingest New SEC Report / Earnings Call")
    st.markdown("Upload or paste corporate filings. *If the ticker matches a company not in the Structured Registry, the LLM will automatically parse key balance sheet numbers and populate the table!*")
    
    ingest_name = st.text_input("Document Name (e.g. netflix_10k_fy25.md)", placeholder="netflix_10k_fy25.md")
    
    # Default template seeder for easy dynamic testing
    template_option = st.selectbox(
        "Seeding Templates (Optional)",
        ["-- Blank --", "Netflix FY25 (NFLX)", "Google Q4 Transcripts (GOOG)"]
    )
    
    default_text = ""
    if template_option == "Netflix FY25 (NFLX)":
        ingest_name = "nflx_10k_fy25.md"
        default_text = """---
ticker: NFLX
fiscal_year: 2025
document_type: 10-K
---
# Netflix Inc. Form 10-K Summary (FY25)

## Business Summary
Netflix Inc. remains the world's leading streaming entertainment service. In FY25, Netflix focused heavily on advertising tiers, cloud gaming integrations, and sports content acquisitions.

## Revenue & Profitability
Netflix reported revenue of $38,000 million (up from $33,000 million in FY24). Operating income reached $9,500 million, Net income was $8,000 million. 

## Balance Sheet
Total cash position was reported at $6,200 million. Total assets were $52,000 million, while total liabilities stood at $30,000 million.

## Risks
1. **Content Capital Commitments**: Securing sports licensing contracts represents heavy initial outlay risks.
2. **Competition**: Disney+, Amazon Prime Video, and Apple TV+ continue pricing pressures.
"""
    elif template_option == "Google Q4 Transcripts (GOOG)":
        ingest_name = "goog_transcript_fy25.md"
        default_text = """---
ticker: GOOG
fiscal_year: 2025
document_type: transcript
---
# Alphabet Inc. Q4 FY25 Earnings Call Transcript

## Executives Present
- Sundar Pichai, CEO
- Ruth Porat, President and CFO

## Highlights
"Our subscription revenues reached $15,000 million for the year. Cloud division revenue expanded by 30% YoY, crossing critical scale milestones. Cash balance stands at $110,000 million, providing massive flexibilities."
"""

    ingest_text = st.text_area("Document Markdown / Text Content", value=default_text, height=300)
    
    if st.button("Ingest Document"):
        if not ingest_name.strip() or not ingest_text.strip():
            st.error("Both document name and text content are required.")
        else:
            with st.spinner("Processing document (splitting, embedding, parsing)..."):
                try:
                    payload = {"doc_name": ingest_name, "text": ingest_text}
                    ingest_params = {}
                    if custom_key:
                        ingest_params["api_key"] = custom_key
                    res = httpx.post(f"{BACKEND_URL}/ingest", json=payload, params=ingest_params).json()
                    
                    st.success(f"Successfully indexed document! Added {res.get('chunks')} chunks.")
                    if res.get("extracted_profile"):
                        st.info("💡 LLM Profile Extractor automatically identified new ticker and populated structured financial stats in registry!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
