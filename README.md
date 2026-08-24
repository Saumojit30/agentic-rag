---
title: Financial Analyst Agentic RAG
emoji: 💼
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Project 05 — Financial Analyst Agentic RAG

This project implements an advanced **Corporate & Market Intelligence Agent** that combines unstructured document retrieval (SEC filings, earnings call transcripts) with a structured relational financial registry to perform complex analysis, run exact arithmetic calculations, and compile investment memos.

Unlike standard RAG pipelines that execute a static sequential workflow (e.g. rewrite -> retrieve -> generate), this system runs a dynamic **ReAct (Reasoning and Action) Agent Loop** using function calling schemas. It enables the model to autonomously plan, invoke specific tools, assess intermediate observations, and iterate until it satisfies the user's research request.

## Features
- **ReAct planning loop**: Displays step-by-step thinking traces (`Thought` $\rightarrow$ `Action` $\rightarrow$ `Observation` $\rightarrow$ `Answer`) in the UI.
- **Relational Financial Registry**: Stores company profiles and balance sheet metrics (Revenue, Net Income, Operating Margin, Cash, Debt, Sector, Competitors) in SQLite.
- **Dynamic Profile Extraction (Any Company Analysis)**: When new company documents are ingested, the system automatically runs an LLM-based parsing step to extract key financial numbers and populate the relational registry.
- **Domain-Specific Toolset**:
  - `search_sec_filings`: Hybrid vector + FTS5 lexical search on 10-K/transcripts with metadata filters.
  - `get_company_profile`: Retrieves structured profile metrics from SQLite.
  - `calculate_financial_ratio`: Performs safe arithmetic (margins, ratios, YoY growth) using exact metrics.
  - `get_financial_news_sentiment`: Simulates external web news searches.
  - `generate_investment_memo`: Formats quantitative ratios and risk factors into a professional markdown investment report.
- **Premium Analyst UI**: Dark-themed Streamlit dashboard with a live reasoning accordion, structured registry database explorer, and document ingestion panel.
- **Robust Mock Mode & API Key Bypass**: Fully operational mock seeder and rule-based agent simulator if API keys are not supplied. Visitors can input their own keys in the sidebar to run the live model, making it ideal for public deployments.

---

## Project Structure
```text
05-agentic-rag/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── agent.py         # Core ReAct agent loop
│   │   ├── chunker.py       # YAML frontmatter-aware chunker
│   │   ├── config.py        # Settings and environment loaders
│   │   ├── llm.py           # Embeddings, chat clients, and profile extractor
│   │   ├── main.py          # FastAPI routes and SSE event endpoints
│   │   ├── rag.py           # Ingestion and orchestrations
│   │   ├── sample_docs.py   # Seeder datasets (Apple, Microsoft, Nvidia, Tesla)
│   │   ├── tools.py         # Domain analyst tools and function schemas
│   │   └── tracer.py        # Latency tracing
├── streamlit_app/
│   └── app.py               # Dark-themed Streamlit terminal
├── requirements.txt         # Package dependencies
├── Dockerfile               # Multi-service container image configuration
├── start.sh                 # Entrypoint shell startup script
├── docker-compose.yml       # Local container deployment configurations
└── README.md                # This file
```

---

## Setup & Local Development

### Option A: Running with Docker (One-Click Local Run)
Make sure you have Docker installed.

1. **Start the Multi-container Application**:
   ```bash
   cd 05-agentic-rag
   docker-compose up --build
   ```
2. **Access the Application**:
   * Streamlit Frontend: `http://localhost:8501`
   * FastAPI Backend: `http://localhost:8000/docs`

---

### Option B: Manual Local Setup

#### 1. Create and Activate Virtual Environment
```bash
# Navigate to the project directory
cd 05-agentic-rag

# Create virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Launch Backend API
Start the FastAPI server on port 8000. On startup, it will automatically build schemas and seed the SQLite database with Apple, Microsoft, NVIDIA, and Tesla reports.
```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 4. Launch Streamlit UI
In a separate terminal (with virtual environment activated):
```bash
python -m streamlit run streamlit_app/app.py --server.port 8501
```
Open `http://localhost:8501` to access the analyst terminal.

---

## Agentic Reasoning Architecture

```
       [User Input Question]
                 │
                 ▼
     [Start Agent Loop (ReAct)]
                 │
                 ▼
       ┌───────────────────┐
       │   LLM Planning    │ <─── (thought text yielded)
       └─────────┬─────────┘
                 │
         [Has tool calls?]
               /   \
             YES    NO  ──► [Final Token Streaming] ──► [Done]
             /
            ▼
    [Execute Tool] ─── (tool_call event yielded)
    (get_company_profile,
     calculate_ratio,
     search_sec_filings, etc.)
            │
            ▼
    [Return Result] ─── (observation event yielded)
            │
            └─────► [Append to context and loop again]
```
