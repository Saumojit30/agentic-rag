"""FastAPI backend server exposing RAG and Financial Analyst Agent routes."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Any

from .config import settings
from .rag import RAGPipeline
from .vectorstore import VectorStore
from .llm import LLMClient
from .sample_docs import populate_sample_data

app = FastAPI(title="Financial Analyst Agentic RAG API", version="1.0.0")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core services
llm = LLMClient()
store = VectorStore()
pipeline = RAGPipeline(llm=llm, store=store)

# Seed database on startup if empty
@app.on_event("startup")
async def startup_event():
    # If no documents are loaded, seed with sample data
    docs = store.list_docs()
    if not docs:
        populate_sample_data(pipeline)


# Pydantic schemas
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class IngestRequest(BaseModel):
    doc_name: str
    text: str

class CompanyProfileRequest(BaseModel):
    ticker: str
    company_name: str
    sector: str
    competitors: list[str]
    revenue: float
    net_income: float
    operating_income: float
    total_assets: float
    total_liabilities: float
    cash: float

class SessionCreate(BaseModel):
    title: str = "New Analysis Chat"


# --- REST API Endpoints ---

@app.get("/health")
def health_check(api_key: Optional[str] = None):
    # Check if a custom key is provided or if the system key is active
    has_key = (api_key and api_key.strip()) or not llm.mock
    return {"status": "ok", "mock_mode": not has_key, "model": settings.chat_model}

@app.get("/sessions")
def get_sessions():
    return store.list_sessions()

@app.post("/sessions")
def create_session(req: SessionCreate):
    import uuid
    session_id = str(uuid.uuid4())
    store.create_session(session_id, req.title)
    return {"session_id": session_id, "title": req.title}

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    store.delete_session(session_id)
    return {"status": "deleted"}

@app.get("/chat/{session_id}/history")
def get_chat_history(session_id: str):
    return store.get_session_history(session_id)

@app.get("/documents")
def get_documents():
    return store.list_docs()

@app.post("/ingest")
def ingest_document(req: IngestRequest, api_key: Optional[str] = None):
    try:
        # If dynamic API key is provided, instantiate context-specific clients
        if api_key and api_key.strip():
            dynamic_llm = LLMClient(api_key=api_key)
            dynamic_pipeline = RAGPipeline(llm=dynamic_llm, store=store)
            res = dynamic_pipeline.ingest(req.doc_name, req.text)
        else:
            res = pipeline.ingest(req.doc_name, req.text)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/companies")
def get_companies():
    return store.list_company_profiles()

@app.post("/companies")
def add_company_profile(req: CompanyProfileRequest):
    try:
        store.upsert_company_metrics(
            ticker=req.ticker,
            company_name=req.company_name,
            sector=req.sector,
            competitors=req.competitors,
            revenue=req.revenue,
            net_income=req.net_income,
            operating_income=req.operating_income,
            total_assets=req.total_assets,
            total_liabilities=req.total_liabilities,
            cash=req.cash
        )
        return {"status": "success", "ticker": req.ticker}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Streaming Agent Event Endpoint ---

@app.get("/query_stream")
async def run_query_stream(question: str, session_id: Optional[str] = None, api_key: Optional[str] = None):
    """Invokes the ReAct agent loop and streams thought process and token events."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    # If dynamic API key is provided, instantiate context-specific clients
    if api_key and api_key.strip():
        dynamic_llm = LLMClient(api_key=api_key)
        dynamic_pipeline = RAGPipeline(llm=dynamic_llm, store=store)
        return StreamingResponse(
            dynamic_pipeline.agent.run(question, session_id),
            media_type="text/event-stream"
        )
        
    return StreamingResponse(
        pipeline.agent.run(question, session_id),
        media_type="text/event-stream"
    )
