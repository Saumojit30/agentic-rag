"""RAG Orchestrator handling ingestion and agent initialization."""

from .chunker import chunk_text
from .llm import LLMClient
from .vectorstore import VectorStore
from .agent import FinancialAnalystAgent


class RAGPipeline:
    def __init__(
        self,
        llm: LLMClient | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.llm = llm or LLMClient()
        self.store = store or VectorStore()
        self.agent = FinancialAnalystAgent(self.store, self.llm)

    def ingest(self, doc_name: str, text: str) -> dict:
        """Chunk, embed, and store document text, then extract company profile metrics if new."""
        chunks, metadata = chunk_text(text)
        if not chunks:
            return {"doc_name": doc_name, "chunks": 0, "message": "empty document"}
            
        embeddings = self.llm.embed_texts(chunks)
        self.store.add_document(
            doc_name, chunks, embeddings, metadata, embedding_name=self.llm.embedding_name
        )
        
        # Check if we should populate structured company metrics
        ticker = metadata.get("ticker")
        if ticker:
            ticker = ticker.strip().upper()
            # If company profile does not exist in relational table, extract it
            if not self.store.get_company_profile(ticker):
                # Run extraction on the full text
                profile = self.llm.extract_company_profile(text)
                if profile and profile.get("ticker") != "UNKNOWN":
                    self.store.upsert_company_metrics(
                        ticker=profile["ticker"],
                        company_name=profile["company_name"],
                        sector=profile["sector"],
                        competitors=profile["competitors"],
                        revenue=profile["revenue"],
                        net_income=profile["net_income"],
                        operating_income=profile["operating_income"],
                        total_assets=profile["total_assets"],
                        total_liabilities=profile["total_liabilities"],
                        cash=profile["cash"]
                    )
                    
        return {
            "doc_name": doc_name,
            "chunks": len(chunks),
            "metadata": metadata,
            "extracted_profile": ticker and not not self.store.get_company_profile(ticker)
        }
