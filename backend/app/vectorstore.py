"""SQLite vector store with metadata filtering, hybrid search, and company metrics database."""

import json
import math
import sqlite3
from pathlib import Path
from typing import Any


class VectorStore:
    def __init__(self, db_path: str | None = None) -> None:
        from .config import settings
        self._path = Path(db_path or settings.db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            # Main chunks table with metadata JSON
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_name       TEXT NOT NULL,
                    chunk_index    INTEGER NOT NULL,
                    text           TEXT NOT NULL,
                    embedding      TEXT NOT NULL,
                    embedding_name TEXT NOT NULL DEFAULT '',
                    metadata       TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            # FTS5 Virtual Table for lexical search
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                    doc_name UNINDEXED,
                    chunk_index UNINDEXED,
                    text
                )
                """
            )
            # Query Cache Table for RAG responses
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_hash     TEXT PRIMARY KEY,
                    question       TEXT NOT NULL,
                    filters_json   TEXT NOT NULL,
                    response_json  TEXT NOT NULL,
                    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Chat sessions table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id     TEXT PRIMARY KEY,
                    title          TEXT NOT NULL DEFAULT 'New Chat',
                    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Chat messages table (supports saving conversation steps & final answers)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id     TEXT NOT NULL,
                    role           TEXT NOT NULL,
                    content        TEXT NOT NULL,
                    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            # Relational Company Metrics table (structured financials)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS company_metrics (
                    ticker             TEXT PRIMARY KEY,
                    company_name       TEXT NOT NULL,
                    sector             TEXT,
                    competitors        TEXT, -- JSON list of tickers
                    revenue            REAL, -- In millions
                    net_income         REAL, -- In millions
                    operating_income   REAL, -- In millions
                    total_assets       REAL, -- In millions
                    total_liabilities  REAL, -- In millions
                    cash               REAL, -- In millions
                    last_updated       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    # ------------------------------------------------------------------
    # Ingest Documents
    # ------------------------------------------------------------------
    def add_document(
        self,
        doc_name: str,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: dict,
        embedding_name: str = "",
    ) -> None:
        self.delete_doc(doc_name)
        metadata_json = json.dumps(metadata)
        with self._connect() as conn:
            for i, (text, vec) in enumerate(zip(texts, embeddings)):
                cursor = conn.execute(
                    "INSERT INTO chunks (doc_name, chunk_index, text, embedding, embedding_name, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_name, i, text, json.dumps(vec), embedding_name, metadata_json)
                )
                chunk_id = cursor.lastrowid
                conn.execute(
                    "INSERT INTO fts_chunks (rowid, doc_name, chunk_index, text) VALUES (?, ?, ?, ?)",
                    (chunk_id, doc_name, i, text)
                )

    def delete_doc(self, doc_name: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM fts_chunks WHERE doc_name = ?", (doc_name,))
            conn.execute("DELETE FROM chunks WHERE doc_name = ?", (doc_name,))
            conn.execute("DELETE FROM query_cache")  # Invalidate cache on doc deletion

    def list_docs(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_name, COUNT(*) AS chunks, SUM(LENGTH(text)) AS chars, "
                "MAX(embedding_name) AS embedding_name, MAX(metadata) AS metadata "
                "FROM chunks GROUP BY doc_name ORDER BY doc_name"
            ).fetchall()
        return [
            {
                "name": r["doc_name"],
                "chunks": r["chunks"],
                "chars": r["chars"] or 0,
                "embedding_name": r["embedding_name"] or "",
                "metadata": json.loads(r["metadata"] or "{}"),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Retrieve Full Document Content (For tool: fetch_document_by_name)
    # ------------------------------------------------------------------
    def get_document_full_text(self, doc_name: str) -> str:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT text FROM chunks WHERE doc_name = ? ORDER BY chunk_index ASC",
                (doc_name,)
            ).fetchall()
        if not rows:
            return f"Document '{doc_name}' not found."
        return "\n".join(r["text"] for r in rows)

    # ------------------------------------------------------------------
    # Search (Hybrid Vector + Lexical with SQL metadata filtering)
    # ------------------------------------------------------------------
    def search(
        self,
        query_vec: list[float],
        lexical_query: str,
        filters: dict[str, Any],
        top_k: int,
        alpha: float = 0.5,
        embedding_name: str = "",
    ) -> list[dict]:
        """Hybrid search combining vector similarity and FTS5 BM25.
        
        alpha=1.0: Pure semantic search
        alpha=0.0: Pure lexical search
        """
        where_clauses = ["embedding_name = ?"]
        params = [embedding_name]

        # Build metadata filters (e.g., {"ticker": "AAPL", "fiscal_year": 2025})
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, dict):
                # Handle operators like {">=": 2024}
                for op, val in value.items():
                    if op in (">", ">=", "<", "<=", "="):
                        where_clauses.append(f"json_extract(metadata, '$.{key}') {op} ?")
                        params.append(val)
            else:
                where_clauses.append(f"json_extract(metadata, '$.{key}') = ?")
                params.append(value)

        where_sql = " AND ".join(where_clauses)
        
        with self._connect() as conn:
            # 1. Semantic Search
            rows = conn.execute(
                f"SELECT id, doc_name, chunk_index, text, metadata, embedding FROM chunks WHERE {where_sql}",
                params,
            ).fetchall()
            
            semantic_results = []
            for r in rows:
                sim = cosine(query_vec, json.loads(r["embedding"]))
                semantic_results.append({"id": r["id"], "score": sim, "row": r})
                
            semantic_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Rank dictionary for RRF
            semantic_ranks = {item["id"]: rank for rank, item in enumerate(semantic_results, 1)}

            # 2. Lexical Search (FTS5)
            lexical_results = []
            if lexical_query.strip():
                # Escaping matching keywords
                safe_query = ' OR '.join(f'"{w}"' for w in lexical_query.replace('"', '').split())
                if safe_query:
                    fts_sql = f"""
                        SELECT f.rowid as id, f.doc_name, f.chunk_index, f.text, bm25(fts_chunks) as rank_score, c.metadata
                        FROM fts_chunks f
                        JOIN chunks c ON f.rowid = c.id
                        WHERE fts_chunks MATCH ? AND c.{where_sql}
                        ORDER BY rank_score ASC LIMIT 100
                    """
                    fts_params = [safe_query] + params
                    lex_rows = conn.execute(fts_sql, fts_params).fetchall()
                    
                    for r in lex_rows:
                        lexical_results.append({"id": r["id"], "score": -r["rank_score"], "row": r})
            
            lexical_ranks = {item["id"]: rank for rank, item in enumerate(lexical_results, 1)}

        # 3. Hybrid Fusion
        fused_scores = {}
        all_ids = set(semantic_ranks.keys()) | set(lexical_ranks.keys())
        
        for doc_id in all_ids:
            s_rank = semantic_ranks.get(doc_id, 1000)
            l_rank = lexical_ranks.get(doc_id, 1000)
            
            score = (alpha * (1.0 / (60 + s_rank))) + ((1.0 - alpha) * (1.0 / (60 + l_rank)))
            fused_scores[doc_id] = score

        # Sort by fused score
        sorted_ids = sorted(list(all_ids), key=lambda x: fused_scores[x], reverse=True)[:top_k]
        
        # Build final results
        final_results = []
        row_lookup = {}
        for item in semantic_results:
            row_lookup[item["id"]] = item["row"]
        for item in lexical_results:
            if item["id"] not in row_lookup:
                row_lookup[item["id"]] = item["row"]
                
        for doc_id in sorted_ids:
            r = row_lookup[doc_id]
            final_results.append({
                "score": round(fused_scores[doc_id], 4),
                "semantic_rank": semantic_ranks.get(doc_id, "N/A"),
                "lexical_rank": lexical_ranks.get(doc_id, "N/A"),
                "doc_name": r["doc_name"],
                "chunk_index": r["chunk_index"],
                "text": r["text"],
                "metadata": json.loads(r["metadata"]),
            })

        return final_results

    # ------------------------------------------------------------------
    # Structured Company Metrics & Profiles
    # ------------------------------------------------------------------
    def upsert_company_metrics(
        self,
        ticker: str,
        company_name: str,
        sector: str,
        competitors: list[str],
        revenue: float,
        net_income: float,
        operating_income: float,
        total_assets: float,
        total_liabilities: float,
        cash: float
    ) -> None:
        competitors_json = json.dumps([c.strip().upper() for c in competitors])
        ticker = ticker.strip().upper()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO company_metrics (
                    ticker, company_name, sector, competitors, revenue, 
                    net_income, operating_income, total_assets, total_liabilities, cash, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (ticker, company_name, sector, competitors_json, revenue, 
                 net_income, operating_income, total_assets, total_liabilities, cash)
            )

    def get_company_profile(self, ticker: str) -> dict | None:
        ticker = ticker.strip().upper()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM company_metrics WHERE ticker = ?", (ticker,)
            ).fetchone()
        if not row:
            return None
        res = dict(row)
        res["competitors"] = json.loads(res["competitors"] or "[]")
        return res

    def list_company_profiles(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM company_metrics ORDER BY ticker ASC").fetchall()
        res = []
        for r in rows:
            d = dict(r)
            d["competitors"] = json.loads(d["competitors"] or "[]")
            res.append(d)
        return res

    # ------------------------------------------------------------------
    # Query Cache
    # ------------------------------------------------------------------
    def get_cached_query(self, query_hash: str) -> dict | None:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT response_json FROM query_cache WHERE query_hash = ?", (query_hash,)
                ).fetchone()
                if row:
                    return json.loads(row["response_json"])
        except Exception:
            pass
        return None

    def set_cached_query(self, query_hash: str, question: str, filters: dict, response: dict) -> None:
        try:
            filters_json = json.dumps(filters, sort_keys=True)
            response_json = json.dumps(response)
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO query_cache (query_hash, question, filters_json, response_json, created_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (query_hash, question, filters_json, response_json),
                )
        except Exception:
            pass

    def clear_cache(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM query_cache")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Chat Memory
    # ------------------------------------------------------------------
    def create_session(self, session_id: str, title: str = "New Analysis Chat") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO chat_sessions (session_id, title) VALUES (?, ?)",
                (session_id, title)
            )

    def list_sessions(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, title, created_at FROM chat_sessions ORDER BY created_at DESC"
            ).fetchall()
            return [{"session_id": r["session_id"], "title": r["title"], "created_at": r["created_at"]} for r in rows]

    def delete_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))

    def get_session_history(self, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,)
            ).fetchall()
            return [{"role": r["role"], "content": r["content"]} for r in rows]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )

    def update_session_title(self, session_id: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE chat_sessions SET title = ? WHERE session_id = ?", (title, session_id))


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)
