"""LLM Client for Embeddings, completions, streaming, and structured profile extraction."""

import hashlib
import json
import math
import re
import asyncio
from urllib.parse import urlparse
from typing import Any, AsyncGenerator

from openai import OpenAI, AsyncOpenAI

from .config import settings
from .tracer import trace


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        chat_model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.api_key = (api_key or settings.api_key).strip()
        self.base_url = base_url or settings.base_url
        self.chat_model = chat_model or settings.chat_model
        self.embedding_model = embedding_model or settings.embedding_model
        self.mock = not self.api_key
        self._client = (
            None
            if self.mock
            else OpenAI(api_key=self.api_key, base_url=self.base_url)
        )
        self._async_client = (
            None
            if self.mock
            else AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        )

    @property
    def embedding_name(self) -> str:
        if self.mock:
            return "mock-hash"
        host = urlparse(self.base_url).netloc or self.base_url
        return f"{self.embedding_model}@{host}"

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        with trace("embed_texts", {"count": len(texts), "model": self.embedding_model}) as t:
            if self.mock:
                res = [self._mock_embed(text) for text in texts]
                t["outputs"] = {"status": "mock"}
                return res
            
            response = self._client.embeddings.create(
                model=self.embedding_model, input=texts
            )
            t["tokens"] = {
                "prompt": response.usage.prompt_tokens,
                "total": response.usage.total_tokens
            }
            return [d.embedding for d in response.data]

    @staticmethod
    def _mock_embed(text: str) -> list[float]:
        vec = [0.0] * settings.embedding_dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        grams: set[str] = set()
        for token in tokens:
            grams.add(token)
            for n in (2, 3):
                grams.update(token[i : i + n] for i in range(len(token) - n + 1))
        for gram in grams:
            idx = int(hashlib.md5(gram.encode()).hexdigest(), 16) % settings.embedding_dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------
    def generate(self, system: str, user: str, json_format: bool = False) -> str:
        with trace("generate", {"json_format": json_format, "model": self.chat_model}) as t:
            if self.mock:
                t["outputs"] = {"status": "mock"}
                return "MOCK ANSWER"
            
            kwargs = {}
            if json_format:
                kwargs["response_format"] = {"type": "json_object"}
                
            response = self._client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                **kwargs
            )
            output = response.choices[0].message.content or ""
            t["outputs"] = {"length": len(output)}
            if hasattr(response, "usage") and response.usage:
                t["tokens"] = {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
            return output

    async def generate_stream(self, system: str, user: str) -> AsyncGenerator[str, None]:
        if self.mock:
            mock_text = "MOCK STREAM - LLM API key not configured."
            for w in mock_text.split(" "):
                yield w + " "
                await asyncio.sleep(0.01)
            return

        response = await self._async_client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            stream=True,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ------------------------------------------------------------------
    # Dynamic Profile Extractor (Structured Parsing)
    # ------------------------------------------------------------------
    def extract_company_profile(self, text: str) -> dict[str, Any]:
        """Extract structured company financials and metadata from text."""
        with trace("extract_company_profile", {"text_len": len(text)}) as t:
            if self.mock:
                # Mock parsing fallback: scan text for ticker, sector, and numbers
                ticker_match = re.search(r"ticker:\s*([A-Z]{1,5})", text, re.I)
                ticker = ticker_match.group(1).upper() if ticker_match else "UNKNOWN"
                
                name_match = re.search(r"company_name:\s*(.+)", text, re.I)
                name = name_match.group(1).strip() if name_match else f"{ticker} Corp"
                
                sector_match = re.search(r"sector:\s*(.+)", text, re.I)
                sector = sector_match.group(1).strip() if sector_match else "Technology"
                
                # Search numbers in YAML-like structure or default to mocked inputs
                def find_num(pattern: str, default: float) -> float:
                    m = re.search(pattern, text, re.I)
                    if m:
                        try:
                            # Strip commas/symbols
                            num_str = m.group(1).replace(",", "").replace("$", "").strip()
                            return float(num_str)
                        except ValueError:
                            pass
                    return default

                res = {
                    "ticker": ticker,
                    "company_name": name,
                    "sector": sector,
                    "competitors": ["AAPL", "MSFT", "NVDA", "TSLA"],
                    "revenue": find_num(r"revenue:\s*([0-9\.,]+)", 125000.0),
                    "net_income": find_num(r"net_income:\s*([0-9\.,]+)", 25000.0),
                    "operating_income": find_num(r"operating_income:\s*([0-9\.,]+)", 35000.0),
                    "total_assets": find_num(r"total_assets:\s*([0-9\.,]+)", 280000.0),
                    "total_liabilities": find_num(r"total_liabilities:\s*([0-9\.,]+)", 150000.0),
                    "cash": find_num(r"cash:\s*([0-9\.,]+)", 40000.0)
                }
                t["outputs"] = res
                return res

            system_prompt = """You are a financial parsing agent. 
Given a text document, extract key corporate metadata and structured financials.
Represent numerical metrics in millions of USD (e.g. 10 billion is 10000).

Your response must be a single JSON object matching this schema:
{
  "ticker": "string (uppercase ticker, e.g. MSFT)",
  "company_name": "string (e.g. Microsoft Corporation)",
  "sector": "string (e.g. Technology)",
  "competitors": ["string" (list of tickers of direct competitors)],
  "revenue": number (total revenue in millions USD),
  "net_income": number (net income in millions USD),
  "operating_income": number (operating income in millions USD),
  "total_assets": number (total assets in millions USD),
  "total_liabilities": number (total liabilities in millions USD),
  "cash": number (total cash and cash equivalents in millions USD)
}
"""
            # Truncate text context to prevent token overflows
            snippet = text[:15000]
            try:
                response_text = self.generate(system_prompt, f"Document Content Snippet:\n\n{snippet}", json_format=True)
                res = json.loads(response_text)
                
                # Validate schema fields and clean
                cleaned = {
                    "ticker": str(res.get("ticker", "UNKNOWN")).upper().strip(),
                    "company_name": str(res.get("company_name", "Unknown Company")).strip(),
                    "sector": str(res.get("sector", "Other")).strip(),
                    "competitors": [str(c).upper().strip() for c in res.get("competitors", []) if c],
                    "revenue": float(res.get("revenue", 0.0)),
                    "net_income": float(res.get("net_income", 0.0)),
                    "operating_income": float(res.get("operating_income", 0.0)),
                    "total_assets": float(res.get("total_assets", 0.0)),
                    "total_liabilities": float(res.get("total_liabilities", 0.0)),
                    "cash": float(res.get("cash", 0.0))
                }
                t["outputs"] = cleaned
                return cleaned
            except Exception as e:
                t["error"] = str(e)
                # Fail-safe default
                return {
                    "ticker": "UNKNOWN",
                    "company_name": "Extraction Failed",
                    "sector": "Unknown",
                    "competitors": [],
                    "revenue": 0.0,
                    "net_income": 0.0,
                    "operating_income": 0.0,
                    "total_assets": 0.0,
                    "total_liabilities": 0.0,
                    "cash": 0.0
                }
