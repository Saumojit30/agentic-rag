"""Domain-specific financial tools for the Corporate Intelligence Agent."""

import json
from typing import Any, Optional
from .vectorstore import VectorStore
from .llm import LLMClient


class FinancialToolset:
    def __init__(self, store: VectorStore, llm: LLMClient) -> None:
        self.store = store
        self.llm = llm

    def search_sec_filings(
        self,
        query: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        doc_type: Optional[str] = None,
        alpha: float = 0.5,
    ) -> str:
        """Search corporate annual summaries (10-K), transcripts, and financial reports.
        
        Args:
            query: Semantic search query keywords.
            ticker: Company stock ticker (e.g. "AAPL", "MSFT"). Optional.
            fiscal_year: Specific year (e.g. 2024, 2025). Optional.
            doc_type: Document type: "10-K", "transcript", etc. Optional.
            alpha: Hybrid weight (0.0 lexical only, 1.0 vector only). Default 0.5.
        """
        # Embed query
        if self.llm.mock:
            query_vec = [0.0] * 1536
        else:
            query_vec = self.llm.embed_texts([query])[0]

        # Build filters
        filters = {}
        if ticker:
            filters["ticker"] = ticker.strip().upper()
        if fiscal_year:
            filters["fiscal_year"] = int(fiscal_year)
        if doc_type:
            filters["document_type"] = doc_type.strip()

        # Query database (requesting top 10 candidates for synthesis)
        candidates = self.store.search(
            query_vec=query_vec,
            lexical_query=query,
            filters=filters,
            top_k=8,
            alpha=alpha,
            embedding_name=self.llm.embedding_name
        )

        if not candidates:
            return "No matching corporate records found in vector store."

        # Format candidates into XML for agent review
        res = "Retrieved Financial Context Chunks:\n\n"
        for i, c in enumerate(candidates):
            meta = c["metadata"]
            t_str = f" ticker='{meta.get('ticker', 'N/A')}'"
            y_str = f" year='{meta.get('fiscal_year', 'N/A')}'"
            dt_str = f" type='{meta.get('document_type', 'N/A')}'"
            res += f"<chunk id='{i+1}' doc='{c['doc_name']}'{t_str}{y_str}{dt_str} score='{c['score']}'>\n"
            res += f"{c['text']}\n"
            res += f"</chunk>\n\n"
        return res

    def get_company_profile(self, ticker: str) -> str:
        """Fetch structured company metadata and key corporate metrics (Revenue, net income, cash, competitors, etc.).
        
        Args:
            ticker: The stock ticker (e.g. "AAPL", "MSFT").
        """
        ticker = ticker.strip().upper()
        profile = self.store.get_company_profile(ticker)
        if not profile:
            return f"Company profile for ticker '{ticker}' not found in structured registry."

        res = {
            "Ticker": profile["ticker"],
            "Company Name": profile["company_name"],
            "Sector": profile["sector"],
            "Competitors": profile["competitors"],
            "Revenue (in Millions USD)": profile["revenue"],
            "Net Income (in Millions USD)": profile["net_income"],
            "Operating Income (in Millions USD)": profile["operating_income"],
            "Total Assets (in Millions USD)": profile["total_assets"],
            "Total Liabilities (in Millions USD)": profile["total_liabilities"],
            "Total Cash (in Millions USD)": profile["cash"],
            "Last Updated": profile["last_updated"]
        }
        return json.dumps(res, indent=2)

    def calculate_financial_ratio(
        self,
        ticker: str,
        metric_a: str,
        metric_b: str,
        operation: str = "/"
    ) -> str:
        """Calculate financial ratios using verified balance sheet or income statement metrics.
        
        Available metrics: "revenue", "net_income", "operating_income", "total_assets", "total_liabilities", "cash".
        
        Args:
            ticker: Stock ticker of company (e.g. "AAPL").
            metric_a: First metric name.
            metric_b: Second metric name.
            operation: Mathematical operator: "/" (ratio), "*" (multiply), "+" (add), "-" (subtract). Default "/".
        """
        ticker = ticker.strip().upper()
        profile = self.store.get_company_profile(ticker)
        if not profile:
            return f"Error: Ticker '{ticker}' profile not found."

        valid_metrics = ["revenue", "net_income", "operating_income", "total_assets", "total_liabilities", "cash"]
        m_a = metric_a.strip().lower()
        m_b = metric_b.strip().lower()

        if m_a not in valid_metrics or m_b not in valid_metrics:
            return f"Error: Invalid metrics. Choose from {valid_metrics}"

        val_a = profile.get(m_a)
        val_b = profile.get(m_b)

        if val_a is None or val_b is None:
            return f"Error: Metric not available in profile."

        try:
            if operation == "/":
                if val_b == 0:
                    return "Error: Division by zero."
                res_val = val_a / val_b
            elif operation == "*":
                res_val = val_a * val_b
            elif operation == "+":
                res_val = val_a + val_b
            elif operation == "-":
                res_val = val_a - val_b
            else:
                return f"Error: Unsupported operation '{operation}'"
                
            return (
                f"Financial Calculation for {ticker}:\n"
                f"{m_a} ({val_a}) {operation} {m_b} ({val_b}) = {round(res_val, 4)}"
            )
        except Exception as e:
            return f"Error executing calculation: {str(e)}"

    def get_financial_news_sentiment(self, ticker: str, query: Optional[str] = None) -> str:
        """Simulate a web search engine to query external financial news, press releases, or analyst sentiments.
        
        Args:
            ticker: Stock ticker (e.g. "AAPL", "MSFT").
            query: Optional key search phrase.
        """
        ticker = ticker.strip().upper()
        # Predefined mock articles to act as dynamic news feed
        news_database = {
            "AAPL": [
                "[Financial Times] Apple's new AI initiative 'Apple Intelligence' drives strong initial upgrade cycle; premium margins expected to sustain through 2026.",
                "[Bloomberg] Supply chain sources indicate Apple face short-term component constraints for camera modules, though impact on holiday sales appears minimal.",
                "[Reuters] Apple declared standard quarterly dividend; strong cash position of $40B+ enables continuous share buybacks."
            ],
            "MSFT": [
                "[Wall Street Journal] Microsoft Cloud Azure growth matches targets, but heavy AI capital expenditures spark debate among macro analysts.",
                "[Yahoo Finance] Teams integration updates drive enterprise productivity growth; competitors like Slack lose market share.",
                "[Barrons] Analysts remain highly bullish on Microsoft's enterprise AI licensing revenue, forecasting 20% YoY expansion."
            ],
            "NVDA": [
                "[TechCrunch] Nvidia Hopper chips in extremely high demand; production yield rates for Blackwell architectures hit critical benchmarks ahead of schedule.",
                "[MarketWatch] NVIDIA registers records margins; supply bottlenecks remain the sole growth limitation.",
                "[Forbes] Competitive threats emerge from custom internal silicon projects at hyper-scalers, but Nvidia software CUDA moat holds strong."
            ],
            "TSLA": [
                "[EV World] Tesla gigafactories hit production records, but gross vehicle margins remain pressured by global price adjustments.",
                "[Bloomberg] Autopilot regulatory review progresses in Europe; full release expected in 2026.",
                "[Reuters] Energy storage storage division grows 120% YoY, compensating for automobile segment headwinds."
            ]
        }

        articles = news_database.get(ticker, [f"[Market News] External news regarding {ticker} indicates normal trading range with stable operations."])
        
        res = f"Simulated Web Search results for '{ticker}' news and sentiment:\n\n"
        for i, art in enumerate(articles):
            res += f"{i+1}. {art}\n"
        return res

    def generate_investment_memo(self, ticker: str, findings: str) -> str:
        """Format research and analytical findings into a formal corporate Investment Memo.
        
        Args:
            ticker: Stock ticker (e.g. "AAPL").
            findings: Unformatted bullet points of analyst findings, ratios, and observations.
        """
        ticker = ticker.strip().upper()
        profile = self.store.get_company_profile(ticker)
        company_name = profile["company_name"] if profile else f"{ticker} Corporation"

        memo = f"""# INVESTMENT RESEARCH MEMORANDUM

**TO:** Investment Committee
**FROM:** Senior Market Intelligence Agent
**DATE:** 2026-08-21
**SUBJECT:** Financial Analysis and Recommendation: **{company_name} ({ticker})**

---

## 1. Executive Summary
This memorandum presents a detailed corporate review and investment outlook for **{company_name} ({ticker})** based on an analysis of its recent SEC filings, earnings transcripts, and core financial ratios.

## 2. Quantitative Assessment
Below are the key financial ratios and metric evaluations performed:
{findings}

## 3. Qualitative Drivers & Operational Outlook
(Synthesized from corporate filings and latest earnings calls)
- **Competitive Advantage**: Strong moat in market sector.
- **Risks & Uncertainties**: Documented supply chain exposure, capex demands, or macro market headwinds.

## 4. Investment Thesis & Recommendation
Based on the combination of quantitative ratios and qualitative transcript reviews:
- **Recommendation**: (Decision based on calculated ratios and risks)
- **Target Audience Relevance**: Suitable for growth/value portfolios.

---
*Disclaimer: Generated autonomously by the Antigravity Financial Analyst Agent. For educational and portfolio demonstration purposes only.*
"""
        return memo


# Schema definitions for Agent function calling
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_sec_filings",
            "description": "Search corporate annual summaries (10-K), transcripts, and financial reports using hybrid lexical + semantic retrieval.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "Search terms or keywords regarding the filings (e.g. 'capital expenditures', 'AI chip constraints', 'risks')."
                    },
                    "ticker": {
                        "type": "STRING",
                        "description": "Stock ticker to filter documents by (e.g. 'AAPL', 'MSFT'). Optional."
                    },
                    "fiscal_year": {
                        "type": "INTEGER",
                        "description": "Specific year of filing to filter by (e.g. 2024, 2025). Optional."
                    },
                    "doc_type": {
                        "type": "STRING",
                        "description": "Type of corporate document: '10-K' or 'transcript'. Optional."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_profile",
            "description": "Fetch structured company description and financial statements (Revenue, Net Income, Operating Income, Assets, Liabilities, Cash) from the relational store.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "ticker": {
                        "type": "STRING",
                        "description": "Stock ticker of the company (e.g. 'MSFT')."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_financial_ratio",
            "description": "Safely computes standard arithmetic operations (+, -, *, /) on company profile financial statements (such as Net Margin: net_income / revenue).",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "ticker": {
                        "type": "STRING",
                        "description": "Stock ticker of the company."
                    },
                    "metric_a": {
                        "type": "STRING",
                        "description": "Numerator/First metric. Options: 'revenue', 'net_income', 'operating_income', 'total_assets', 'total_liabilities', 'cash'."
                    },
                    "metric_b": {
                        "type": "STRING",
                        "description": "Denominator/Second metric. Options: 'revenue', 'net_income', 'operating_income', 'total_assets', 'total_liabilities', 'cash'."
                    },
                    "operation": {
                        "type": "STRING",
                        "description": "Mathematical operator. Must be '/', '*', '+', or '-'."
                    }
                },
                "required": ["ticker", "metric_a", "metric_b", "operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_news_sentiment",
            "description": "Simulates a web search engine to query external financial news feed, press releases, or general sentiment around a stock ticker.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "ticker": {
                        "type": "STRING",
                        "description": "Company stock ticker (e.g. 'NVDA')."
                    },
                    "query": {
                        "type": "STRING",
                        "description": "Optional search term for news filtering. Optional."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_investment_memo",
            "description": "Formats analyst quantitative findings and ratios into a formal Investment Committee Research Memorandum markdown structure.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "ticker": {
                        "type": "STRING",
                        "description": "Stock ticker of the company."
                    },
                    "findings": {
                        "type": "STRING",
                        "description": "Bullet points of key margins, calculated growth metrics, and observations to put inside the memo."
                    }
                },
                "required": ["ticker", "findings"]
            }
        }
    }
]
