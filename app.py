"""Native Gradio application for RAGA (Corporate Intelligence & Multi-Persona Agentic RAG).
Optimized for Hugging Face Spaces Gradio SDK.
"""

import json
import asyncio
import os
import pandas as pd
import gradio as gr

from backend.app.config import settings
from backend.app.llm import LLMClient
from backend.app.vectorstore import VectorStore
from backend.app.guardrails import FinancialGuardrails
from backend.app.agent import FinancialAnalystAgent
from backend.app.pipeline import RAGPipeline

# Initialize shared components
store = VectorStore()
llm = LLMClient()
guardrails = FinancialGuardrails(llm_client=llm)
pipeline = RAGPipeline(llm=llm, store=store)

# Seed database on startup if empty
try:
    pipeline.seed_data()
except Exception as e:
    print(f"Seeding info: {e}")


async def chat_stream(message: str, history: list, persona: str, api_key: str):
    """Processes user queries through the ReAct Agent and streams responses into the Gradio chatbot."""
    if not message.strip():
        return

    # Dynamic client configuration if custom API key is supplied
    active_agent = pipeline.agent
    active_guard = guardrails
    if api_key and api_key.strip():
        dynamic_llm = LLMClient(api_key=api_key.strip())
        active_guard = FinancialGuardrails(llm_client=dynamic_llm)
        active_agent = FinancialAnalystAgent(store=store, llm=dynamic_llm)

    # 1. Guardrail Validation Check
    validation = active_guard.validate_prompt(message)
    if not validation["safe"]:
        rejection_msg = f"🛡️ **Security Alert**: {validation['reason']}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": rejection_msg})
        yield history, ""
        return

    # Add user message to history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    thoughts = []
    final_answer = ""
    session_id = "default_gradio_session"

    async for event in active_agent.run(message, session_id=session_id, persona=persona):
        lines = event.strip().split("\n")
        event_type = None
        event_data = {}

        for line in lines:
            if line.startswith("event:"):
                event_type = line.replace("event:", "").strip()
            elif line.startswith("data:"):
                try:
                    event_data = json.loads(line.replace("data:", "").strip())
                except ValueError:
                    event_data = {}

        if event_type == "thought":
            thoughts.append(f"> 💭 **Thought**: {event_data.get('text', '')}")
        elif event_type == "tool_call":
            thoughts.append(f"> ⚙️ **Tool Call**: `{event_data.get('name')}` with args `{json.dumps(event_data.get('args'))}`")
        elif event_type == "observation":
            obs_preview = str(event_data.get('output', ''))[:300]
            thoughts.append(f"> 🔍 **Observation**: *{obs_preview}...*")
        elif event_type == "token":
            final_answer += event_data.get("token", "")
        elif event_type == "error":
            final_answer += f"\n\n❌ **Error**: {event_data.get('message')}"

        # Format markdown output
        trace_md = "\n\n".join(thoughts)
        full_content = ""
        if trace_md:
            full_content += f"<details open><summary>🧠 <b>Agent Reasoning Trace</b></summary>\n\n{trace_md}\n</details>\n\n"
        if final_answer:
            full_content += f"### 📊 Response\n{final_answer}"
        elif not trace_md:
            full_content = "⏳ *Agent is planning and analyzing records...*"

        history[-1]["content"] = full_content
        yield history, ""


def get_companies_dataframe():
    """Fetches formatted metrics table for the Structured Registry tab."""
    companies = store.get_all_company_metrics()
    if not companies:
        return pd.DataFrame()
    df = pd.DataFrame(companies)
    cols = ["ticker", "company_name", "sector", "revenue", "net_income", "operating_income", "cash", "total_assets", "total_liabilities", "competitors"]
    existing_cols = [c for c in cols if c in df.columns]
    df_display = df[existing_cols].copy()
    df_display.columns = [c.replace("_", " ").title() for c in df_display.columns]
    return df_display


def upsert_company(ticker, name, sector, revenue, net_income, op_income, cash, assets, liab, competitors_str):
    """Inserts or updates a company profile in SQLite."""
    if not ticker or not name:
        return "❌ Error: Ticker and Company Name are required.", get_companies_dataframe()
    try:
        competitors = [c.strip().upper() for c in competitors_str.split(",") if c.strip()]
        store.upsert_company_metrics(
            ticker=ticker.upper().strip(),
            company_name=name.strip(),
            sector=sector.strip(),
            competitors=competitors,
            revenue=float(revenue or 0),
            net_income=float(net_income or 0),
            operating_income=float(op_income or 0),
            total_assets=float(assets or 0),
            total_liabilities=float(liab or 0),
            cash=float(cash or 0)
        )
        return f"✅ Successfully saved profile for {ticker.upper()}!", get_companies_dataframe()
    except Exception as e:
        return f"❌ Error saving profile: {str(e)}", get_companies_dataframe()


def get_documents_dataframe():
    """Fetches formatted documents list for Document Explorer tab."""
    docs = store.get_all_documents()
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    return df[["name", "chunks", "chars", "metadata"]]


def load_template(option):
    """Fills the ingestion text area with sample filings."""
    if option == "Netflix FY25 (NFLX)":
        name = "nflx_10k_fy25.md"
        text = """---
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
1. Content Capital Commitments: Securing sports licensing contracts represents heavy initial outlay risks.
2. Competition: Disney+, Amazon Prime Video, and Apple TV+ continue pricing pressures.
"""
        return name, text
    elif option == "Google Q4 Transcripts (GOOG)":
        name = "goog_transcript_fy25.md"
        text = """---
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
        return name, text
    return "", ""


def ingest_document(name, text, api_key):
    """Processes document ingestion and automatic LLM entity parsing."""
    if not name.strip() or not text.strip():
        return "❌ Error: Both Document Name and Content are required.", get_documents_dataframe(), get_companies_dataframe()
    try:
        active_pipeline = pipeline
        if api_key and api_key.strip():
            dynamic_llm = LLMClient(api_key=api_key.strip())
            active_pipeline = RAGPipeline(llm=dynamic_llm, store=store)

        chunks_added = active_pipeline.ingest(name=name.strip(), text=text.strip())
        msg = f"✅ Ingested {name}! Added {chunks_added} chunks to vector store."
        return msg, get_documents_dataframe(), get_companies_dataframe()
    except Exception as e:
        return f"❌ Ingestion failed: {str(e)}", get_documents_dataframe(), get_companies_dataframe()


# ------------------ GRADIO THEME & INTERFACE ------------------
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="#0d1117",
    block_background_fill="#161b22",
    block_border_width="1px",
    block_title_text_color="#58a6ff"
)

with gr.Blocks(theme=custom_theme, title="RAGA: Financial Analyst Agent") as demo:
    gr.Markdown("# 💼 RAGA: Corporate Intelligence & Multi-Persona Agent")
    gr.Markdown("Autonomous Multi-Persona ReAct Agent | SQL Registry + SEC Hybrid Vector RAG + Llama Prompt Guard")

    with gr.Tabs():
        # TAB 1: ANALYST TERMINAL
        with gr.TabItem("📈 Analyst Terminal"):
            with gr.Row():
                with gr.Column(scale=1):
                    persona_dropdown = gr.Dropdown(
                        label="Select Agent Persona",
                        choices=[
                            ("💼 RAGA (Finance Analyst)", "finance"),
                            ("💻 DevHelper (Code Planner)", "coder"),
                            ("🔍 Scholar (Research Agent)", "researcher"),
                            ("🤖 General Assistant", "general")
                        ],
                        value="finance"
                    )
                    api_key_input = gr.Textbox(
                        label="Groq / OpenAI API Key (Optional)",
                        type="password",
                        placeholder="gsk_... or sk-...",
                        info="If empty, the agent runs in Simulated Mock Mode."
                    )
                    gr.Markdown("""
                    ### Suggested Queries:
                    - *\"Calculate Tesla's Gross Margin in FY25 and outline pricing risk metrics.\"*
                    - *\"Compare Apple and Microsoft operating efficiency ratios for FY25 and compile a memo.\"*
                    - *\"What did Nvidia say about Blackwell supply chain allocations in transcripts?\"*
                    """)

                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Agent Conversation",
                        height=550,
                        type="messages",
                        show_copy_button=True
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask RAGA to calculate ratios, search SEC filings, or analyze filings...",
                            show_label=False,
                            scale=4
                        )
                        send_btn = gr.Button("Submit", variant="primary", scale=1)
                        clear_btn = gr.Button("Clear", scale=1)

                    send_btn.click(
                        chat_stream,
                        inputs=[msg_input, chatbot, persona_dropdown, api_key_input],
                        outputs=[chatbot, msg_input]
                    )
                    msg_input.submit(
                        chat_stream,
                        inputs=[msg_input, chatbot, persona_dropdown, api_key_input],
                        outputs=[chatbot, msg_input]
                    )
                    clear_btn.click(lambda: [], None, chatbot)

        # TAB 2: STRUCTURED REGISTRY
        with gr.TabItem("📊 Structured Registry"):
            gr.Markdown("### Relational Database Statistics (SQLite)")
            registry_df = gr.Dataframe(value=get_companies_dataframe, interactive=False)
            refresh_reg_btn = gr.Button("🔄 Refresh Registry Data")
            refresh_reg_btn.click(get_companies_dataframe, None, registry_df)

            gr.Markdown("#### Upsert Company Profile")
            with gr.Row():
                up_ticker = gr.Textbox(label="Ticker (e.g. AMZN)")
                up_name = gr.Textbox(label="Company Name")
                up_sector = gr.Textbox(label="Sector")
            with gr.Row():
                up_rev = gr.Number(label="Revenue ($M)")
                up_net = gr.Number(label="Net Income ($M)")
                up_op = gr.Number(label="Operating Income ($M)")
            with gr.Row():
                up_cash = gr.Number(label="Cash ($M)")
                up_assets = gr.Number(label="Total Assets ($M)")
                up_liab = gr.Number(label="Total Liabilities ($M)")
            up_comp = gr.Textbox(label="Competitors (Comma-separated tickers, e.g. MSFT, GOOG)")
            
            upsert_btn = gr.Button("Save Profile", variant="primary")
            upsert_status = gr.Markdown()
            upsert_btn.click(
                upsert_company,
                inputs=[up_ticker, up_name, up_sector, up_rev, up_net, up_op, up_cash, up_assets, up_liab, up_comp],
                outputs=[upsert_status, registry_df]
            )

        # TAB 3: DOCUMENT EXPLORER
        with gr.TabItem("📁 Document Explorer"):
            gr.Markdown("### Indexed SEC Reports & Transcripts")
            docs_df = gr.Dataframe(value=get_documents_dataframe, interactive=False)
            refresh_docs_btn = gr.Button("🔄 Refresh Documents")
            refresh_docs_btn.click(get_documents_dataframe, None, docs_df)

            gr.Markdown("#### Ingest New SEC Filing / Transcript")
            template_picker = gr.Dropdown(
                label="Load Sample Filing Template",
                choices=["-- Blank --", "Netflix FY25 (NFLX)", "Google Q4 Transcripts (GOOG)"],
                value="-- Blank --"
            )
            doc_name_in = gr.Textbox(label="Document Name (e.g. netflix_10k_fy25.md)")
            doc_text_in = gr.TextArea(label="Document Markdown / Text Content", lines=12)

            template_picker.change(load_template, inputs=[template_picker], outputs=[doc_name_in, doc_text_in])

            ingest_btn = gr.Button("Ingest Document", variant="primary")
            ingest_status = gr.Markdown()
            ingest_btn.click(
                ingest_document,
                inputs=[doc_name_in, doc_text_in, api_key_input],
                outputs=[ingest_status, docs_df, registry_df]
            )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
