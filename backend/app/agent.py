"""Core ReAct Agent executor loop yielding SSE trace events."""

import json
import asyncio
from typing import AsyncGenerator, Any
from .llm import LLMClient
from .vectorstore import VectorStore
from .tools import FinancialToolset, TOOLS_SCHEMA
from .config import settings

SYSTEM_PROMPT = """You are a Senior Corporate Financial Analyst Agent. Your objective is to answer research requests using corporate financial records and quantitative tools.

RULES:
1. **Planning & Thought**: Before making any tool call or answering, write a thorough explanation of your reasoning inside `<thought>...</thought>` tags. Planning is critical.
2. **Quantitative Precision**: Never make up or guess financial figures. Always use `get_company_profile` to retrieve structured financials and `calculate_financial_ratio` to compute ratios.
3. **Information Retrieval**: Use `search_sec_filings` to search annual reports and earnings call transcripts for qualitative details (e.g. risks, supply chains, growth strategies).
4. **News Sentiment**: Use `get_financial_news_sentiment` to search for latest external market updates.
5. **Synthesis**: Use `generate_investment_memo` to format your findings into a formal report when requested.
6. **Iterate**: Analyze tool observations to determine if you need further searches or calculations. Stop once you have fully resolved the user's question.
"""


class FinancialAnalystAgent:
    def __init__(self, store: VectorStore, llm: LLMClient) -> None:
        self.store = store
        self.llm = llm
        self.toolset = FinancialToolset(store, llm)

    async def run(self, question: str, session_id: str | None = None) -> AsyncGenerator[str, None]:
        """Runs the ReAct agent loop and yields SSE events.
        
        Events format:
        event: thought | tool_call | observation | token | done
        data: {json_payload}
        """
        # Fetch session history if available
        history = []
        if session_id:
            history = await asyncio.to_thread(self.store.get_session_history, session_id)

        # ------------------ MOCK RUNNER ------------------
        if self.llm.mock:
            async for event in self._run_mock_agent(question, session_id):
                yield event
            return

        # ------------------ REAL AGENT RUNNER ------------------
        # Setup conversation history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Append last 4 messages from session history to stay context-aware
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        messages.append({"role": "user", "content": question})

        iteration = 0
        final_answer = ""
        
        while iteration < settings.max_agent_iterations:
            iteration += 1
            
            # 1. Call LLM with tool schemas
            try:
                response = await asyncio.to_thread(
                    self.llm._client.chat.completions.create,
                    model=self.llm.chat_model,
                    messages=messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=0.0
                )
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'LLM call failed: {str(e)}'})}\n\n"
                return

            assistant_msg = response.choices[0].message
            content = assistant_msg.content or ""
            tool_calls = assistant_msg.tool_calls or []

            # 2. Extract and yield thought if present
            thought_text = ""
            if "<thought>" in content:
                # Extract text between <thought> and </thought>
                parts = content.split("</thought>")
                thought_text = parts[0].replace("<thought>", "").strip()
                # Yield thought
                yield f"event: thought\ndata: {json.dumps({'text': thought_text})}\n\n"
                
                # If there's content after </thought>, keep it as partial final answer
                if len(parts) > 1 and parts[1].strip():
                    content = parts[1].strip()
            elif content.strip() and not tool_calls:
                # If it's just raw text, treat it as thought or start of answer
                yield f"event: thought\ndata: {json.dumps({'text': content.strip()})}\n\n"

            # Append assistant message to context
            messages.append(assistant_msg)

            # 3. Handle Tool Calls
            if tool_calls:
                # If the agent returned tool calls, we execute them one by one
                tool_msg_list = []
                for tool_call in tool_calls:
                    func_name = tool_call.function.name
                    func_args_str = tool_call.function.arguments
                    call_id = tool_call.id
                    
                    try:
                        args = json.loads(func_args_str)
                    except ValueError:
                        args = {}

                    yield f"event: tool_call\ndata: {json.dumps({'name': func_name, 'args': args})}\n\n"

                    # Execute tool in a thread pool (blocking call)
                    observation = await asyncio.to_thread(self._execute_tool, func_name, args)
                    
                    yield f"event: observation\ndata: {json.dumps({'output': observation})}\n\n"

                    # Build tool response message
                    tool_msg_list.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": observation
                    })
                
                # Append tool results to conversation
                messages.extend(tool_msg_list)
                await asyncio.sleep(0.2) # small delay for smooth UI feedback
                
            else:
                # No tool calls: this is the final answer!
                final_answer = content
                
                # Stream the final answer tokens for premium UX
                # (For simplicity here, we split the final answer by characters/words to simulate typing)
                words = final_answer.split(" ")
                for word in words:
                    yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
                    await asyncio.sleep(0.01)
                
                break

        # Save to database if session exists
        if session_id and final_answer:
            await asyncio.to_thread(self.store.add_message, session_id, "user", question)
            await asyncio.to_thread(self.store.add_message, session_id, "assistant", final_answer)

        # Yield done
        yield f"event: done\ndata: {json.dumps({'status': 'completed', 'iterations': iteration})}\n\n"

    def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Executes a financial tool by name and arguments."""
        try:
            if name == "search_sec_filings":
                return self.toolset.search_sec_filings(
                    query=args.get("query", ""),
                    ticker=args.get("ticker"),
                    fiscal_year=args.get("fiscal_year"),
                    doc_type=args.get("doc_type")
                )
            elif name == "get_company_profile":
                return self.toolset.get_company_profile(
                    ticker=args.get("ticker", "")
                )
            elif name == "calculate_financial_ratio":
                return self.toolset.calculate_financial_ratio(
                    ticker=args.get("ticker", ""),
                    metric_a=args.get("metric_a", ""),
                    metric_b=args.get("metric_b", ""),
                    operation=args.get("operation", "/")
                )
            elif name == "get_financial_news_sentiment":
                return self.toolset.get_financial_news_sentiment(
                    ticker=args.get("ticker", ""),
                    query=args.get("query")
                )
            elif name == "generate_investment_memo":
                return self.toolset.generate_investment_memo(
                    ticker=args.get("ticker", ""),
                    findings=args.get("findings", "")
                )
            else:
                return f"Error: Tool '{name}' not found."
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"

    # ------------------ DYNAMIC MOCK SIMULATOR ------------------
    async def _run_mock_agent(self, question: str, session_id: str | None = None) -> AsyncGenerator[str, None]:
        """Simulates agent execution in Mock Mode to show a rich pipeline trace without API key."""
        q_lower = question.lower()
        
        # Determine ticker
        ticker = "AAPL"
        for t in ["AAPL", "MSFT", "NVDA", "TSLA"]:
            if t.lower() in q_lower:
                ticker = t
                break

        # Step 1: Initial Planning
        yield f"event: thought\ndata: {json.dumps({'text': f'User is requesting financial analysis for ticker: {ticker}. I need to retrieve the structured company profile to fetch core metrics (revenue, assets, cash, debt) and perform required calculations.'})}\n\n"
        await asyncio.sleep(1.0)
        
        # Tool Call 1: get_company_profile
        yield f"event: tool_call\ndata: {json.dumps({'name': 'get_company_profile', 'args': {'ticker': ticker}})}\n\n"
        await asyncio.sleep(0.5)
        
        profile_out = self.toolset.get_company_profile(ticker)
        yield f"event: observation\ndata: {json.dumps({'output': profile_out})}\n\n"
        await asyncio.sleep(1.0)

        # Step 2: Next plan step (Calculating margin/ratios)
        yield f"event: thought\ndata: {json.dumps({'text': f'Company profile metrics retrieved. Now I will calculate the Operating Margin (operating_income / revenue) to assess efficiency.'})}\n\n"
        await asyncio.sleep(1.0)
        
        # Tool Call 2: calculate_financial_ratio
        ratio_args = {'ticker': ticker, 'metric_a': 'operating_income', 'metric_b': 'revenue', 'operation': '/'}
        yield f"event: tool_call\ndata: {json.dumps({'name': 'calculate_financial_ratio', 'args': ratio_args})}\n\n"
        await asyncio.sleep(0.5)
        
        ratio_out = self.toolset.calculate_financial_ratio(**ratio_args)
        yield f"event: observation\ndata: {json.dumps({'output': ratio_out})}\n\n"
        await asyncio.sleep(1.0)

        # Step 3: Check Qualitatives (SEC filings search)
        yield f"event: thought\ndata: {json.dumps({'text': f'Operating efficiency calculated. I should search recent SEC filings and annual reports to identify primary risks and competitive strategies.'})}\n\n"
        await asyncio.sleep(1.0)
        
        # Tool Call 3: search_sec_filings
        search_args = {'query': 'supply chain risks capital spending inflation', 'ticker': ticker, 'doc_type': '10-K'}
        yield f"event: tool_call\ndata: {json.dumps({'name': 'search_sec_filings', 'args': search_args})}\n\n"
        await asyncio.sleep(0.5)
        
        search_out = self.toolset.search_sec_filings(**search_args)
        yield f"event: observation\ndata: {json.dumps({'output': search_out})}\n\n"
        await asyncio.sleep(1.0)

        # Step 4: Generate memo
        yield f"event: thought\ndata: {json.dumps({'text': 'Financial parameters, ratios, and risk profiles extracted. I have sufficient data. I will now compile these analytical insights and generate a formal investment research memo.'})}\n\n"
        await asyncio.sleep(1.0)
        
        # Tool Call 4: generate_investment_memo
        memo_findings = (
            f"- Calculated Operating Margin: {ratio_out.split('= ')[-1]}\n"
            f"- Structured registry shows Sector: Technology, Cash position: $40B+.\n"
            f"- SEC filings review highlights component sourcing bottlenecks and competitive hyper-scaler cap-ex pressures."
        )
        memo_args = {'ticker': ticker, 'findings': memo_findings}
        yield f"event: tool_call\ndata: {json.dumps({'name': 'generate_investment_memo', 'args': memo_args})}\n\n"
        await asyncio.sleep(0.5)
        
        memo_out = self.toolset.generate_investment_memo(**memo_args)
        yield f"event: observation\ndata: {json.dumps({'output': 'Investment Memo formatted and returned.'})}\n\n"
        await asyncio.sleep(1.0)

        # Final Answer Streaming
        final_answer = memo_out
        words = final_answer.split(" ")
        for word in words:
            yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
            await asyncio.sleep(0.005)

        # Save to database
        if session_id:
            await asyncio.to_thread(self.store.add_message, session_id, "user", question)
            await asyncio.to_thread(self.store.add_message, session_id, "assistant", final_answer)

        yield f"event: done\ndata: {json.dumps({'status': 'mock_completed', 'iterations': 4})}\n\n"
