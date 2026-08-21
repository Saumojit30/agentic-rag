"""LangSmith-lite local tracing for observability."""

import json
import time
import uuid
from typing import Any
from pathlib import Path
from contextlib import contextmanager

TRACE_FILE = Path(__file__).parent.parent.parent / "data" / "traces.jsonl"

@contextmanager
def trace(span_name: str, inputs: dict[str, Any]):
    TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    trace_data = {
        "id": str(uuid.uuid4()),
        "span": span_name,
        "inputs": inputs,
        "start_time": start_time,
        "outputs": {},
        "tokens": {}
    }
    
    try:
        yield trace_data
    except Exception as e:
        trace_data["error"] = str(e)
        raise
    finally:
        end_time = time.time()
        trace_data["end_time"] = end_time
        trace_data["latency_sec"] = round(end_time - start_time, 4)
        
        with open(TRACE_FILE, "a") as f:
            f.write(json.dumps(trace_data) + "\n")

def get_traces() -> list[dict]:
    """Retrieve all traces from the log file, newest first."""
    if not TRACE_FILE.exists():
        return []
    
    traces = []
    with open(TRACE_FILE, "r") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
                
    return sorted(traces, key=lambda x: x["start_time"], reverse=True)


class PipelineTimer:
    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self._starts: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if name in self._starts:
            duration = time.perf_counter() - self._starts[name]
            self.metrics[name] = round(duration, 4)
