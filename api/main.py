"""
FastAPI service wrapping the agentic RAG pipeline (agent/graph.py).

Endpoints:
    GET  /health          - liveness check
    POST /ask              - ask a question, get an answer + sources + agent trace

Usage:
    export OPENAI_API_KEY=sk-...
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))
from graph import run_agent  # noqa: E402

app = FastAPI(
    title="Enterprise Knowledge Agent",
    description="Agentic RAG over SEC 10-K filings, backed by Weaviate + LangGraph.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    ticker: str
    doc_id: str
    chunk_id: str
    score: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    retries: int
    trace: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = run_agent(request.question)
    except Exception as e:
        # Don't leak internal stack traces to callers - log server-side
        # instead in a real deployment (this is where LangSmith tracing
        # + structured logging would hook in).
        raise HTTPException(status_code=502, detail=f"Agent execution failed: {type(e).__name__}") from e

    return AskResponse(
        question=request.question,
        answer=result["answer"],
        sources=[
            Source(ticker=c["ticker"], doc_id=c["doc_id"], chunk_id=c["chunk_id"], score=c.get("score"))
            for c in result["contexts"]
        ],
        retries=result["retries"],
        trace=result["trace"],
    )
