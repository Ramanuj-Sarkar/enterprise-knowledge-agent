"""
Agentic RAG loop built on LangGraph: retrieve -> judge -> (reformulate ->
retrieve)* -> generate.

This wraps the naive chain from rag/chain.py with one extra piece of
reasoning: after retrieving, an LLM call judges whether the context is
actually sufficient to answer the question. If not, it rewrites the
search query and retries - up to MAX_RETRIES times - before falling back
to generating with whatever it has.

This is what makes the system "agentic" rather than plain RAG: it makes a
decision about its own retrieval quality instead of blindly trusting the
first search.

Usage:
    export OPENAI_API_KEY=sk-...
    python graph.py --question "What was the discussion of segment revenue?"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langsmith import traceable
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import weaviate

# reuse the naive chain's retrieval + generation building blocks
sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))
from chain import retrieve_context, generate_answer, COLLECTION_NAME, EMBED_MODEL  # noqa: E402
from agent_prompts import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT, build_judge_prompt,
    REFORMULATE_SYSTEM_PROMPT, build_reformulate_prompt,
)

MAX_RETRIES = 2
JUDGE_MODEL = "gpt-4o-mini"


class AgentState(TypedDict):
    original_question: str
    current_query: str
    contexts: list[dict]
    is_sufficient: bool
    retries: int
    answer: str
    trace: list[str]  # human-readable log of what the agent decided, for demoing


def make_graph(weaviate_client, openai_client: OpenAI, embed_model: SentenceTransformer):
    collection = weaviate_client.collections.get(COLLECTION_NAME)

    @traceable(name="retrieve", run_type="retriever")
    def retrieve_node(state: AgentState) -> AgentState:
        contexts = retrieve_context(collection, embed_model, state["current_query"])
        trace = state["trace"] + [f"Retrieved {len(contexts)} chunks for query: \"{state['current_query']}\""]
        return {**state, "contexts": contexts, "trace": trace}

    @traceable(name="judge_sufficiency", run_type="chain")
    def judge_node(state: AgentState) -> AgentState:
        prompt = build_judge_prompt(state["original_question"], state["contexts"])
        response = openai_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        verdict = response.choices[0].message.content.strip().upper()
        is_sufficient = verdict.startswith("SUFFICIENT")
        trace = state["trace"] + [f"Judge verdict: {verdict}"]
        return {**state, "is_sufficient": is_sufficient, "trace": trace}

    @traceable(name="reformulate_query", run_type="chain")
    def reformulate_node(state: AgentState) -> AgentState:
        prompt = build_reformulate_prompt(
            state["original_question"], state["current_query"], state["contexts"]
        )
        response = openai_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": REFORMULATE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        new_query = response.choices[0].message.content.strip()
        trace = state["trace"] + [f"Reformulated query: \"{new_query}\""]
        return {**state, "current_query": new_query, "retries": state["retries"] + 1, "trace": trace}

    @traceable(name="generate_answer", run_type="chain")
    def generate_node(state: AgentState) -> AgentState:
        answer = generate_answer(openai_client, state["original_question"], state["contexts"])
        trace = state["trace"] + ["Generated final answer."]
        return {**state, "answer": answer, "trace": trace}

    def route_after_judge(state: AgentState) -> str:
        if state["is_sufficient"]:
            return "generate"
        if state["retries"] >= MAX_RETRIES:
            return "generate"  # give up and answer with best-effort context
        return "reformulate"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("judge", judge_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "judge")
    graph.add_conditional_edges("judge", route_after_judge, {
        "generate": "generate",
        "reformulate": "reformulate",
    })
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


@traceable(name="agentic_rag_run", run_type="chain")
def run_agent(question: str, weaviate_host="localhost", weaviate_port=8080,
              weaviate_grpc_port=50051) -> dict:
    embed_model = SentenceTransformer(EMBED_MODEL)
    openai_client = OpenAI()
    weaviate_client = weaviate.connect_to_local(
        host=weaviate_host, port=weaviate_port, grpc_port=weaviate_grpc_port
    )
    try:
        app = make_graph(weaviate_client, openai_client, embed_model)
        initial_state: AgentState = {
            "original_question": question,
            "current_query": question,
            "contexts": [],
            "is_sufficient": False,
            "retries": 0,
            "answer": "",
            "trace": [],
        }
        final_state = app.invoke(initial_state)
    finally:
        weaviate_client.close()

    return final_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this.")

    result = run_agent(args.question)

    print(f"\nQuestion: {args.question}\n")
    print("Agent trace:")
    for step in result["trace"]:
        print(f"  - {step}")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
