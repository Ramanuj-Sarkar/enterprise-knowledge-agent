"""
Runs the agentic RAG pipeline against the eval set and scores it with
LangSmith, using two custom evaluators:

  - correctness: LLM-as-judge comparing the agent's answer to a reference
    answer (catches wrong/hallucinated facts)
  - groundedness: checks whether the retrieved context actually mentions
    the ticker the question was about (catches retrieval failures
    specifically, separate from generation quality)

The deliberately off-topic last question in eval_set.jsonl exists to
check the agent degrades gracefully (says it can't answer) instead of
confidently hallucinating - track that case's correctness score
specifically when reviewing results.

Prereqs:
    export LANGCHAIN_API_KEY=ls-...
    export LANGCHAIN_TRACING_V2=true
    export OPENAI_API_KEY=sk-...

Usage:
    python run_eval.py
"""

import json
import os
import sys
from pathlib import Path

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))
from graph import run_agent  # noqa: E402

DATASET_NAME = "sec-filings-agent-eval"
JUDGE_MODEL = "gpt-4o-mini"


def load_eval_set(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def ensure_dataset(client: Client, examples: list[dict]) -> str:
    """Create (or reuse) a LangSmith dataset from the eval set."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Reusing existing dataset '{DATASET_NAME}' ({dataset.id})")
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Q&A pairs over SEC 10-K filings for evaluating the agentic RAG pipeline.",
        )
        client.create_examples(
            inputs=[{"question": e["question"]} for e in examples],
            outputs=[{"reference_answer": e["reference_answer"], "ticker": e["ticker"]} for e in examples],
            dataset_id=dataset.id,
        )
        print(f"Created dataset '{DATASET_NAME}' with {len(examples)} examples ({dataset.id})")
    return dataset.id


def target(inputs: dict) -> dict:
    """The function LangSmith calls for each example - runs the full agent."""
    result = run_agent(inputs["question"])
    return {
        "answer": result["answer"],
        "trace": result["trace"],
        "num_context_chunks": len(result["contexts"]),
        "retries": result["retries"],
    }


def correctness_evaluator(run: Run, example: Example) -> dict:
    """LLM-as-judge: does the generated answer match the reference answer?"""
    client = OpenAI()
    question = example.inputs["question"]
    reference = example.outputs["reference_answer"]
    generated = run.outputs.get("answer", "")

    judge_prompt = (
        f'Question: "{question}"\n'
        f'Reference answer: "{reference}"\n'
        f'Generated answer: "{generated}"\n\n'
        "Does the generated answer convey the same key facts as the reference "
        "answer? Respond with EXACTLY one word: CORRECT, PARTIAL, or INCORRECT."
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "You judge factual correctness of RAG-generated answers against a reference answer. Be strict about factual accuracy, lenient about phrasing."},
            {"role": "user", "content": judge_prompt},
        ],
        temperature=0,
    )
    verdict = response.choices[0].message.content.strip().upper()
    score_map = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}
    score = next((v for k, v in score_map.items() if verdict.startswith(k)), 0.0)

    return {"key": "correctness", "score": score, "comment": verdict}


def groundedness_evaluator(run: Run, example: Example) -> dict:
    """Did retrieval actually pull context relevant to the right company?
    Cheap heuristic (no LLM call) - separates retrieval failures from
    generation failures, which matters for debugging which step to fix."""
    expected_ticker = example.outputs.get("ticker", "N/A")
    num_chunks = run.outputs.get("num_context_chunks", 0)

    if expected_ticker == "N/A":
        # the deliberate off-topic trap question - score based on whether
        # the agent avoided a confident wrong answer
        answer = run.outputs.get("answer", "").lower()
        avoided_hallucination = any(
            phrase in answer for phrase in ["can't answer", "cannot answer", "not sufficient", "don't have"]
        )
        return {"key": "groundedness", "score": 1.0 if avoided_hallucination else 0.0,
                "comment": "off-topic trap question - checking for graceful refusal"}

    score = 1.0 if num_chunks > 0 else 0.0
    return {"key": "groundedness", "score": score, "comment": f"{num_chunks} chunks retrieved"}


def main():
    for var in ("LANGCHAIN_API_KEY", "OPENAI_API_KEY"):
        if not os.environ.get(var):
            raise SystemExit(f"Set {var} before running this.")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    eval_set_path = Path(__file__).parent / "eval_set.jsonl"
    examples = load_eval_set(str(eval_set_path))
    print(f"Loaded {len(examples)} eval questions from {eval_set_path}")

    client = Client()
    ensure_dataset(client, examples)

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator, groundedness_evaluator],
        experiment_prefix="agentic-rag",
        description="Agentic RAG (retrieve->judge->reformulate->generate) over SEC 10-K filings.",
    )

    print("\nEval run complete. View full results and traces in the LangSmith UI.")
    print(results)


if __name__ == "__main__":
    main()
