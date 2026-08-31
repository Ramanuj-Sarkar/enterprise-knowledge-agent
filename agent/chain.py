"""
Naive RAG chain: hybrid-retrieve top-k chunks from Weaviate, stuff them
into a prompt, and generate an answer with citations.

This is deliberately the *simplest* version that works end-to-end -
single retrieve, single generate, no reformulation or multi-step
reasoning. Step 5 (agent/graph.py) wraps this in a LangGraph loop that
can decide to reformulate the query and retry when retrieval looks weak.
Get this working and demoable before adding that complexity.

Usage:
    export OPENAI_API_KEY=sk-...
    python chain.py --question "What are Apple's main revenue segments?"
"""

import argparse
import os

import weaviate
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from prompts import SYSTEM_PROMPT, build_user_prompt

COLLECTION_NAME = "FilingChunk"
EMBED_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "gpt-4o-mini"
TOP_K = 5
HYBRID_ALPHA = 0.5  # 0 = pure keyword (BM25), 1 = pure vector


def retrieve_context(collection, embed_model: SentenceTransformer, question: str,
                      top_k: int = TOP_K, alpha: float = HYBRID_ALPHA) -> list[dict]:
    """Hybrid search (vector + BM25) against Weaviate, returned as plain dicts."""
    query_vector = embed_model.encode(question).tolist()

    results = collection.query.hybrid(
        query=question,
        vector=query_vector,
        limit=top_k,
        alpha=alpha,
    )

    contexts = []
    for obj in results.objects:
        contexts.append({
            "ticker": obj.properties["ticker"],
            "doc_id": obj.properties["doc_id"],
            "chunk_id": obj.properties["chunk_id"],
            "text": obj.properties["text"],
            "score": obj.metadata.score if obj.metadata else None,
        })
    return contexts


def generate_answer(client: OpenAI, question: str, contexts: list[dict],
                     model: str = GENERATION_MODEL) -> str:
    if not contexts:
        return "No relevant context was retrieved, so I can't answer this confidently."

    user_prompt = build_user_prompt(question, contexts)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # keep answers grounded, not creative
    )
    return response.choices[0].message.content


def answer_question(question: str, weaviate_host="localhost", weaviate_port=8080,
                     weaviate_grpc_port=50051) -> dict:
    """End-to-end: retrieve -> generate. Returns answer + the sources used
    so the caller can display citations."""
    embed_model = SentenceTransformer(EMBED_MODEL)
    openai_client = OpenAI()  # reads OPENAI_API_KEY from env

    client = weaviate.connect_to_local(
        host=weaviate_host, port=weaviate_port, grpc_port=weaviate_grpc_port
    )
    try:
        collection = client.collections.get(COLLECTION_NAME)
        contexts = retrieve_context(collection, embed_model, question)
        answer = generate_answer(openai_client, question, contexts)
    finally:
        client.close()

    return {"question": question, "answer": answer, "sources": contexts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this.")

    result = answer_question(args.question)

    print(f"\nQuestion: {result['question']}\n")
    print(f"Answer:\n{result['answer']}\n")
    print("Sources used:")
    for i, s in enumerate(result["sources"]):
        score_str = f"{s['score']:.3f}" if s["score"] is not None else "n/a"
        print(f"  [{i+1}] {s['ticker']} / {s['chunk_id']} (score: {score_str})")


if __name__ == "__main__":
    main()
