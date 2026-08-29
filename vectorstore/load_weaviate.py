"""
Embed chunked filing passages (output of ingestion/process_docs.py) and
upsert them into a local Weaviate instance, with hybrid search enabled.

We generate embeddings ourselves with sentence-transformers (free, local,
no API key) rather than using one of Weaviate's built-in vectorizer
modules - this keeps the pipeline provider-agnostic and free to run.

Prereqs:
    docker compose up -d          # starts Weaviate on localhost:8080
    pip install -r requirements.txt

Usage:
    python load_weaviate.py --input ../data/processed
"""

import argparse
from pathlib import Path

import pandas as pd
import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "FilingChunk"
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good enough for a demo project
BATCH_SIZE = 100


def get_or_create_collection(client: weaviate.WeaviateClient):
    if client.collections.exists(COLLECTION_NAME):
        return client.collections.get(COLLECTION_NAME)

    return client.collections.create(
        name=COLLECTION_NAME,
        # vectors are supplied by us (sentence-transformers), not generated
        # by Weaviate, so the vectorizer is "none"
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        # BM25 + vector both need to be present for hybrid search - BM25
        # is on by default per-property, vector comes from our upsert
        properties=[
            wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="doc_id", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="ticker", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.INT),
            wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
        ],
    )


def load_chunks(input_dir: str) -> pd.DataFrame:
    df = pd.read_parquet(input_dir)
    print(f"Loaded {len(df)} chunks from {input_dir}")
    return df


def embed_and_upsert(collection, df: pd.DataFrame, model: SentenceTransformer):
    texts = df["text"].tolist()
    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL}...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    with collection.batch.fixed_size(batch_size=BATCH_SIZE) as batch:
        for row, vector in zip(df.itertuples(index=False), embeddings):
            batch.add_object(
                properties={
                    "chunk_id": row.chunk_id,
                    "doc_id": row.doc_id,
                    "ticker": row.ticker,
                    "chunk_index": int(row.chunk_index),
                    "text": row.text,
                },
                vector=vector.tolist(),
            )

    if collection.batch.failed_objects:
        print(f"WARNING: {len(collection.batch.failed_objects)} objects failed to upsert")
        for fail in collection.batch.failed_objects[:5]:
            print(" ", fail.message)
    else:
        print("All chunks upserted successfully.")


def sanity_check_search(collection, model: SentenceTransformer):
    """Run one vector query and one hybrid query so you can confirm
    retrieval actually works before wiring up the agent."""
    query = "revenue recognition and reportable segments"
    query_vector = model.encode(query).tolist()

    print(f"\n--- Vector search sanity check: '{query}' ---")
    results = collection.query.near_vector(near_vector=query_vector, limit=3)
    for obj in results.objects:
        print(f"  [{obj.properties['ticker']}] {obj.properties['text'][:100]}...")

    print(f"\n--- Hybrid search (vector + BM25) sanity check: '{query}' ---")
    results = collection.query.hybrid(query=query, vector=query_vector, limit=3, alpha=0.5)
    for obj in results.objects:
        print(f"  [{obj.properties['ticker']}] {obj.properties['text'][:100]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/processed")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--grpc-port", type=int, default=50051)
    args = parser.parse_args()

    df = load_chunks(args.input)
    model = SentenceTransformer(EMBED_MODEL)

    client = weaviate.connect_to_local(host=args.host, port=args.port, grpc_port=args.grpc_port)
    try:
        collection = get_or_create_collection(client)
        embed_and_upsert(collection, df, model)
        sanity_check_search(collection, model)
    finally:
        client.close()


if __name__ == "__main__":
    main()
