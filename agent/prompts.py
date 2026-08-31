SYSTEM_PROMPT = """You are a financial research assistant answering questions
about SEC 10-K filings. Answer ONLY using the provided context passages.
If the context doesn't contain enough information to answer confidently,
say so explicitly rather than guessing. Always cite which source(s) you
used by their [n] marker."""


def build_user_prompt(question: str, contexts: list[dict]) -> str:
    """Assemble the retrieved chunks + question into a single prompt.

    contexts: list of dicts with keys ticker, doc_id, chunk_id, text, score
    """
    sources_block = "\n\n".join(
        f"[{i+1}] (Ticker: {c['ticker']}, Chunk: {c['chunk_id']})\n{c['text']}"
        for i, c in enumerate(contexts)
    )

    return f"""Context passages:

{sources_block}

Question: {question}

Answer the question using only the passages above, and cite sources like [1], [2]."""
