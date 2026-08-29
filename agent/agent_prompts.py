JUDGE_SYSTEM_PROMPT = """You judge whether retrieved context passages are
sufficient to answer a question about SEC 10-K filings. Respond with
EXACTLY one word: "SUFFICIENT" or "INSUFFICIENT". Context is insufficient
if it's off-topic, too vague, or missing the specific fact the question
asks for."""


def build_judge_prompt(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return f'Question: "{question}"\n\nContext: (nothing retrieved)\n\nIs this sufficient?'

    sources_block = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(contexts))
    return f'Question: "{question}"\n\nRetrieved context:\n{sources_block}\n\nIs this sufficient to answer the question?'


REFORMULATE_SYSTEM_PROMPT = """You rewrite search queries to improve
retrieval against a vector+keyword index of SEC 10-K filing text. The
previous query didn't retrieve sufficient context. Produce ONE improved
search query - more specific, using financial-filing terminology, or
approaching the topic from a different angle. Respond with ONLY the new
query text, nothing else."""


def build_reformulate_prompt(original_question: str, previous_query: str,
                              contexts: list[dict]) -> str:
    context_note = (
        "No relevant results were retrieved." if not contexts
        else f"The retrieved results were off-topic or too vague, e.g.: \"{contexts[0]['text'][:150]}...\""
    )
    return (
        f'Original question: "{original_question}"\n'
        f'Previous search query: "{previous_query}"\n'
        f"{context_note}\n\n"
        "Write one improved search query."
    )
