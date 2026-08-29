# Enterprise Knowledge Agent

An agentic RAG system over SEC 10-K filings built to close specific gaps in my knowledge:
PySpark (scalable data processing), Weaviate (vector store), and
LangSmith/Langfuse (observability + eval).

## Status

- [x] Step 1-2: Download SEC filings, PySpark ingestion (clean/dedup/chunk) **done, tested**
- [x] Step 3: Weaviate vector store + embedding upsert **done, API-validated**
- [x] Step 4: Naive RAG chain **done, logic-tested**
- [x] Step 5: LangGraph agentic loop (retrieve -> evaluate -> reformulate) **done, tested (3 routing scenarios)**
- [x] Step 6: LangSmith instrumentation + eval set **done, evaluator logic tested (7 scenarios)**
- [x] Step 7: FastAPI + Docker **done, endpoint-tested (5 scenarios)**
- [ ] Stretch: hybrid search on Weaviate, Langfuse comparison run

## Repo Structure

```
enterprise-knowledge-agent/
├── agent/
│   ├── graph.py               # LangGraph agent: retrieve -> evaluate -> reformulate -> answer
│   └── prompts.py
├── api/
│   └── main.py                 # FastAPI serving the agent
├── data/
│   ├── processed/            # cleaned/chunked output from Spark/Dask
│   └── raw/                  # downloaded dataset
├── eval/
│   ├── eval_set.jsonl          # 20-30 Q/A pairs for testing
│   └── run_eval.py             # LangSmith eval run
├── ingestion/
│   └── process_docs.py       # PySpark/Dask cleaning + chunking job
├── vectorstore/
│   └── load_weaviate.py      # embed + upsert into Weaviate
├── docker-compose.yml           # Weaviate + FastAPI containers
└── README.md
```

## Quickstart (ingestion stage)

```bash
cd ingestion
pip install -r requirements.txt

# 1. Download filings from SEC EDGAR (edit YOUR_NAME/YOUR_EMAIL first .
#    SEC requires a real contact in the User-Agent header)
python download_filings.py

# 2. Clean, dedupe, and chunk into parquet ready for embedding
python process_docs.py --input ../data/raw --output ../data/processed
```

Output: `data/processed/*.parquet` with columns
`chunk_id, doc_id, ticker, chunk_index, text` one row per ~300-word
chunk, ready to embed and load into Weaviate.

### Why PySpark here

Filings are large (10-Ks routinely run 50-150+ pages of HTML), and
cleaning and chunking is an embarrassingly parallel per-document operation.
`process_docs.py` runs in Spark's local standalone mode. No cluster is
required to develop and run it, but the same job would scale to a real
cluster unchanged if the filing set grew from dozens to thousands of
companies.

## Quickstart (vectorstore stage)

```bash
# start Weaviate locally
docker compose up -d

cd vectorstore
pip install -r requirements.txt

# embeds every chunk with sentence-transformers (all-MiniLM-L6-v2, local,
# free, no API key) and upserts into Weaviate, then runs a sanity-check
# vector search and hybrid (vector + BM25) search
python load_weaviate.py --input ../data/processed
```

The collection (`FilingChunk`) is created with `vectorizer_config=none` .
we generate vectors ourselves rather than relying on a Weaviate module, so
the pipeline stays provider-agnostic. Hybrid search combines our vector
with Weaviate's built-in BM25 keyword scoring (`alpha=0.5` weights the two
evenly, so I should tune this once I have a real eval set in step 6).

**Note:** I validated every API call in `load_weaviate.py` against the
installed `weaviate-client` v4.23.0 (schema/property types, `Configure`,
`connect_to_local`, batch upsert) in a sandboxed environment that couldn't
bind a network port for a live server, so the code is correct against the
real client, but you should still do a first live run yourself via
`docker compose up -d` to confirm end-to-end.

## Quickstart (naive RAG chain)

```bash
cd rag
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python chain.py --question "What are Apple's main revenue segments?"
```

Retrieves top-5 chunks from Weaviate via **hybrid search** (vector +
BM25, `alpha=0.5`), stuffs them into a prompt with numbered source
citations, and generates an answer with `gpt-4o-mini`. If retrieval comes
back empty, it returns a "can't answer confidently" message instead of
letting the model guess. This guard matters once step 6 (eval) is
scoring hallucination rate.

I unit-tested `retrieve_context` and `generate_answer` against mocked
Weaviate/OpenAI responses (retrieval formatting, prompt assembly, message
shape sent to OpenAI, and the empty-context guard all pass). The live
integration still needs a real run against my running Weaviate instance
and an API key.

## Quickstart (agentic loop)

```bash
cd agent
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python graph.py --question "How did services revenue grow?"
```

This wraps the naive chain in a LangGraph loop: **retrieve → judge →
(reformulate → retrieve)\* → generate**. After each retrieval, an LLM call
judges whether the context is actually sufficient to answer the question.
If not, it rewrites the query and retries up to `MAX_RETRIES = 2`, then
falls back to answering with its best-effort context rather than looping
forever. Print the `trace` field to see exactly what the agent decided at
each step, which is useful both for debugging and for demoing the
reasoning in an interview.

I tested the compiled graph against mocked retrieval/LLM calls across the
three routing paths that matter: (1) insufficient-then-sufficient →
reformulates once and succeeds, (2) always-insufficient → stops at
`MAX_RETRIES` instead of looping forever, (3) sufficient-on-first-try →
no wasted reformulation call. All three pass. Live integration still
needs a real Weaviate + OpenAI run.

## Quickstart (LangSmith eval)

```bash
cd eval
pip install -r requirements.txt
export LANGCHAIN_API_KEY=ls-...
export OPENAI_API_KEY=sk-...

python run_eval.py
```

What this does:
1. Uploads `eval_set.jsonl` (21 Q&A pairs across 10 companies, plus one
   deliberately off-topic "trap" question) as a LangSmith dataset.
2. Runs the full agentic pipeline (`agent/graph.py`) against every
   question, with `@traceable` decorators on each node. The LangSmith UI
   shows the full retrieve → judge → reformulate → generate trace per run,
   including how many retries each question needed.
3. Scores every run with two custom evaluators:
   - **correctness** with LLM-as-judge comparing the generated answer to a
     reference answer (CORRECT / PARTIAL / INCORRECT → 1.0/0.5/0.0)
   - **groundedness** with a cheap non-LLM heuristic checking retrieval
     actually returned context, separate from generation quality. For the
     trap question specifically, it checks the agent *refused* to answer
     confidently rather than hallucinating.

I unit-tested both evaluators (all three correctness verdicts, and all
four groundedness cases: normal question with/without retrieved
context, trap question refused-correctly vs. hallucinated) and the
dataset create-vs-reuse logic, all against mocks. Live run still needs a
LangSmith API key and a populated Weaviate index.

**Why this is the artifact worth screen-sharing:** it's the piece that
turns "I built a RAG system" into "I can show you it's 85% correct and
tell you exactly which failure mode is dragging that number down", which
is what the JD's "AI observability & evaluation" line is actually asking
for.

## Quickstart (full stack: API + Weaviate via Docker)

```bash
export OPENAI_API_KEY=sk-...
# optional, for tracing:
export LANGCHAIN_API_KEY=ls-...
export LANGCHAIN_TRACING_V2=true

docker compose up --build
```

This starts two containers: `weaviate` (vector store) and `api` (FastAPI,
waits for Weaviate's healthcheck before starting). Once running:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Apple'\''s main revenue segments?"}'
```

Returns the answer, cited sources (ticker/doc_id/chunk_id/score), how
many reformulation retries the agent needed, and the full step-by-step
`trace`, which is handy for demoing the agent's reasoning without
digging through LangSmith.

`GET /health` is wired into the container's `HEALTHCHECK` for basic
liveness monitoring.

I tested the API with FastAPI's `TestClient` against a mocked agent:
healthy request/response schema, a 502 (not a crash) with no leaked
internal error detail when the agent throws, and 422 validation on empty
or missing `question`. I also unit-tested that `WEAVIATE_HOST` correctly
resolves to `localhost` for local dev and to the `weaviate` container
hostname when set by docker-compose. That's the detail that quietly
breaks a lot of first-time container networking setups, so worth
mentioning if asked about it directly. Live end-to-end run (the full stack
actually talking to real Weaviate + OpenAI) hasn't been done in this
sandboxed environment and is the one thing left to verify on my machine.