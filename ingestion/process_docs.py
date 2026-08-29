"""
PySpark ingestion job: clean, chunk, and deduplicate SEC filing text
before it goes to embedding + Weaviate.

Why Spark here (and not pandas): filings are large (10-Ks routinely run
50-150+ pages of raw HTML/text), and chunking + dedup is an
embarrassingly parallel per-document operation. Once you're past a few
hundred filings, a single-process pandas loop becomes the bottleneck -
this job distributes that work across partitions instead.

Usage:
    pip install pyspark beautifulsoup4
    python process_docs.py --input ../data/raw --output ../data/processed
"""

import argparse
import re
import uuid
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, StructType, StructField

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


CHUNK_SIZE_WORDS = 300      # target words per chunk
CHUNK_OVERLAP_WORDS = 50    # overlap between consecutive chunks


def strip_html(raw_text: str) -> str:
    """Strip HTML tags/boilerplate from a raw filing document."""
    if not raw_text:
        return ""
    if HAS_BS4 and "<html" in raw_text.lower():
        soup = BeautifulSoup(raw_text, "html.parser")
        text = soup.get_text(separator=" ")
    else:
        text = raw_text
    # collapse whitespace, drop non-printable junk common in EDGAR dumps
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E]", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Split cleaned text into overlapping word-count chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 20:  # skip near-empty trailing scraps
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_spark_session(app_name="filing-ingestion"):
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")  # standalone mode - no cluster needed to run this
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/raw", help="Raw filings directory")
    parser.add_argument("--output", default="../data/processed", help="Output parquet directory")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    spark = build_spark_session()

    # Read every filing under input_dir as (path, content) rows, recursing
    # through sec-edgar-downloader's nested ticker/form/accession folders.
    # wholeTextFiles keeps each document intact so cleaning/chunking stays
    # per-document rather than per-line.
    spark.sparkContext._jsc.hadoopConfiguration().set(
        "mapreduce.input.fileinputformat.input.dir.recursive", "true"
    )
    raw_rdd = spark.sparkContext.wholeTextFiles(str(input_dir))
    raw_df = raw_rdd.toDF(["file_path", "raw_text"])

    print(f"Loaded {raw_df.count()} raw files from {input_dir}")

    # Derive ticker/doc id from the folder structure sec-edgar-downloader
    # produces: .../sec-edgar-filings/<TICKER>/10-K/<accession>/...
    extract_ticker = F.udf(lambda p: p.split("/sec-edgar-filings/")[-1].split("/")[0]
                            if "/sec-edgar-filings/" in p else "unknown", StringType())
    extract_doc_id = F.udf(lambda p: p.split("/")[-2] if "/" in p else p, StringType())

    with_meta_df = (
        raw_df
        .withColumn("ticker", extract_ticker(F.col("file_path")))
        .withColumn("doc_id", extract_doc_id(F.col("file_path")))
    )

    # Clean HTML -> plain text (UDF wraps the non-Spark cleaning logic)
    clean_udf = F.udf(strip_html, StringType())
    cleaned_df = with_meta_df.withColumn("clean_text", clean_udf(F.col("raw_text")))

    # Drop empty/near-empty documents before the expensive chunking step
    cleaned_df = cleaned_df.filter(F.length("clean_text") > 200)

    # Deduplicate near-identical filings (EDGAR sometimes stores the same
    # filing under multiple file variants - exhibit copies, etc.)
    deduped_df = cleaned_df.dropDuplicates(["ticker", "clean_text"])

    print(f"After cleaning + dedup: {deduped_df.count()} documents")

    # Chunk each document's text into overlapping passages
    chunk_udf = F.udf(chunk_text, ArrayType(StringType()))
    chunked_df = deduped_df.withColumn("chunks", chunk_udf(F.col("clean_text")))

    # Explode into one row per chunk, with a stable chunk id for Weaviate upsert
    exploded_df = (
        chunked_df
        .select("ticker", "doc_id", F.posexplode("chunks").alias("chunk_index", "text"))
        .withColumn("chunk_id", F.concat_ws("-", F.col("doc_id"), F.col("chunk_index")))
        .select("chunk_id", "doc_id", "ticker", "chunk_index", "text")
    )

    total_chunks = exploded_df.count()
    print(f"Produced {total_chunks} chunks ready for embedding")

    exploded_df.write.mode("overwrite").parquet(str(output_dir))
    print(f"Wrote chunked output to {output_dir}")

    spark.stop()


if __name__ == "__main__":
    main()
