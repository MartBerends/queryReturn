import functions_framework
from google.cloud import bigquery, aiplatform, storage
import numpy as np
import pandas as pd
import json
import logging
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

# Google Cloud Config
PROJECT_ID = "corded-forge-417909"
REGION = "europe-west4"
BQ_DATASET_ID = "ProjectRAGMart"
BQ_TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET_ID}.processed_documents"
EMBEDDING_TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET_ID}.document_embeddings"
BATCH_SIZE = 10  # Adjust as needed based on your quota and model performance.

aiplatform.init(project=PROJECT_ID, location=REGION)
model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")
bq_client = bigquery.Client()

def fetch_documents(limit, offset):
    """Retrieve documents in batches from BigQuery."""
    query = f"""
    SELECT document_id, text, subject, title 
    FROM `{BQ_TABLE_ID}`
    WHERE text IS NOT NULL AND document_id NOT IN (SELECT document_id FROM `{EMBEDDING_TABLE_ID}`)
    LIMIT {limit} OFFSET {offset}
    """
    return bq_client.query(query).to_dataframe()

def generate_embeddings_batch(texts):
    """Generate embeddings for a batch of texts."""
    try:
        response = model.get_embeddings(texts)
        return [embedding.values for embedding in response]
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return None

def store_embeddings_batch(df):
    """Store a batch of embeddings in BigQuery."""
    if df.empty:
        return
    print(f"Embeddings data before JSON conversion:\n{df['embedding'].head()}")

    #df["embedding"] = df["embedding"].apply(json.dumps)
    df["embedding"] = df[["embedding"]].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)

    table_ref = bq_client.dataset(BQ_DATASET_ID).table("document_embeddings")
    try:
        bq_client.load_table_from_dataframe(df, table_ref).result()
        logging.info(f"Stored {len(df)} embeddings in BigQuery.")
    except Exception as e:
        logging.error(f"Error storing embeddings: {e}")


def store_embeddings():
    """Main function to fetch, process, and store embeddings in batches."""
    offset = 0
    while True:
        df = fetch_documents(BATCH_SIZE, offset)
        if df.empty:
            break

        texts = df["text"].tolist()
        embeddings = generate_embeddings_batch(texts)
        if embeddings:
            df["embedding"] = embeddings
            store_embeddings_batch(df)
        offset += BATCH_SIZE
    return "Embeddings processed successfully."

@functions_framework.http
def generate_embeddings(request):
    """Cloud Function HTTP Entry Point."""
    return store_embeddings()