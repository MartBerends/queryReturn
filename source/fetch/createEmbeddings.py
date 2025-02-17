import functions_framework
from google.cloud import bigquery, aiplatform
import numpy as np
import pandas as pd
import json
import logging

# Google Cloud Config
PROJECT_ID = "corded-forge-417909"
REGION = "europe-west4"
BQ_DATASET_ID = "ProjectRAGMart"
BQ_TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET_ID}.processed_documents"
EMBEDDING_TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET_ID}.document_embeddings"

# Initialize BigQuery client
bq_client = bigquery.Client()

def fetch_documents():
    """Retrieve documents without embeddings from BigQuery."""
    query = f"""
    SELECT Id, text FROM `{BQ_TABLE_ID}`
    WHERE text IS NOT NULL AND Id NOT IN (SELECT Id FROM `{EMBEDDING_TABLE_ID}`)
    """
    return bq_client.query(query).to_dataframe()

def generate_embedding(text):
    """Generate an embedding for a given text using Vertex AI."""
    aiplatform.init(project=PROJECT_ID, location=REGION)

    model = aiplatform.generation.TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    response = model.get_embeddings([text])  # Generate embeddings
    return response[0].values  # Returns a list of floats

def store_embeddings():
    """Fetch documents, generate embeddings, and store in BigQuery."""
    df = fetch_documents()

    if df.empty:
        logging.info("No new documents to process.")
        return "No new documents to process."

    # Generate embeddings
    df["embedding"] = df["text"].apply(generate_embedding)

    # Convert embeddings to JSON format for BigQuery
    df["embedding"] = df["embedding"].apply(json.dumps)

    # Store in BigQuery
    table_ref = bq_client.dataset(BQ_DATASET_ID).table("document_embeddings")
    bq_client.load_table_from_dataframe(df, table_ref).result()
    
    logging.info(f"Stored {len(df)} embeddings in BigQuery.")
    return f"Stored {len(df)} embeddings in BigQuery."

@functions_framework.http
def generate_embeddings(request):
    """Cloud Function HTTP Entry Point."""
    return store_embeddings()
