import os
import pandas as pd
from google.cloud import bigquery

# Configuration
TEXT_FOLDER = "/home/user/texts"
PROJECT_ID = "corded-forge-417909"  # Replace with your actual Google Cloud Project ID
DATASET_ID = "ProjectRAGMart"  # Replace with your BigQuery dataset name
TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.documents"

# Initialize BigQuery client
bq_client = bigquery.Client()

def load_texts_to_dataframe():
    """Load only new extracted text files into a DataFrame."""
    existing_docs = get_existing_document_ids_from_bigquery()
    data = []
    
    for filename in os.listdir(TEXT_FOLDER):
        doc_id = filename.replace(".txt", "")
        if doc_id in existing_docs:  
            continue  # Skip already uploaded documents

        with open(os.path.join(TEXT_FOLDER, filename), "r", encoding="utf-8") as f:
            text = f.read()
        data.append({"document_id": doc_id, "text": text})

    return pd.DataFrame(data)

def upload_to_bigquery(df):
    """Upload DataFrame to BigQuery."""
    job = bq_client.load_table_from_dataframe(df, TABLE_ID)
    job.result()  # Wait for job completion
    print(f"Uploaded {len(df)} documents to BigQuery.")

# Run the pipeline
df_texts = load_texts_to_dataframe()
upload_to_bigquery(df_texts)
