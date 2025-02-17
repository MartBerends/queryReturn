import os
import requests
import time
import logging
from PyPDF2 import PdfReader
from google.cloud import bigquery, storage

# Google Cloud Configuration
PROJECT_ID = "corded-forge-417909"  # Your Google Cloud Project ID
BQ_DATASET_ID = "ProjectRAGMart"  # Your BigQuery Dataset
BQ_TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET_ID}.documents"  # BigQuery Table
GCS_BUCKET = "projectragmart"  # Google Cloud Storage Bucket for PDFs

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize BigQuery and Storage Clients
bq_client = bigquery.Client()
storage_client = storage.Client()

# Cloud Function entry point
def fetch_and_process_documents(request):
    """Cloud Function to fetch PDFs, extract text, and upload to BigQuery."""
    logging.info("Starting document processing...")

    # Query BigQuery for unprocessed document metadata
    query = f"""
    SELECT Id, Titel, Onderwerp, ContentType
    FROM `{BQ_TABLE_ID}`
    WHERE ContentType = 'application/pdf' AND Id NOT IN (SELECT document_id FROM `{BQ_TABLE_ID}`)
    LIMIT 10  # Process 10 at a time to avoid timeouts
    """
    query_job = bq_client.query(query)
    results = query_job.result()

    if results.total_rows == 0:
        logging.info("No new documents to process.")
        return "No new documents to process."

    processed_docs = []

    for row in results:
        document_id = row["Id"]
        title = row["Titel"]
        subject = row["Onderwerp"]

        # Download PDF
        pdf_text = download_and_extract_text(document_id)

        if pdf_text:
            # Upload text to BigQuery
            upload_text_to_bigquery(document_id, title, subject, pdf_text)
            processed_docs.append(document_id)

    logging.info(f"Processed {len(processed_docs)} documents.")
    return f"Processed {len(processed_docs)} documents."

def download_and_extract_text(document_id):
    """Download a PDF and extract its text."""
    pdf_url = f"https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Document({document_id})/resource"

    try:
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            pdf_text = extract_text_from_pdf(response.content)
            return pdf_text

        elif response.status_code == 429:
            logging.warning(f"Rate limit reached (429) for {pdf_url}. Retrying after delay...")
            time.sleep(20)  # Wait and retry
            return None

        else:
            logging.error(f"Failed to download {pdf_url}: {response.status_code}")
            return None

    except Exception as e:
        logging.error(f"Error downloading PDF {document_id}: {e}")
        return None

def extract_text_from_pdf(pdf_content):
    """Extract text from a PDF byte stream."""
    try:
        reader = PdfReader(pdf_content)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() if text.strip() else None

    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        return None

def upload_text_to_bigquery(document_id, title, subject, text):
    """Upload extracted text to BigQuery."""
    table_ref = bq_client.dataset(BQ_DATASET_ID).table("processed_documents")
    rows_to_insert = [{
        "document_id": document_id,
        "title": title,
        "subject": subject,
        "text": text
    }]
    errors = bq_client.insert_rows_json(table_ref, rows_to_insert)
    
    if errors:
        logging.error(f"Failed to insert into BigQuery: {errors}")
    else:
        logging.info(f"Uploaded document {document_id} to BigQuery.")

