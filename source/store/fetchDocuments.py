import os
import requests
import time
import logging
from io import BytesIO
from PyPDF2 import PdfReader
from google.cloud import bigquery, storage

# Google Cloud Configuration
PROJECT_ID = "corded-forge-417909"
BQ_DATASET_ID = "ProjectRAGMart"
BQ_TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET_ID}.documents"
PROCESSED_TABLE = f"{PROJECT_ID}.{BQ_DATASET_ID}.processed_documents"
GCS_BUCKET = "projectragmart"

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize BigQuery and Storage Clients
bq_client = bigquery.Client()
storage_client = storage.Client()

def fetch_and_process_documents(request):
    """Cloud Function to fetch PDFs, extract text, and upload to BigQuery."""
    logging.info("Starting document processing...")

    while True:  # 🚀 Keep processing until there are no more documents
        # Query BigQuery for unprocessed documents
        query = f"""
        SELECT Id, Titel, Onderwerp, ContentType
        FROM `{BQ_TABLE_ID}`
        WHERE ContentType = 'application/pdf'
        AND Id NOT IN (SELECT document_id FROM `{PROCESSED_TABLE}`)
        LIMIT 100  -- Process in batches of 100
        """
        query_job = bq_client.query(query)
        results = query_job.result()

        # If no more results, stop processing
        if results.total_rows == 0:
            logging.info("✅ All documents have been processed.")
            return "✅ All documents have been processed."

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

        logging.info(f"✅ Processed {len(processed_docs)} documents.")

        # Short delay to prevent API rate limits (optional)
        time.sleep(2)

def download_and_extract_text(document_id):
    """Download a PDF and extract its text."""
    pdf_url = f"https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Document({document_id})/resource"

    try:
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            return extract_text_from_pdf(response.content)

        elif response.status_code == 429:
            logging.warning(f"⚠️ Rate limit reached (429) for {pdf_url}. Retrying after delay...")
            time.sleep(20)
            return None

        else:
            logging.error(f"❌ Failed to download {pdf_url}: {response.status_code}")
            return None

    except Exception as e:
        logging.error(f"❌ Error downloading PDF {document_id}: {e}")
        return None

def extract_text_from_pdf(pdf_content):
    """Extract text from a PDF byte stream."""
    try:
        pdf_file = BytesIO(pdf_content)
        reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() if text.strip() else None
    except Exception as e:
        logging.error(f"❌ Error extracting text: {e}")
        return None

def upload_text_to_bigquery(document_id, title, subject, text):
    """Upload extracted text to BigQuery."""
    table_ref = bq_client.dataset(BQ_DATASET_ID).table("processed_documents")
    rows_to_insert = [{
        "document_id": document_id,
        "title": title if title else "Unknown",
        "subject": subject if subject else "Unknown",
        "text": text
    }]
    errors = bq_client.insert_rows_json(table_ref, rows_to_insert)

    if errors:
        logging.error(f"❌ Failed to insert into BigQuery: {errors}")
    else:
        logging.info(f"✅ Uploaded document {document_id} to BigQuery.")
