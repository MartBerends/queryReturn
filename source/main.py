import functions_framework
import fetchDocuments
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

@functions_framework.http
def process_documents(request):
    logging.info("Processing and extracting text from PDFs...")
    response = fetchDocuments.fetch_and_process_documents(request)  # Call fetchDocuments function
    return response