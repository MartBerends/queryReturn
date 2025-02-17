import functions_framework
import logging
import createEmbeddings
# Initialize logging
logging.basicConfig(level=logging.INFO)


@functions_framework.http
def embed_documents(request):
    """
    Step 2: Process the documents (download PDFs, extract text, upload to BigQuery).
    """
    logging.info("Processing and extracting text from PDFs...")
    response = createEmbeddings.generate_embeddings(request)  # Call fetchDocuments function
    return response