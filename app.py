import os
from flask import Flask, request, jsonify, send_from_directory, Response
from google.cloud import bigquery
from vertexai.preview.language_models import ChatModel  # Correct import
from vertexai.language_models import TextEmbeddingModel
from mistralai_gcp import MistralGoogleCloud # Correct import
from google.cloud import aiplatform
import pandas as pd
app = Flask(__name__)

# Google Cloud Config
PROJECT_ID = "corded-forge-417909"
REGION = "europe-west4"
BQ_DATASET_ID = "ProjectRAGMart"
EMBEDDING_TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET_ID}.document_embeddings"
TOP_N = int(os.environ.get("TOP_N", 3))
MODEL_NAME = "mistral-nemo"
MODEL_VERSION = "2407"

# Initialize Clients and Models (ONCE at app startup)
aiplatform.init(project=PROJECT_ID, location=REGION)
bq_client = bigquery.Client()
embedding_model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")
mistral_client = MistralGoogleCloud(region=REGION, project_id=PROJECT_ID)

def generate_pdf_links(top_matches):
    """Generate download links for the top matches."""
    base_url = "https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Document"
    sources = []
    for _, row in top_matches.iterrows():
        document_id = row["document_id"]
        download_link = f"{base_url}({document_id})/resource"
        sources.append({
            "document_id": document_id,
            "download_link": download_link,
            "distance": row["distance"]  # Include distance for reference
        })
    return sources

def get_query_embedding(query_text):
    """Generate an embedding for the input query."""
    response = embedding_model.get_embeddings([query_text])
    return response[0].values
def get_top_matches(query_embedding, top_n=TOP_N):
    """Retrieve the top N documents that match the query embedding."""
    query_embedding_str = ', '.join(map(str, query_embedding))  # Convert embedding to string for SQL
    query = f"""
    WITH query_embedding AS (
        SELECT ARRAY[{query_embedding_str}] AS query_embedding  -- Rename query embedding
    )
    SELECT 
        d.document_id, 
        d.text, 
        SQRT(SUM(POW(d.embedding[OFFSET(i)] - qe.query_embedding[OFFSET(i)], 2))) AS distance
    FROM `{EMBEDDING_TABLE_ID}` AS d, 
         query_embedding AS qe,
         UNNEST(d.embedding) WITH OFFSET i  -- Explicitly use 'd.embedding'
    GROUP BY d.document_id, d.text
    ORDER BY distance ASC
    LIMIT {top_n}
    """
    try:
        # Execute the query and convert the results to a DataFrame
        results = bq_client.query(query).to_dataframe()
        return results
    except Exception as e:
        print(f"Error querying BigQuery: {e}")
        return pd.DataFrame(columns=["document_id", "text"])



@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query_text = data.get("query", "")
    chat_history = data.get("chat_history", [])
    # Step 1: Generate query embedding
    query_embedding = get_query_embedding(query_text)

    # Step 2: Retrieve top matching documents
    top_matches = get_top_matches(query_embedding)

    if top_matches.empty:
        # No matches found, fallback to generic context
        context = (
            "Jij weet zoveel dingen van de wereld, ook deze vraag kan jij beantwoorden "
            "ondanks dat je er niet helemaal zeker van bent, geef antwoord op deze vraag:"
        )
        full_prompt = f"Context: {context}\n\nUser Question: {query_text}"
        sources = []
    else:
        # Matches found, use documents as context
        context = "\n\n".join(top_matches["text"].tolist())
        sources = generate_pdf_links(top_matches)
        

    # Combine chat history into the prompt
    # Combine chat history with the current prompt
    history_as_prompt = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in chat_history])
    full_prompt = (
        f"Jij bent een behulpzame assistent op het gebied van nederlandse politiek (met een tikje humor en je bent ook een beetje grof) die de volgende informatie tot zijn beschikking heeft:\n\n"
        f"Context:\n{context}\n\n"
        f"geef antwoord op de volgende vraag en gebruik daarbij bovenstaande informatie zoveel als mogelijk:\n"
        f"{query_text}"
        f"Dit is ons gesprek tot zover:{history_as_prompt}\n\n"
    ) 
    print(full_prompt)

    def generate_response():
        try:
            stream = mistral_client.chat.stream(
                model="mistral-nemo-2407",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
            )
            for chunk in stream:
                print("Raw chunk:", chunk)  # Log the raw chunk to inspect the response
                try:
                    yield chunk.data.choices[0].delta.content
                except (IndexError, AttributeError, KeyError) as e:
                    print(f"Error parsing chunk: {e}")
                    yield "Fout in het ontvangen antwoord. Probeer het opnieuw."
        except Exception as e:
            print(f"Error during streaming: {e}")
            yield f"An error occurred: {e}"
    
    # # Stream response from the model
    # def generate_response():
    #     try:
    #         # Send PDF links as the first chunk of the response
    #         if sources:
    #             pdf_links = "\n".join([f"<a href='{source['download_link']}' target='_blank'>{source['download_link']}</a>" for source in sources])
    #             yield f"Bronnen:\n{pdf_links}\n\n"
    
    #         # Stream the assistant's generated response
    #         stream = mistral_client.chat.stream(
    #             model=f"{MODEL_NAME}-{MODEL_VERSION}",
    #             max_tokens=1024,
    #             messages=[
    #                 {"role": "user", "content": full_prompt}
    #             ],
    #         )
    
    #         for chunk in stream:
    #             yield chunk.data.choices[0].delta.content  # Stream the assistant's response
    #     except Exception as e:
    #         yield f"An error occurred: {e}"
    
        # Stream the response to the frontend
    return Response(generate_response(), content_type="text/plain")


@app.route("/")
def index():
    return send_from_directory(".", "index.html")  # Assumes `index.html` is in the same directory as app.py
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)  # Only for local
