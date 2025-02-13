import os
from flask import Flask, request, jsonify
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
TOP_N = int(os.environ.get("TOP_N", 5))
MODEL_NAME = "mistral-nemo"
MODEL_VERSION = "2407"

# Initialize Clients and Models (ONCE at app startup)
aiplatform.init(project=PROJECT_ID, location=REGION)
bq_client = bigquery.Client()
embedding_model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")
mistral_client = MistralGoogleCloud(region=REGION, project_id=PROJECT_ID)


def get_query_embedding(query_text):
    """Generate an embedding for the input query."""
    response = embedding_model.get_embeddings([query_text])
    print(response)
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

    # Step 1: Generate query embedding
    query_embedding = get_query_embedding(query_text)

    # Step 2: Retrieve top matching documents
    top_matches = get_top_matches(query_embedding)

    if  top_matches.empty:
        # No matches found, use the query itself as the context
        context = "Jij weet zoveel dingen van de wereld, ook deze vraag kan jij beantwoorden ondanks dat je er niet helemaal zeker van bent, geef antwoord op deze vraag:"
        full_prompt = f"Context: {context}\n\nUser Question: {query_text}"
    else:
        # Matches found, use them as the context
        context = "\n\n".join(top_matches["text"].tolist())
        full_prompt = (
            f"Jij bent een behulpzame assistent die de volgende informatie tot zijn beschikking heeft:\n\n"
            f"Context:\n{context}\n\n"
            f"geef antwoord op de volgende vraag en de bovenstaande informatie:\n"
            f"{query_text}"
)

    # Step 3: Call the model for response generation
    try:
        resp = mistral_client.chat.complete(
            model=f"{MODEL_NAME}-{MODEL_VERSION}",
            messages=[
                {"role": "user", "content": full_prompt},
            ],
        )
        return jsonify({
            "response": resp.choices[0].message.content,
            "sources": top_matches.to_dict(orient="records") if not top_matches.empty else [],
        })
    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": "Error generating response"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)  # Only for local
