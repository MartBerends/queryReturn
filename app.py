import os
from flask import Flask, request, jsonify
from google.cloud import bigquery
from vertexai.preview.language_models import ChatModel  # Correct import
from vertexai.language_models import TextEmbeddingModel
from mistralai_gcp import MistralGoogleCloud # Correct import
app = Flask(__name__)

# Google Cloud Config
PROJECT_ID = "corded-forge-417909"
REGION = "europe-west4"
BQ_DATASET_ID = "ProjectRAGMart"
EMBEDDING_TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET_ID}.document_embeddings"
TOP_N = int(os.environ.get("TOP_N", 5))
MODEL_NAME = os.environ.get("VERTEX_MODEL_NAME")  # Correct env variable name
MODEL_VERSION = os.environ.get("VERTEX_MODEL_VERSION")  # Correct env variable name

# Initialize Clients and Models (ONCE at app startup)
aiplatform.init(project=PROJECT_ID, location=REGION)
bq_client = bigquery.Client()
embedding_model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")
mistral_client = MistralGoogleCloud(region=REGION, project_id=PROJECT_ID)


def get_query_embedding(query_text):
    """Generate an embedding for the input query."""
    response = embedding_model.get_embeddings([query_text])
    return response[0].values

def get_top_matches(query_embedding, top_n=TOP_N):
    """Retrieve the top N documents that match the query embedding."""
    query_embedding_str = ', '.join(map(str, query_embedding))  # Convert embedding to string for SQL
    query = f"""
    SELECT 
        document_id, text, 
        SQRT(SUM(POW(embedding[i] - ARRAY[{query_embedding_str}][OFFSET(i)], 2))) AS distance
    FROM `{EMBEDDING_TABLE_ID}`, UNNEST(embedding) AS embedding WITH OFFSET i
    GROUP BY document_id, text
    ORDER BY distance ASC
    LIMIT {top_n}
    """
    results = bq_client.query(query).to_dataframe()
    return results




@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query_text = data.get("query", "")

    # ... (get embeddings and top matches - same as before)
    query_embedding = get_query_embedding(query_text)

    # Step 2: Retrieve top matching documents
    top_matches = get_top_matches(query_embedding)
    context = "\n\n".join(top_matches["text"].tolist())
    full_prompt = f"Context: {context}\n\nUser Question: {query_text}"

    try:
        resp = mistral_client.chat.complete(
            model=f"{MODEL_NAME}-{MODEL_VERSION}",
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
        )
        return resp.choices[0].message.content  # Extract content
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error generating response", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)  # Only for local
