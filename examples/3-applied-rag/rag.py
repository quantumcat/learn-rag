import pandas as pd

df = pd.read_csv('F:\\aiprojects\\genai\\evpopdata.csv')

# Create synthetic text fields IN THE DATAFRAME (before converting to dict)
df["vehicle_description"] = df.apply(
    lambda row: f"{row['VIN']} {row['Make']} {row['Model']} Range: {row['Electric Range']} miles | Location: {row['City']}, {row['State']}",
    axis=1
)

data = df.head(50).to_dict('records')

from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer('all-MiniLM-L6-v2') # Output dim = 384

# create the vector database client
qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance

qdrant.recreate_collection(
    collection_name="ev_sales",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)

# vectorize!
qdrant.upload_points(
    collection_name="ev_sales",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc["vehicle_description"]).tolist(),
            payload={
                  "make": doc["Make"],
                  "model": doc["Model"],
                  "range": doc["Electric Range"],
                  "city": doc["City"],
                  "state": doc["State"],
                  "vin": doc["VIN"],
                  # Add synthetic description for debugging
                  "description": doc["vehicle_description"]  
            }
         ) for idx, doc in enumerate(data)
    ]
)

user_prompt = "Tesla with over 300 mile range. Is it available in Lynnwood, WA?"

hits = qdrant.search(
    collection_name="ev_sales",
    query_vector=encoder.encode(user_prompt).tolist(),
    limit=50
)

for hit in hits:
  print(hit.payload, "score:", hit.score)

# Format results for LLM
formatted_results = [
    f"{hit.payload['make']} {hit.payload['model']} (VIN: {hit.payload['vin']}): "
    f"{hit.payload['range']} miles, located in {hit.payload['city']}, {hit.payload['state']}"
    for hit in hits
]

# define a variable to hold the search results
search_results = [hit.payload for hit in hits]

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",  # Llamafile's default endpoint
    api_key="sk-no-key-required"          # API key is ignored by llamafile
)

completion = client.chat.completions.create(
    model="Qwen_Qwen3-4B-Q4_K_M",
    messages=[
        {"role": "system", "content": "You are an EV sales assistant. Use ONLY the provided data."},
        {"role": "user", "content": f"Based on this data: {formatted_results}\n\nAnswer: {user_prompt}"}
    ]
)

print(completion.choices[0].message.content)