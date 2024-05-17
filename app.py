
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
# Initialize Chroma DB client
chroma_client = chromadb.PersistentClient(path="vision_ias")

# Initialize Google Generative AI embedding function
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=google_api_key)

# Get or create Chroma DB collection
collection = chroma_client.get_or_create_collection(name="mains365_2024", embedding_function=google_ef)

# print(collection.count())
results = collection.query(
    query_texts=[
        "why social stock exchange is in news?"
    ],
    n_results=5
)

print(results['documents'])