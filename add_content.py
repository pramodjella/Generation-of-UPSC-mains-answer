from langchain_community.document_loaders import PyPDFLoader
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Load PDF and split into pages
loader = PyPDFLoader(r"C:\Users\DELL\Downloads\VisionIAS Mains 365 July 2023 Polity and Governance Aug22-May23.pdf")
pages = loader.load_and_split()

# Split text into chunks
# text_splitter = CharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )
# text_chunks = text_splitter.split_documents(pages)

# Initialize Chroma DB client
chroma_client = chromadb.PersistentClient(path="vision_ias")

# Initialize Google Generative AI embedding function
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=google_api_key)

# Get or create Chroma DB collection
collection = chroma_client.get_or_create_collection(name="mains365_2024", embedding_function=google_ef)

# Check the number of existing chunks in the collection
chunk_id = collection.count()
print(f"Collection already contains {chunk_id} chunks")

# Prepare documents and IDs for insertion
documents = [chunk.page_content for chunk in pages]
ids = [str(i) for i in range(chunk_id+1,chunk_id+len(pages)+1)]


# Add documents to the collection
collection.add(documents=documents, ids=ids)

print(f"Inserted {len(documents)} chunks into the collection.")
