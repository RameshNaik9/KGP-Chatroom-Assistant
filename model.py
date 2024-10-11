import os
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004",api_key=google_api_key) # set the embedding model, add models/ before the model name
Settings.llm = Gemini(model_name="models/gemini-1.5-flash-002",temperature=1,api_key=google_api_key)

def get_index():
    pinecone_index = pc.Index("kgp-chatroom")
    storage_context = StorageContext.from_defaults(persist_dir="pinecone index", vector_store=PineconeVectorStore(pinecone_index=pinecone_index))
    pc_index = load_index_from_storage(storage_context)
    
    return pc_index

def new_index():
    pc.create_index(name="kgp-chatroom",dimension=768,metric="cosine",spec=ServerlessSpec(cloud="aws",region="us-east-1"))
    pinecone_index = pc.Index("kgp-chatroom")
    
    return pinecone_index