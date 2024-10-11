import os
from dotenv import load_dotenv
from llama_index.readers.json import JSONReader
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore

from model import new_index

load_dotenv()

intern_input_dir = 'database/internship json'
placement_input_dir = 'database/placement json'
communique_dir = 'database/communique docs'
general_input_dir = 'database/general docs'

internfiles = os.listdir(intern_input_dir)
placementfiles = os.listdir(placement_input_dir)
intern_documents = []
placement_documents = []

json_reader = JSONReader()

communique_reader = SimpleDirectoryReader(input_dir=communique_dir)
document_reader = SimpleDirectoryReader(input_dir=general_input_dir)

for path in internfiles:
  documents = json_reader.load_data(input_file=os.path.join(intern_input_dir, path))
  intern_documents = documents + intern_documents
for path in placementfiles:
  documents = json_reader.load_data(input_file=os.path.join(placement_input_dir, path))
  placement_documents = documents + placement_documents

communique = communique_reader.load_data()
general_documents = document_reader.load_data()

documents = intern_documents + placement_documents + general_documents + communique

text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
Settings.text_splitter = text_splitter

# CREATION OF VECTOR STORE USING PINECONE API
def create_pinecone_index():
    vector_store = PineconeVectorStore(pinecone_index= new_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    pc_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, transformations=[text_splitter])
    pc_index.storage_context.persist(persist_dir="pincone index")
    return pc_index