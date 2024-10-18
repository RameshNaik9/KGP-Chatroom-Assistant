import os
import docx2txt
# from typing import List, Tuple
from llama_index.core import SimpleDirectoryReader #Document in case someone wanted to define what would be the ooutput dataype in terms of Llama index
from llama_index.readers.json import JSONReader
from llama_index.core.node_parser import SentenceSplitter, JSONNodeParser, SemanticSplitterNodeParser
from kgpchatroom import KGPChatroomModel

embedding_model = KGPChatroomModel().get_embedding_model()

class DocumentLoader:
    """Document loader to load files from a directory into documents."""

    def __init__(self, input_dir):
        self.input_dir = input_dir

    def load_documents_and_nodes(self): #-> Tuple[List[Document],List[node_parser.Node]] .... Since there is no pre-defined datatype for Node, we can't use it in the return type
        """Loads documents from the directory, handling .json files separately from other formats."""
        documents = []
        nodes = []

        # Iterate through all files in the directory
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                file_path = os.path.join(root, file)

                # If the file is a .json, use JSONReader
                if file.lower().endswith('.json'):
                    json_reader = JSONReader()
                    try:
                        # Read the JSON file and load it as a Document
                        json_documents = json_reader.load_data(file_path)
                        documents.extend(json_documents)
                    except Exception as e:
                        print(f"Error loading JSON file {file}: {e}")
                    json_parser = JSONNodeParser()
                    json_nodes = json_parser.get_nodes_from_documents(documents)
                    nodes.extend(json_nodes)

                # For all other formats, use SimpleDirectoryReader
                else:
                    try:
                        simple_reader = SimpleDirectoryReader(input_dir=self.input_dir)
                        other_documents = simple_reader.load_data()
                        documents.extend(other_documents)
                    except Exception as e:
                        print(f"Error loading file {file}: {e}")
                    semantic_splitter = SemanticSplitterNodeParser(buffer_size=1,breakpoint_percentile_threshold=90,include_metadata=True,embed_model=embedding_model)
                    semantic_nodes = semantic_splitter.get_nodes_from_documents(documents)
                    nodes.extend(semantic_nodes)
                    
        return documents, nodes

    def text_splitter(self):
        """Splits the text content of a document into sentences."""
        text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
        return text_splitter