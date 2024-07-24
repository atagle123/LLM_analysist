import os
import sys
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor

from dotenv import load_dotenv

# Load environment variables from .env file into the script's environment
load_dotenv()





class PDF_loading:
    def __init__(self,input_dir=None,input_files=None): # loads all the pdfs in the given directory
        if input_dir:
            current_dir=os.getcwd()
            input_dir=os.path.join(current_dir,input_dir)

        self.load_pdf(input_dir,input_files)

    def load_pdf(self,input_dir=None,input_files=None):
        documents = SimpleDirectoryReader(input_dir=input_dir,input_files=input_files).load_data()
        self.documents=documents

    def make_nodes(self):
        pipeline = IngestionPipeline(  # default uses open ai embedding (ada)
        transformations=[
            SentenceSplitter(chunk_size=256, chunk_overlap=20),
            TitleExtractor(),
            OpenAIEmbedding(),
        ])

        nodes = pipeline.run(documents=self.documents)
        return(nodes)


