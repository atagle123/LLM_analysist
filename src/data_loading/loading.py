import os
import sys
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor





class PDF_loading:
    def __init__(self,data_path="data"): # loads all the pdfs in the given directory
        current_dir=os.getcwd()
        self.filespath=os.path.join(current_dir,data_path)
        self.load_pdf()

    def load_pdf(self):
        documents = SimpleDirectoryReader(self.filespath).load_data()
        self.documents=documents

    def make_nodes(self):
        pipeline = IngestionPipeline(  # default uses open ai embedding (ada)
        transformations=[
            SentenceSplitter(chunk_size=128, chunk_overlap=10),
            TitleExtractor(),
            OpenAIEmbedding(),
        ])

        nodes = pipeline.run(documents=self.documents)
        return(nodes)


