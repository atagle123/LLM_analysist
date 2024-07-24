import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loading import PDF_loading
from src.indexing import Nodes_Indexing,Index_loading
from src.retriever import Index_query_engine
from src.tools import Tools
from dotenv import load_dotenv

from llama_index.core.tools import QueryEngineTool, ToolMetadata



load_dotenv()



class Tools_from_docs(Tools):
    def __init__(self) -> None:
        super().__init__()

    def add_retriever(self,query_engine,name,description):

        vector_query_engine_tool = QueryEngineTool(
                query_engine = query_engine,
                metadata = ToolMetadata(
                name=name,
                description=description,
                ),
            )
    
        self.add_tools(vector_query_engine_tool)

    def make_retriever(self,input_dir=None,input_files=None,chunk_size=256,chunk_overlap=20,index_id="razonados_besalco",save_index=True,storage_index="./storage"):

        pdf_load=PDF_loading(input_dir=input_dir,input_files=input_files)
        nodes=pdf_load.make_nodes(chunk_size=chunk_size,chunk_overlap=chunk_overlap)

        nodes_index=Nodes_Indexing()
        nodes_index.add_nodes(nodes)
        index=nodes_index.make_index(index_id=index_id)
        if save_index:
            self.save_index(storage_index,nodes_index=nodes_index)
            
        query_engine=Index_query_engine(index)
        query_engine=query_engine.make_query_engine()
        return(query_engine)
    
    def save_index(self,storage_index,nodes_index):
        nodes_index.store_index(data_dir=storage_index)


    def make_retriever_from_index_dir(self,storage_index="./storage",index_id=None):

        index_load=Index_loading()
        index_load.load_index(data_dir=storage_index,index_id=index_id)
        index=index_load()
        query_engine=Index_query_engine(index)
        query_engine=query_engine.make_query_engine()
        return(query_engine)

