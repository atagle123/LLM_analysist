from src.data_loading import PDF_loading
from src.indexing import Nodes_Indexing
from src.retriever import Index_query_engine

def main():
    pdf_load=PDF_loading()
    nodes=pdf_load.make_nodes()

    nodes_index=Nodes_Indexing()
    nodes_index.add_nodes(nodes)
    index=nodes_index.make_index()

    query_engine=Index_query_engine(index)
    query_engine=query_engine.make_query_engine()

    
    pass







if __name__=="__main__":
    pass