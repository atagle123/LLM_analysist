
# load index


def sk_context(industry:str) -> str:
    """Given an industry name, this tool provide information and context about the company"""
     index_load=Index_loading()
    index_load.load_index(data_dir=storage_index,index_id=index_id)
    index=index_load()
    query_engine=Index_query_engine(index)
    query_engine=query_engine.make_query_engine()
    # return the query engine chat when interacting with this tool
    return 