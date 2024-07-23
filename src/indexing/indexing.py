from llama_index.core import VectorStoreIndex,StorageContext, load_index_from_storage



class Nodes_Indexing:

    def __init__(self):
        self.nodes=[]

    def add_nodes(self,nodes):
        self.nodes+=nodes
        print("Added Nodes")

    def make_index(self,index_id):
        self.index = VectorStoreIndex(self.nodes)
        self.index.set_index_id(index_id)
        return(self.index)

    def __call__(self):
        return(self.index)
    
    def store_index(self,data_dir):
        self.index.storage_context.persist(persist_dir=data_dir) #"./storage


class Index_loading:
    def __init__(self) -> None:
        pass

    def load_index(self,data_dir,index_id):
        storage_context = StorageContext.from_defaults(persist_dir=data_dir)  # storage
        # load index
        index = load_index_from_storage(storage_context, index_id=index_id)
        self.index=index