import os
import sys
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext



class PDF_loading:
    def __init__(self,data_path="data"):
        current_dir=os.getcwd()
        self.filespath=os.path.join(current_dir,data_path)


    def load_pdf(self):
        documents = SimpleDirectoryReader(self.filespath).load_data()

    def make_nodes(self):
        pipeline = IngestionPipeline(  # default uses open ai embedding (ada)
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
            TitleExtractor(),
            OpenAIEmbedding(),
        ])

        nodes = pipeline.run(documents=documents)




### loading and chunking -- nodes ###


documents = SimpleDirectoryReader("./data").load_data()


pipeline = IngestionPipeline(  # default uses open ai embedding (ada)
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
        OpenAIEmbedding(),
    ]
)

nodes = pipeline.run(documents=documents)

### indexing and embedding ###

from llama_index.core import VectorStoreIndex

vector_index = VectorStoreIndex.from_documents(documents)

index = VectorStoreIndex(nodes)

index.storage_context.persist(persist_dir="<persist_dir>")
### storing ###

db = chromadb.PersistentClient(path="./chroma_db")

# create collection
chroma_collection = db.get_or_create_collection("quickstart")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create your index
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
### querying ###
# https://medium.com/@dylanyap122/unlocking-the-power-of-llamaindex-a-comprehensive-introduction-to-its-functionality-demo-on-news-2246ffc417ee
# 
query_engine = index.as_query_engine()
response = query_engine.query("What is the meaning of life?")
print(response)

vector_index.as_query_engine()

### agent ###

from llama_index.agent import OpenAIAgent
agent = OpenAIAgent.from_tools(
                    [vector_query_engine_tool],
                    llm=OpenAI(model = "gpt-3.5-turbo-0613", temperature = 0),
                    verbose = True,
                    system_prompt=system_prompt
                )


### interact ###

response = agent.chat('Explain me news about American Airlines.')
print(response.response)

