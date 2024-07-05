from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from dotenv import load_dotenv


from llama_index.core import PromptTemplate



# Load environment variables from .env file into the script's environment
load_dotenv()
documents = SimpleDirectoryReader("C:/Users/ataglem/Desktop/LLM_analysis/data").load_data()

pipeline = IngestionPipeline(  # default uses open ai embedding (ada)
    transformations=[
        SentenceSplitter(chunk_size=128, chunk_overlap=10),
        OpenAIEmbedding(model = 'text-embedding-ada-002', 
    embed_batch_size = 100),
    ]
)

nodes = pipeline.run(documents=documents)

index = VectorStoreIndex(nodes)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor,KeywordNodePostprocessor

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)

#query_engine = index.as_query_engine()


#response = query_engine.query("rentabilidad bruta ingresos 2021")
#print(response)
node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=0.7),
    ]
#response = query_engine.query("rentabilidad bruta ingresos 2021, percentage with decimals")
#print(response)


from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import ReActAgent

from llama_index.llms.openai import OpenAI

text_qa_template = PromptTemplate("""rentabilidad bruta ingresos 2021, percentage with decimals
  """)


from llama_index.core import get_response_synthesizer

response_synthesizer = get_response_synthesizer(
    response_mode="compact",
    text_qa_template=text_qa_template
)

from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=node_postprocessors,
    response_synthesizer=response_synthesizer
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata

vector_query_engine_tool = QueryEngineTool(
            query_engine = query_engine,
            metadata = ToolMetadata(
            name="besalco financial statement",
            description="You can find financial statement information here",
            )
        )


system_prompt = """You are a expert financial analyst provide an answer to the following"""
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

agent = ReActAgent.from_tools(
                    [vector_query_engine_tool],
                    llm=llm,
                    system_prompt=system_prompt
                )


### interact ###

response = agent.chat('rentabilidad bruta ingresos 2021, percentage with decimals')
print(response.response)