import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loading import PDF_loading
from src.indexing import Nodes_Indexing,Index_loading
from src.retriever import Index_query_engine
from src.tools import Tools

from src.prompts import financial_prompt

from dotenv import load_dotenv
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import ReActAgent

from llama_index.llms.openai import OpenAI


# Load environment variables from .env file into the script's environment


def main():

    load_dotenv()
   
    """pdf_load=PDF_loading(input_files=["./data/razonados/Analisis_Razonado_1Q23.pdf"])
    nodes=pdf_load.make_nodes()

    nodes_index=Nodes_Indexing()
    nodes_index.add_nodes(nodes)
    index=nodes_index.make_index(index_id="razonados_besalco")
    nodes_index.store_index(data_dir="./storage")"""
    index_load=Index_loading()
    index_load.load_index(data_dir="./storage",index_id="razonados_besalco")
    index=index_load()
    query_engine=Index_query_engine(index)
    query_engine=query_engine.make_query_engine()
    
    tools=Tools()

    vector_query_engine_tool = QueryEngineTool(
            query_engine = query_engine,
            metadata = ToolMetadata(
            name='besalco_financial_statement',
            description='You can find financial statement information here',
            ),
        )
    
    tools.add_tools(vector_query_engine_tool)

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

    agent = OpenAIAgent.from_tools(
                        tools(),
                        llm=llm,
                        system_prompt=financial_prompt,
                        verbose=True
                    )


    ### interact ###
    while True:
        user_input = input("Ask something: ")

        if not user_input:
            break
        
        response = agent.chat(user_input)#('besalco rentabilidad bruta ingresos 2021, percentage with decimals')
        print(response.response)


    pass







if __name__=="__main__":
    main()