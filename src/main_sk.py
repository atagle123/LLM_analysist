import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools import Tools_from_docs

from src.prompts import financial_prompt

from dotenv import load_dotenv
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import ReActAgent

from llama_index.llms.openai import OpenAI


# Load environment variables from .env file into the script's environment
# see https://docs.llamaindex.ai/en/stable/examples/agent/ for agents examples 

def main():

    load_dotenv()
   
    tools=Tools_from_docs()

    ### razonados ### 
    for quarter in ["1Q23","1Q24","2Q23","3Q23","4Q23"]:
        
        name=f"Analisis_Razonado_{quarter}"

        ### make indexes ###
        
       # path=f"C:/Users/ataglem/Desktop/LLM_analysis/data/razonados/{name}.pdf"

       # qe=tools.make_retriever(input_files=[path],index_id=name,save_index=True,storage_index=f"./storage/razonados/{name}",chunk_size=256,chunk_overlap=20)

        ### load indexes ###

        qe=tools.make_retriever_from_index_dir(storage_index=f"./storage/razonados/{name}",index_id=name)

        ### add retriever ###
        description=f"analisis razonado con informacion financiera y detalle de las compañias que forman parte de sigdo koppers para {quarter}""Use a detailed plain text question as input to the tool."
        tools.add_retriever(query_engine=qe,name=name,description=description)


    

    ### financial statements ### 
    for quarter in ["1Q23","2Q23","3Q23","4Q23"]:

        name=f"Estados_financieros_{quarter}"
        ### make indexes ###
        #path=f"C:/Users/ataglem/Desktop/LLM_analysis/data/financials/{name}.pdf"

        #qe=tools.make_retriever(input_files=[path],index_id=name,save_index=True,storage_index=f"./storage/financials/{name}",chunk_size=2048,chunk_overlap=60)

        ### load indexes ###

        qe=tools.make_retriever_from_index_dir(storage_index=f"./storage/financials/{name}",index_id=name)

        ### add retriever ###

        description=f"estados financieras con informacion financiera y detalle de las compañias que forman parte de sigdo koppers para {quarter}""Use a detailed plain text question as input to the tool."
        tools.add_retriever(query_engine=qe,name=name,description=description)

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