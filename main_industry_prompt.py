import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.prompts.prompt_building import Industry_prompt
from src.prompts.prompt import financial_prompt
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine import SimpleChatEngine
# Load environment variables from .env file into the script's environment
# see https://docs.llamaindex.ai/en/stable/examples/agent/ for agents examples 

def main(industry):

    load_dotenv()

    industry_prompt=Industry_prompt(industry=industry)
    prompt=industry_prompt.main_prompt_builder()
    llm = OpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
    chat_engine = SimpleChatEngine.from_defaults(llm=llm,system_prompt=financial_prompt)

    ### interact ###
    while True:
        user_input = input("Ask something: ")

        if not user_input:
            break
        user_input+=prompt
        response = chat_engine.chat(user_input)
        print(response.response)

    pass

if __name__=="__main__":
    industry="Enaex"
    main(industry)