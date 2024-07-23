from llama_index.core import Prompt
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine








class Index_query_engine:
    def __init__(self,index):
        self.index=index
    
    def make_query_engine(self):

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=3,
        )
        template = (
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Given this information, please answer the question and each answer should start with code word AI Demos: {query_str}\n"
        )

        qa_template = Prompt(template)

        # configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            text_qa_template=qa_template
        )

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        return(query_engine)