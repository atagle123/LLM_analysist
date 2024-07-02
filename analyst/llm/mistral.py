import torch
import transformers

from torch import cuda
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

DEVICE = f'cuda:{cuda.current_device()}' if cuda.is_available() else None
MODEL_NAME = "mistralai/Mixtral-8x22B-Instruct-v0.1"

def get_llm():
    """
    Returns a HuggingFacePipeline object for the Mistral-22B-Instruct-v0.1 model.

    :return: HuggingFacePipeline object
    """
    bnb_config=None
    if DEVICE:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    tokenizer.pad_token = tokenizer.eos_token

    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        max_new_tokens=512,
        repetition_penalty=1.1,
        pad_token_id=2
    )

    llm = HuggingFacePipeline(pipeline=pipeline)
    return llm

llm = get_llm()