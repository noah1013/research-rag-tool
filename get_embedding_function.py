from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
# from langchain_huggingface.HuggingFaceEmbeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_aws import BedrockEmbeddings

# # Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM




def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="mistral")
    # embeddings = OllamaEmbeddings(model="znbang/bge:small-en-v1.5-q8_0")
    # embeddings = OllamaEmbeddings(model="llama2:7b")
    
    
    
    # Load tokenizer and model
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
    
    # # Check if pad_token exists, if not, add it
    # # if tokenizer.pad_token is None:
    #     # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token = tokenizer.eos_token
    
    # Resize model token embeddings
    # model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")
    # model.resize_token_embeddings(len(tokenizer))
    
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="EleutherAI/gpt-j-6b",
    #     model_kwargs={'device': 'cpu'},
    #     encode_kwargs={'normalize_embeddings': False},
    # )

    
    
    return embeddings


get_embedding_function()