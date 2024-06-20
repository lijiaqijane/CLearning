from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
import chromadb
from torch import bfloat16
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from openai import AzureOpenAI
class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        # Determine model type from the kwargs
        model_name = kwargs.get('model_name', 'gpt-3.5-turbo')

    def __call__(self, prompt: str):
        prompt = str(prompt)
        prompt = prompt.replace('"', '^')
        while True:
            if '\n' in prompt:
                prompt = prompt.strip().replace('\n', ' ')
            else:
                break
        args = [{"role": "user", "content": prompt}]
        ENDPOINT = f"https://api.tonggpt.mybigai.ac.cn/proxy/canadaeast"
        client = AzureOpenAI(
                    api_key="bf34690d256c8856366f78eb83e9c771",   
                    api_version="2024-02-01",
                    azure_endpoint=ENDPOINT,
                    )

        response = client.chat.completions.create(
            model="gpt-35-turbo-0125",
            messages=args,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
        )

        return response.choices[0].message.content

def get_model(df_new_token, model="/scratch2/nlp/plm/Meta-Llama-3-8B-Instruct"):  #Llama-2-7b-chat-hf, LLaMA-2-7B-32K, Meta-Llama-3-8B-Instruct,Llama-2-13b-chat-hf
    bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

    model_config = AutoConfig.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True,
        config=model_config,
        #load_in_8bit=True
        quantization_config=bnb_config,
    )#.half().to(device)
    query_pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            repetition_penalty=1.1,
            max_new_tokens = df_new_token,
            temperature=0,
            # torch_dtype=torch.float16,
            device_map="auto")
    llm = HuggingFacePipeline(pipeline=query_pipeline)
    return llm


def get_similarity_encoder(encode_model='/scratch/nlp/lijiaqi/models/bert-base-nli-mean-tokens'):
    encoder = SentenceTransformer(encode_model)
    return encoder


def get_vectordb(re_no=1, df_collection_name="RAMC", df_path="/scratch/nlp/lijiaqi/models/chroma_RC",df_model_name = "/scratch/nlp/lijiaqi/models/all-MiniLM-L6-v2"):
    embeddings = SentenceTransformerEmbeddings(model_name = df_model_name)
    chroma = chromadb.PersistentClient(path = df_path)
    collection = chroma.get_collection(df_collection_name)
    vectordb = Chroma(
        client=chroma,
        collection_name= df_collection_name,
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 1, "filter":{'source': {'$nin': ['shortmem']}}})  #search_type="similarity_score_threshold", "score_threshold": 1.3, 
    return retriever, collection, vectordb
