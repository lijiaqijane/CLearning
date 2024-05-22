import os           
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma

embeddings = SentenceTransformerEmbeddings(model_name="/scratch/nlp/lijiaqi/models/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
chroma = chromadb.PersistentClient(path="/scratch/nlp/lijiaqi/models/chroma_RC")
#collection = chroma.delete_collection("math")
collection = chroma.create_collection("RAMC")

# path = './data/minif2f/'
# files = os.listdir(path)
# cnt = 1
# for file in files:
#     with open(path+file,'r') as f:
#         raw_text = f.read()
#         collection.add(
#             documents=[raw_text],
#             metadatas=[{"name": file}],
#             ids=["id_"+ file ])
#         print(cnt, file,len(raw_text.split()))
#     cnt += 1