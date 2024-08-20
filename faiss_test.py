import time

import torch
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


embedding_model_name = "text2vec-large-chinese/"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
s = time.time()


def init_source_vector(path, vector_store_path):
    docs = []
    loader = UnstructuredFileLoader(path, mode="elements", content_type="")
    doc = loader.load()
    # print(doc)
    docs.extend(doc)
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(vector_store_path)
    print("建立索引：", time.time() - s)
    # search_result = vector_store.similarity_search_with_score(query)
    return vector_store


def search(index_path, query):
    vector_store = FAISS.load_local(index_path, embeddings)
    search_result = vector_store.similarity_search_with_score(query)
    return search_result


vector_store = init_source_vector("company_entity_sim_1.txt", "company")
ii = vector_store.similarity_search_with_score("马鞍山")[:10]
print("检索：", time.time() - s)
for i in ii:
    print(i)

# for i in search("./company", "马鞍山")[:10]:
#     print(i)
print(time.time() - s)


def read_text(path):
    f = open(path, "r", encoding="utf-8")
    doc = f.readlines()
    # print(doc)
    f.close()
    return doc


path = "company_entity_sim.txt"
corpus = read_text(path)
print(corpus)

corpus = ["卍", "宝信软件", "马鞍山钢铁"]

db1 = FAISS.from_texts(corpus, embeddings)
db1.save_local("company")
for i in db1.similarity_search_with_score("马鞍山")[:10]:
    print(i)


texts = ["宝信软件", "马鞍山钢铁"]

x = embeddings.embed_documents(texts)
print(torch.Tensor(x).size())
for i, t in enumerate(texts):
    print()

    print(torch.Tensor(embeddings.embed_query(t)).size())
