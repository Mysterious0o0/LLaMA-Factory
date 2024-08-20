import json
from uuid import NAMESPACE_DNS, uuid5

import lmdb
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from utils import export_lmdb, log


EMBEDDING_MODEL_NAME = "/home/songfucheng/model/embedding/bge-m3"
CACHE_QA_PATH = "dataDB/qa/vector"
JSON_PATH = "training_data.json"
QID2QA_PATH = "dataDB/qa/qid2qa"
PREFIX = "qa"

embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def encoding_question(path):
    f = open(path, "r", encoding="utf-8")
    data = json.loads(f.read())
    f.close()
    for qa in data:
        question = qa["input"].split("输入输出要素")[0].replace("#", "")
        answers = qa["output"]
        qid = uuid5(NAMESPACE_DNS, question)
        yield qid, question, answers


def save_vector_and_db():
    qids = []
    questions = []
    qid2qa = []
    # db_fai = FAISS.from_texts(["卍"], embeddings)
    env = lmdb.open(QID2QA_PATH, map_size=1099511627776)

    for qid, question, answers in encoding_question(JSON_PATH):
        qids.append({"qid": qid})
        questions.append(question)
        qid2qa.append((qid, "### question: %s \n ### answers: %s" % (question, answers)))
    db_fai = FAISS.from_texts(questions, embeddings, metadatas=qids)
    # db_fai.merge_from(db)
    db_fai.save_local(CACHE_QA_PATH)
    log.info("faiss save success")
    export_lmdb(env, qid2qa, prefix=PREFIX)
    log.info("lmdb save success")
    return db_fai


vector_store = FAISS.load_local(CACHE_QA_PATH, embeddings)
lmdb_env = lmdb.open(QID2QA_PATH, map_size=1099511627776)


def search_qa(query, sim_p=0.4):
    library_result = []
    txn = lmdb_env.begin(write=False)
    search_result = vector_store.similarity_search_with_score(query=query)
    search_result = sorted(search_result, key=lambda x: x[1], reverse=False)
    for res in search_result:
        if res[1] > sim_p:
            qid = res[0].metadata["qid"]
            qa = txn.get("qa-%s" % qid).encode("utf-8")
            if qa:
                result = "### similary: {:0.3} \n".format(res[1]) + qa.decode("utf-8")
                library_result.append(result)
    if library_result:
        return "\n".join(library_result)
    return None


if __name__ == "__main__":
    save_vector_and_db()
