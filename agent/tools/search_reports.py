import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.tools import tool

# 模型已固化在项目 data/models/，禁止联网下载（比赛现场无网也能跑）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).parent.parent.parent
CHROMA_DIR = ROOT / "data" / "chroma"
EMBED_MODEL = str(ROOT / "data" / "models" / "bge-small-zh-v1.5")  # 本地路径，不联网

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    name="research_reports",
    embedding_function=SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL),
    metadata={"hnsw:space": "cosine"},
)

@tool
def search_reports(query:str,sec_code:str = ""):
    """在 5.5 万篇券商研报库中做语义检索，返回标题、机构、评级、日期和摘要片段。

    Args:
        query: 检索主题，如"垃圾焚烧 业绩点评"、"算力产业链"
        sec_code: 可选，6 位股票代码如 "002242"，传入则只检索该公司的研报
    """
    results = collection.query(
        query_texts=query,  # ← 传文本，自动 embed
        n_results=5,
        where={"sec_code": sec_code} if sec_code else None,  # ← 字典过滤，比 expr 直观
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    rst = []
    for doc,meta in zip(docs, metas):
        head = f"【{meta['title']}】{meta['org_name']} | 评级：{meta['rating_org']} | {meta['publish_date']}"
        body = doc[:300]
        rst.append(head + "\n" + body)
    if not rst:
        return "未找到相关研报"
    return "\n\n".join(rst)

if __name__ == '__main__':
    print(search_reports.invoke({"query": "永兴股份 垃圾焚烧"}))


