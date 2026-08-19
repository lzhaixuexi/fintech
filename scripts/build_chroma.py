import os
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 模型已固化在项目 data/models/，禁止联网下载（比赛现场无网也能跑）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).parent.parent
DATA_CSV = ROOT / "data" / "race" / "5" / "rr_main_202605281537.csv"
CHROMA_DIR = ROOT / "data" / "chroma"
EMBED_MODEL = str(ROOT / "data" / "models" / "bge-small-zh-v1.5")  # 本地路径，不联网
BATCH_SIZE = 2000

def init_chroma():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name="research_reports",
        embedding_function=SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL),
        metadata={"hnsw:space": "cosine"},
    )
    return collection

def load_data():
    df = pd.read_csv(DATA_CSV,usecols=[
        "report_id", "title", "abstract", "sec_code", "sec_name",
        "publish_date", "report_type", "report_sub_type",
        "rating_org", "org_name", "industry_l1",
    ])
    df["text"] = df["title"].fillna("") + " " + df["abstract"].fillna("")
    df["report_id"] = df["report_id"].astype(str)
    print(f"加载{len(df)}条研报")
    return df


def build(collection,df):
    total = len(df)
    for i in range(0,total,BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE]
        collection.add(
            ids = batch["report_id"].tolist(),
            documents=batch["text"].tolist(),
            metadatas = [
                {
                    "sec_code": str(r.sec_code) if pd.notna(r.sec_code) else "",
                    "sec_name": str(r.sec_name) if pd.notna(r.sec_name) else "",
                    "publish_date": str(r.publish_date) if pd.notna(r.publish_date) else "",
                    "report_type": str(r.report_type) if pd.notna(r.report_type) else "",
                    "report_sub_type": str(r.report_sub_type) if pd.notna(r.report_sub_type) else "",
                    "rating_org": str(r.rating_org) if pd.notna(r.rating_org) else "",
                    "org_name": str(r.org_name) if pd.notna(r.org_name) else "",
                    "industry_l1": str(r.industry_l1) if pd.notna(r.industry_l1) else "",
                    "title": str(r.title) if pd.notna(r.title) else "",
                }
                for r in batch.itertuples()
            ],
        )
        print(f"  进度: {min(i + BATCH_SIZE, total)}/{total} ({min(i + BATCH_SIZE, total) * 100 // total}%)")

def main():
    df = load_data()
    collection = init_chroma()
    print(f"Chroma 初始化完成，开始灌库...")
    build(collection, df)
    print(f"完成！collection 共 {collection.count()} 条记录")
    print(f"存储位置: {CHROMA_DIR}")

if __name__ == '__main__':
    main()