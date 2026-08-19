"""
build_graph.py —— 从比赛数据灌股权图谱（只读 2/clean.xlsx）
只取每公司最新一期前十大股东 ≈ 6161 家 × 10 ≈ 6 万条关系
"""
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "12345678")   # 和你 shareholder_graph.py 里一致
DATA_URI = Path(__file__).parent.parent / "data" / "race" / "2" / "clean.xlsx"
R5_URI = Path(__file__).parent.parent / "data" / "race" / "5" / "rr_main_202605281537.csv"
STOCK_LIST_URI = Path(__file__).parent.parent / "data" / "market" / "stock_list.csv"  # akshare 全市场代码名称

def to_code(wind_code):
    """'002742.SZ' → '002742'，与查询端传的 6 位 code 对齐"""
    return str(wind_code).split(".")[0]

def norm(name):
    """去掉'股份有限公司/有限责任公司/有限公司'等后缀 + 空格(含全角)，让全称和简称能对上。
    注意顺序：先 replace 长的再 replace 短的，否则'集团股份有限公司'会被先吃掉'股份有限公司'剩'集团'。"""
    n = str(name).strip()
    for suf in ["集团股份有限公司", "股份有限公司", "有限责任公司", "有限公司"]:
        n = n.replace(suf, "")
    return n.replace(" ", "").replace("\u3000", "").strip()


def main():
    # ---- 1. 读数据 + 清洗 ----
    df = pd.read_excel(DATA_URI, sheet_name=0)
    df = df[["s_info_windcode", "s_holder_enddate", "s_holder_name",
             "s_holder_pct", "s_holder_holdercategory"]]
    df["code"] = df["s_info_windcode"].map(to_code)
    df = df.dropna(subset=["s_holder_name", "s_holder_pct",
                           "s_holder_holdercategory"])  # 脏行直接丢

    # ---- 2. 只取每公司最新一期 ----
    # transform("max")：返回一个和 df 等长的 Series，
    # 每行 = 该行所属 code 组里最大的 enddate → 布尔筛选即拿到"每组最新那批"
    latest_mask = df["s_holder_enddate"] == df.groupby("code")["s_holder_enddate"].transform("max")
    latest = df[latest_mask]

    # ---- 3. 每组按持股比例降序取前 10 ----
    latest = latest.sort_values(["code", "s_holder_pct"], ascending=[True, False])
    top10 = latest.groupby("code").head(10)   # groupby 后每组前 10 行

    print(f"待灌入: {top10['code'].nunique()} 家公司, {len(top10)} 条持股关系")
    # 新增：建"归一化公司简称 → 6位代码"字典。
    # 双来源：akshare 全市场列表(5547只, 主) + race/5 研报(3388只, 补)。
    # 单靠 race/5 会漏掉 ST/重整股(如 002742→冀衡医药) 和部分上市公司股东。
    code2name = {}
    for src, code_col, name_col in [
        (pd.read_csv(STOCK_LIST_URI, dtype={"code": str}), "code", "name"),
        (pd.read_csv(R5_URI, usecols=["sec_code", "sec_name"]), "sec_code", "sec_name"),
    ]:
        for _, row in src.dropna(subset=[name_col, code_col]).iterrows():
            code = to_code(row[code_col])
            code2name.setdefault(code, row[name_col])  # 先到先得：akshare 优先
    name2code = {}
    for code, name in code2name.items():
        name2code[norm(name)] = code
    print(f"归一化字典: {len(name2code)} 家上市公司（命中后股东会建成 Company 节点）")

    # ---- 4. 灌库 ----
    records = []
    for _, row in top10.iterrows():
        holder_code = name2code.get(norm(row["s_holder_name"]))  # 命中 = 该股东是上市公司
        records.append({
            "code": row["code"],
            "name": code2name.get(row["code"]),  # 新增：目标公司正式简称
            "holder": row["s_holder_name"],
            "holder_code": holder_code,
            "holder_name": code2name.get(holder_code) if holder_code else None,  # 新增：股东公司简称
            "category": int(row["s_holder_holdercategory"]),
            "ratio": float(row["s_holder_pct"]),
            "enddate": int(row["s_holder_enddate"]),
        })
    listed_count = sum(1 for r in records if r["holder_code"])
    print(f"其中上市公司股东(可穿透): {listed_count} 条, 普通股东: {len(records) - listed_count} 条")

    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as session:
        # 先建索引/约束：Holder.name 和 Company.code 唯一，MERGE 才走索引不快全库扫描
        session.run("CREATE CONSTRAINT company_code IF NOT EXISTS FOR (c:Company) REQUIRE c.code IS UNIQUE")
        session.run("CREATE CONSTRAINT holder_name IF NOT EXISTS FOR (h:Holder) REQUIRE h.name IS UNIQUE")
        print("索引就绪")

        session.run("MATCH (n) DETACH DELETE n")  # 清空重建
        print("已清空")

        # (a) 上市公司股东 → 建 Company 节点（带 code，链条能穿过去）
        listed = [r for r in records if r["holder_code"]]
        for i in range(0, len(listed), 500):
            batch = listed[i:i + 500]
            try:
                session.run("""
                                UNWIND $batch AS r
                                MERGE (c:Company {code: r.code})
                                SET c.name = coalesce(c.name, r.name)
                                MERGE (h:Company {code: r.holder_code})
                                SET h.name = coalesce(h.name, r.holder_name, r.holder)
                                MERGE (h)-[:HOLDS_SHARE {ratio: r.ratio, enddate: r.enddate}]->(c)
                            """, batch=batch)
            except Exception as e:
                print(f"  [a] batch {i}-{i + len(batch)} 失败: {e}", flush=True)
        print(f"  已灌上市公司股东关系 {len(listed)} 条", flush=True)

        # (b) 普通股东 → 维持 Holder 节点
        plain = [r for r in records if not r["holder_code"]]
        for i in range(0, len(plain), 500):
            batch = plain[i:i + 500]
            try:
                session.run("""
                                UNWIND $batch AS r
                                MERGE (c:Company {code: r.code})
                                SET c.name = coalesce(c.name, r.name)
                                MERGE (h:Holder {name: r.holder})
                                SET h.category = r.category
                                MERGE (h)-[:HOLDS_SHARE {ratio: r.ratio, enddate: r.enddate}]->(c)
                            """, batch=batch)
            except Exception as e:
                print(f"  [b] batch {i}-{i + len(batch)} 失败: {e}", flush=True)
            if i % 5000 == 0:
                print(f"  [b] 进度 {i}/{len(plain)}", flush=True)
        print(f"  已灌普通股东关系 {len(plain)} 条", flush=True)
    driver.close()
    print("灌库完成 ✅")
    driver.close()
    print("灌库完成 ✅")


if __name__ == "__main__":
    main()
