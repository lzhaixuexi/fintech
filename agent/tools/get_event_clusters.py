import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

DATA_URL = Path(__file__).parent.parent.parent / "data" / "race" / "3" / "clean.xlsx"
CLUSTER_RULES = [
    ("财务造假", ["资金占用", "违规担保", "会计差错", "非标"]),
    ("监管处罚", ["立案", "处罚", "警示", "谴责", "责令改正", "监管措施", "纪律处分"]),
    ("重整处置", ["重整", "破产", "清算", "冻结"]),
    ("退市风险", ["退市", "停牌", "风险警示"]),
    ("股权变动", ["质押", "增持", "减持", "回购", "控制权", "实控人", "收购", "出售"]),
    ("经营动态", ["业绩", "中标", "合同", "投资", "募集", "分红"]),
]
_NEWS_DF = None

def _load_news():
    global _NEWS_DF
    if _NEWS_DF is None:
        _NEWS_DF = pd.read_excel(
            DATA_URL, sheet_name="Sheet2",
            usecols=["s_info_windcode", "ann_dt", "n_info_title","n_info_fcode", "n_info_windlink"]
        )
        _NEWS_DF["code"] = _NEWS_DF["s_info_windcode"].map(lambda x: str(x).split(".")[0])
    return _NEWS_DF

def _classify(title:str)->str:
    for cluster_name,keywords in CLUSTER_RULES:
        for kw in keywords:
            if kw in title:
                return cluster_name
    return "其他"

@tool
def get_event_clusters(symbol:str)->str:
    """查询上市公司公告并按事件主题自动归类（事件簇）。
    当用户询问公司近期动态、风险事件、事件脉络、公告分类、发生了什么大事时调用。
    返回按主题分组的公告簇，每个簇内含按日期排序的公告明细。

    Args:
        symbol:6位股票代码，如"002742"
    """
    df = _load_news()
    sub = df[df["code"] == symbol].sort_values("ann_dt")

    if sub.empty:
        return json.dumps({"company":symbol,"st_flag":False,"clusters":[],"note":"该股票暂未收录公告数据"},ensure_ascii=False)

    stripped = [str(t).split(":",1)[-1] for t in sub["n_info_title"]]
    st_flag = any("ST" in str(t).split(":",1)[0] for t in sub["n_info_title"])

    buckets = {}
    for i,row in enumerate(sub.itertuples()):
        cluster = _classify(stripped[i])
        buckets.setdefault(cluster,[]).append({
            "date":row.ann_dt.strftime("%Y-%m-%d"),
            "title":stripped[i],
            "type":row.n_info_fcode,
            "link":row.n_info_windlink,
        })
    clusters = []
    for name, items in buckets.items():
        clusters.append({
            "name": name,
            "count": len(items),
            "period": f"{items[0]['date']} ~ {items[-1]['date']}",
            "announcements": items,
        })

    return json.dumps({"company": symbol, "st_flag": st_flag,
                       "count": len(sub), "clusters": clusters}, ensure_ascii=False)

if __name__ == '__main__':
    print(get_event_clusters.invoke("999999"))
