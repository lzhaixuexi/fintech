import json

from langchain_core.tools import tool
from neo4j import GraphDatabase


@tool
def trace_shareholder(symbol:str)->str:
    """查询上市公司的股权穿透链路，找出实际控制人及完整持股路径。
        当用户询问谁控制了某家公司、实际控制人是谁、股权穿透、股东链路时调用。

        Args:
            symbol: 6位股票代码，如"002242"
    """
    driver = GraphDatabase.driver(uri="neo4j://127.0.0.1:7687", auth=("neo4j", "12345678"))
    session = driver.session(database="neo4j")
    result = session.run("""MATCH path = (controller)-[:HOLDS_SHARE*1..5]->(c:Company {code: $code})
                    WHERE NOT (controller)<-[:HOLDS_SHARE]-()
                    RETURN controller.name AS controller,
                    labels(controller) AS controller_type,
                    [n IN nodes(path) | [labels(n)[0], coalesce(n.name, n.code)]] AS node_chain,
                    [r IN relationships(path) | r.ratio] AS ratio_chain,
                    reduce(t=1.0, r IN relationships(path) | t * r.ratio / 100.0) AS total_pct""",code = symbol)
    records = list(result)
    if not records:
        session.close()
        driver.close()
        return json.dumps({"company": symbol, "paths": [], "note": "该股票暂未收录股权图谱数据"}, ensure_ascii=False)
    paths = []
    for record in records:
        controller = record["controller"]
        controller_type = record["controller_type"]
        node_chain = record["node_chain"]      # [[label, name_or_code], ...]
        ratio_chain = record["ratio_chain"]
        total_pct = record["total_pct"]
        chain = []
        for (from_type, from_name), (to_type, to_name), ratio in zip(node_chain[:-1], node_chain[1:], ratio_chain):
            chain.append({
                "from": {"type": from_type, "name": from_name},
                "to": {"type": to_type, "name": to_name},
                "ratio": ratio,
            })
        paths.append({
            "controller": controller,
            "type": controller_type[0],
            "chain": chain,
            "total_pct": round(total_pct, 4),
        })

    session.close()
    driver.close()

    result_dict = {
        "company":symbol,
        "paths":paths,
    }
    return json.dumps(result_dict,ensure_ascii=False)

if __name__ == '__main__':
    print(trace_shareholder.invoke({"symbol": "002742"}))  # 预期: 冀衡集团 17.54 在列
    print(trace_shareholder.invoke({"symbol": "000531"}))  # 预期: 招商局金控→招商证券→广州发展→广州港 4 跳穿透链
    print(trace_shareholder.invoke({"symbol": "999999"}))  # 预期: paths 空 + note "该股票暂未收录股权图谱数据"


