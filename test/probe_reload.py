"""验证重灌后的图谱数据：
1. 002742 的名字是否修复（应为"三圣股份"）
2. 2~5 跳路径是否还在
3. 002742 的多跳链路（预期: 王秀国→冀衡集团→002742）
"""
from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "12345678")

driver = GraphDatabase.driver(URI, auth=AUTH)
with driver.session() as s:
    # 1. 002742 名字
    r = s.run("MATCH (c:Company {code:'002742'}) RETURN c.name AS name, c.code AS code").single()
    print("1) 002742 name =", r["name"] if r else "NOT FOUND")

    # 2. 节点/关系统计
    r = s.run("MATCH (n) RETURN labels(n)[0] AS lab, count(*) AS cnt ORDER BY cnt DESC").data()
    print("2) 节点统计:", r)
    r = s.run("MATCH ()-[r:HOLDS_SHARE]->() RETURN count(r) AS cnt").single()
    print("   关系总数:", r["cnt"])

    # 3. 2~5 跳路径数
    r = s.run("""
        MATCH path = (a)-[:HOLDS_SHARE*2..5]->(c:Company)
        WHERE ALL(n IN nodes(path) WHERE n:Company)
        RETURN count(DISTINCT path) AS cnt
    """).single()
    print("3) 全 Company 的 2~5 跳路径:", r["cnt"])

    # 4. 002742 的最长链路
    r = s.run("""
        MATCH path = (controller)-[:HOLDS_SHARE*1..5]->(c:Company {code:'002742'})
        WHERE NOT (controller)<-[:HOLDS_SHARE]-()
        RETURN [n IN nodes(path) | coalesce(n.name, n.code)] AS chain,
               [r IN relationships(path) | r.ratio] AS ratios
        ORDER BY length(path) DESC LIMIT 10
    """).data()
    print("4) 002742 顶层链路:")
    for p in r:
        print("   ", p["chain"], p["ratios"])
driver.close()
