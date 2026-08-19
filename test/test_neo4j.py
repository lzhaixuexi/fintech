from neo4j import GraphDatabase

driver = GraphDatabase.driver(uri="neo4j://127.0.0.1:7687",auth=("neo4j","12345678"))
session = driver.session(database="neo4j")
result = session.run("CREATE (n:Test {name:'hello'}) RETURN n")
record = result.single()
print(record["n"])
session.run("MATCH (n) DETACH DELETE n")