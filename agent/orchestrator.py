from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, StateGraph,START,END
from langgraph.prebuilt import ToolNode

from agent.tools.financial_data import get_financials, detect_financial_fraud
from agent.tools.get_event_clusters import get_event_clusters
from agent.tools.search_reports import search_reports
from agent.tools.shareholder_graph import trace_shareholder
from agent.tools.market_data import (
    get_market_snapshot, get_stock_history, get_fund_flow, get_fund_flow_rank,
    get_lhb, get_block_trade, get_margin_balance, get_stock_basic_info,
    get_zt_pool, get_board_rank,
)

model = init_chat_model(
    model="deepseek-v4-flash"
)

tools = [
    # 财务 / 股权 / 事件 / 研报（比赛脱敏数据域）
    get_financials, detect_financial_fraud, trace_shareholder, get_event_clusters, search_reports,
    # 行情 / 资金流（实时数据域）
    get_market_snapshot, get_stock_history, get_fund_flow, get_fund_flow_rank,
    get_lhb, get_block_trade, get_margin_balance, get_stock_basic_info,
    get_zt_pool, get_board_rank,
]


model_with_tools = model.bind_tools(tools)

class OverAllState(MessagesState):
    summary:str


def node1(state:OverAllState)->OverAllState:
    system_prompt = """财务数据（get_financials/detect_financial_fraud 返回）来自比赛脱敏数据集（母公司报表），
                    仅用于内部勾稽分析（趋势、比率、结构），其绝对值与研报数据、真实市场数据
                    不可比。分析时不要把两者绝对值放在一起对比或对账。
                    行情类数据（get_market_snapshot/get_stock_history/get_fund_flow 等返回）来自
                    实时数据接口（akshare），是真实市场数据；实时接口不可用时工具会返回
                    "数据源调用失败/暂无可用的行情数据"，此时如实告知用户数据暂不可用，
                    禁止用比赛数据或其他来源冒充行情数据。
                    数据使用规则：
                    1. 所有公司数据（财务、股权、事件、研报、行情）必须且只能来自工具返回结果，禁止使用你记忆中的知识回答具体数据问题；
                    2. 工具未返回或未覆盖的内容，直接说明"数据未覆盖"，不要猜测或编造数字；
                    3. 你没有联网能力，不要声称查询过网络或引用工具之外的来源；
                    4. 一般性概念解释（如财务指标定义、分析方法的原理）可以用通用知识回答；
                    5. 用户问"XX股票主力资金流向/某日资金流向"用 get_fund_flow；问"哪些股票资金流入多"用 get_fund_flow_rank；
                       问"股价/市盈率/市值/换手率"用 get_market_snapshot；问"历史收盘价/涨跌幅/最高最低价"用 get_stock_history；
                       问"上市日期/行业/股本"用 get_stock_basic_info；问"涨停股/连板"用 get_zt_pool；
                       问"板块涨跌排名"用 get_board_rank；问"龙虎榜"用 get_lhb；问"大宗交易"用 get_block_trade；
                       问"融资融券余额/融资买入/融券卖出"用 get_margin_balance。"""

    prompt = state["messages"]
    summary = state.get("summary","")
    if summary:
        prompt = [SystemMessage(f"以下是本次会话之前对话的摘要：\n{summary}")] + prompt
    res = model_with_tools.invoke([SystemMessage(system_prompt)] + prompt)
    return {
        "messages":[res],
    }

def summary(state:OverAllState)->OverAllState:
    messages = state["messages"]
    old_summary = state.get("summary","")
    if not old_summary:
        prompt = f"把以下对话总结成 300 字以内的摘要，保留公司名、结论和关键数字：\n{messages}"
    else:
        prompt = f"以下是已有摘要：\n{old_summary}\n\n这是新增对话：\n{messages}\n\n请合并更新成一段 300 字以内的新摘要，保留公司名、结论和关键数字。"
    new_summary = model.invoke(prompt).content
    deletes = [RemoveMessage(id = m.id) for m in state["messages"][:-2]]
    return {
        "summary":new_summary,
        "messages":deletes,
    }

def should_use_tools(state:OverAllState)->Literal["tool_node","summary",END]:
    if state["messages"][-1].tool_calls:
        return "tool_node"
    elif len(state["messages"]) > 30:
        return "summary"
    else:
        return END

builder = StateGraph(state_schema=OverAllState)
builder.add_node("node1",node1)
builder.add_node("tool_node",ToolNode(tools=tools,handle_tool_errors=lambda e: f"工具执行失败: {type(e).__name__}: {e}"))
builder.add_node("summary",summary)

builder.add_edge(START,"node1")
builder.add_conditional_edges("node1",should_use_tools,path_map=["tool_node","summary",END])
builder.add_edge("tool_node","node1")
builder.add_edge("summary",END)
checkpointer = InMemorySaver()

config = {
    "configurable":{
        "thread_id":"1"
    }
}

graph = builder.compile(checkpointer=checkpointer)

if __name__ == '__main__':
    while True:
        text = input("你: ")
        if text.strip().lower() in ("quit", "exit", "q"):
            break
        result = graph.invoke({"messages": [HumanMessage(text)]}, config=config)
        for m in result["messages"]:
            if isinstance(m,AIMessage) and m.tool_calls:
                print(">>> 工具调用:", [c["name"] for c in m.tool_calls])
        print("AI:", result["messages"][-1].content)