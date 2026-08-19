import json
import uuid
import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage


@st.cache_resource
def load_graph():
    from agent.orchestrator import graph   # import 放函数内：只有第一次真正加载
    return graph


@st.cache_resource
def get_model():
    from agent.orchestrator import model   # 复用主 Agent 的 LLM（不重复初始化）
    return model


@st.cache_data
def load_stock_name():
    """代码 -> 名称，来自 akshare 全市场列表（离线文件）"""
    try:
        df = pd.read_csv("data/market/stock_list.csv", dtype={"code": str})
        return dict(zip(df["code"], df["name"]))
    except Exception:
        return {}


@st.cache_data(ttl=600)
def get_clusters(symbol: str):
    from agent.tools.get_event_clusters import get_event_clusters
    return get_event_clusters.invoke(symbol)


# ---------- 事件簇配色（中国股市惯例：红涨绿跌，风险用红橙） ----------
CLUSTER_COLORS = {
    "财务造假": "#d32f2f",
    "监管处罚": "#f57c00",
    "重整处置": "#7b1fa2",
    "退市风险": "#c62828",
    "股权变动": "#1565c0",
    "经营动态": "#2e7d32",
    "其他": "#757575",
}


def render_timeline(clusters: list) -> str:
    """把事件簇渲染成竖向 HTML 时间轴（按日期排序，簇着色）"""
    items = []
    for c in clusters:
        color = CLUSTER_COLORS.get(c["name"], "#757575")
        for a in c["announcements"]:
            items.append({"date": a["date"], "title": a["title"],
                          "cluster": c["name"], "color": color})
    items.sort(key=lambda x: x["date"])

    parts = ['<div style="position:relative;padding-left:26px;border-left:2px solid #d0d0d0;">']
    for it in items:
        parts.append(
            f'<div style="position:relative;margin-bottom:14px;">'
            f'<div style="position:absolute;left:-31px;top:5px;width:11px;height:11px;'
            f'border-radius:50%;background:{it["color"]};border:2px solid #fff;'
            f'box-shadow:0 0 0 1px {it["color"]};"></div>'
            f'<div style="font-size:12px;color:#999;">{it["date"]} '
            f'<span style="color:{it["color"]};border:1px solid {it["color"]};'
            f'border-radius:8px;padding:0 6px;font-size:11px;margin-left:4px;">{it["cluster"]}</span></div>'
            f'<div style="font-size:14px;color:#333;margin-top:2px;line-height:1.5;">{it["title"]}</div>'
            f'</div>'
        )
    parts.append("</div>")
    return "".join(parts)


def generate_story(symbol: str, company: str, raw: str) -> str:
    """LLM 基于事件簇生成事件脉络叙事"""
    model = get_model()
    prompt = (
        f"你是资深证券分析师。以下是 {company}({symbol}) 的公告事件簇数据，"
        f"请生成 250 字以内的事件脉络叙事：按时间顺序讲清这家公司发生了什么、"
        f"事件之间的因果逻辑、当前状态如何。只基于给定数据，不要编造任何信息：\n{raw}"
    )
    try:
        return model.invoke(prompt).content
    except Exception as e:
        return f"叙事生成失败（LLM 调用出错）: {e}"


# ==================== 页面 ====================

st.set_page_config(page_title="智能投研助手 Demo", page_icon="📈", layout="wide")

tab_chat, tab_events = st.tabs(["💬 智能问答", "📅 事件时间轴"])

# ---------- Tab 1：智能问答 ----------
with tab_chat:
    st.title("智能投研助手 Demo")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.write(content)

    if prompt := st.chat_input("请输入你的问题（如：冀衡医药的股权结构？）"):
        st.session_state.messages.append(("user", prompt))
        with st.chat_message("user"):
            st.write(prompt)

        graph = load_graph()
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        before = len(graph.get_state(config).values.get("messages", []) or [])

        def token_gen():
            for chunk, _ in graph.stream(
                {"messages": [HumanMessage(prompt)]},
                config=config,
                stream_mode="messages",
            ):
                if chunk.content:
                    yield chunk.content

        with st.chat_message("assistant"):
            answer = st.write_stream(token_gen())

            final = graph.get_state(config).values.get("messages", []) or []
            tool_names = []
            for m in final[before:]:
                if isinstance(m, AIMessage) and m.tool_calls:
                    tool_names += [c["name"] for c in m.tool_calls]
            if tool_names:
                uniq = list(dict.fromkeys(tool_names))
                st.caption("调用了 " + str(len(uniq)) + " 个工具：" + " · ".join(uniq))
        st.session_state.messages.append(("assistant", answer))

# ---------- Tab 2：事件时间轴 ----------
with tab_events:
    st.title("📅 公司事件时间轴")
    st.caption("基于 race/3 公告库（7311 条公告 · 2585 家公司）自动聚类，AI 生成事件脉络叙事")

    name_map = load_stock_name()
    col1, col2 = st.columns([2, 3])
    with col1:
        symbol = st.text_input("6 位股票代码", value="002742", max_chars=6)
    with col2:
        st.markdown("###")  # 对齐
        if symbol:
            company = name_map.get(symbol, "")
            if company:
                st.markdown(f"**{company}**（{symbol}）")
            else:
                st.markdown(f"**{symbol}**（未在股票列表，可能已退市/改名）")

    if symbol:
        raw = get_clusters(symbol)
        data = json.loads(raw)

        if not data.get("clusters"):
            st.warning(data.get("note", "该股票暂未收录公告数据"))
        else:
            st.markdown(f"共 **{data['count']}** 条公告，聚类为 **{len(data['clusters'])}** 个事件簇")

            # 簇概览卡片
            cols = st.columns(len(data["clusters"]))
            for col, c in zip(cols, data["clusters"]):
                color = CLUSTER_COLORS.get(c["name"], "#757575")
                with col:
                    st.markdown(
                        f'<div style="border:1px solid {color};border-radius:8px;padding:10px;text-align:center;">'
                        f'<div style="color:{color};font-weight:bold;">{c["name"]}</div>'
                        f'<div style="font-size:24px;font-weight:bold;">{c["count"]}</div>'
                        f'<div style="font-size:12px;color:#999;">{c["period"]}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            st.subheader("事件时间轴")
            st.markdown(render_timeline(data["clusters"]), unsafe_allow_html=True)

            st.markdown("---")
            if st.button("✨ AI 生成事件脉络叙事", type="primary"):
                with st.spinner("正在分析事件脉络..."):
                    story = generate_story(symbol, company or symbol, raw)
                st.markdown(story)
