"""market_data.py —— 行情 / 资金流 / 龙虎榜 / 大宗交易 / 两融 数据工具
数据源：akshare（东方财富/上交所/深交所 公开接口），全部带磁盘缓存 + 离线降级。
口径声明：本文件返回的是真实市场实时/历史数据，与比赛脱敏财务数据（race/4）不可比，
          使用时不得与比赛数据交叉对账。
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import akshare as ak
from langchain_core.tools import tool

from agent.tools._utils import (
    cached_ak, frames_to_records, normalize_symbol,
    parse_date_cn, pick_nearest_date, to_ak_market, to_ak_symbol,
)

# 各接口缓存 TTL（秒）
TTL_SPOT = 60          # 全市场快照
TTL_HIST = 3600 * 12   # 历史行情
TTL_FLOW = 3600 * 24   # 个股资金流
TTL_OTHER = 3600 * 24  # 龙虎榜/大宗/两融/基本信息


def _dstr(d: date | None = None) -> str:
    return (d or date.today()).strftime("%Y%m%d")


def _dash(d: date | None = None) -> str:
    return (d or date.today()).isoformat()


def _today_str() -> str:
    """今天日期字符串（避免参数名 date 遮蔽 datetime.date 类）"""
    return date.today().isoformat()


def _relative_start(text: str) -> date | None:
    """把 '近一周/近一月/近三月/近一年/今年/近N天/近N月' 解析成起始日期"""
    t = str(text or "").strip()
    today = date.today()
    if t in ("近一周", "近1周", "本周"):
        return today - timedelta(days=7)
    if t in ("近一月", "近1月", "本月"):
        return today - timedelta(days=30)
    if t in ("近三月", "近3月"):
        return today - timedelta(days=90)
    if t in ("近半年", "近6月"):
        return today - timedelta(days=180)
    if t in ("近一年", "近1年"):
        return today - timedelta(days=365)
    if t == "今年":
        return date(today.year, 1, 1)
    import re
    m = re.match(r"近(\d+)(天|日|周|月|年)", t)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = n * {"天": 1, "日": 1, "周": 7, "月": 30, "年": 365}[unit]
        return today - timedelta(days=days)
    return None


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _code(symbol: str) -> str:
    return normalize_symbol(symbol)


# ================================================================ 工具 1：行情快照

@tool
def get_market_snapshot(symbols: str = "", top_n: int = 20) -> str:
    """查询 A 股实时行情快照（最新价、涨跌幅、换手率、量比、市盈率-动态、市净率、总市值、流通市值等）。
    当用户询问某只/某些股票的股价、涨跌幅、市盈率、市值、换手率、量比时调用；
    不传股票代码时返回全市场涨幅榜前 top_n 名（用于'今天哪些股票涨得好/涨幅靠前'类问题）。

    Args:
        symbols: 逗号分隔的 6 位股票代码，可空（空 = 全市场涨幅榜）
        top_n: 返回条数上限，默认 20
    """
    codes = [_code(s) for s in symbols.split(",")] if symbols.strip() else []
    ok, df = cached_ak("spot_em", TTL_SPOT, ak.stock_zh_a_spot_em)

    # 单票查询时：全市场快照失败/未覆盖 → 回退单票盘口接口
    if (not ok or df.empty) and len(codes) == 1:
        ok2, bid = cached_ak(f"bid_ask_{codes[0]}", TTL_SPOT, ak.stock_bid_ask_em, symbol=codes[0])
        if ok2:
            rows = [{"item": str(r.get("item")), "value": r.get("value")} for _, r in bid.iterrows()]
            return _json({
                "data_source": "实时行情接口(akshare/东方财富)",
                "note": "与比赛脱敏财务数据不可比",
                "count": 1,
                "rows": rows,
            })
        return f"未找到代码 {symbols} 的实时行情（可能已退市/代码错误）"
    if not ok:
        return df

    try:
        df = df.sort_values("涨跌幅", ascending=False)
    except KeyError:
        pass

    if codes:
        sub = df[df["代码"].astype(str).isin(codes)]
        if sub.empty:
            return f"未找到代码 {symbols} 的实时行情（可能已退市/代码错误）"
        out = sub
    else:
        out = df.head(top_n)

    cols = ["代码", "名称", "最新价", "涨跌幅", "涨跌额", "今开", "昨收", "最高", "最低",
            "成交量", "成交额", "振幅", "换手率", "量比", "市盈率-动态", "市净率",
            "总市值", "流通市值", "年初至今涨跌幅"]
    cols = [c for c in cols if c in out.columns]
    return _json({
        "data_source": "实时行情接口(akshare/东方财富)",
        "note": "与比赛脱敏财务数据不可比",
        "count": len(out),
        "rows": frames_to_records(out[cols]),
    })


# ================================================================ 工具 2：历史行情

@tool
def get_stock_history(symbol: str, start_date: str = "", end_date: str = "",
                      period: str = "daily", adjust: str = "qfq") -> str:
    """查询个股历史行情（K线），支持自然语言日期，自动计算区间统计。
    当用户询问历史股价、某日收盘价、近N月涨跌幅、区间最高/最低价、均线、K线时调用。

    Args:
        symbol: 6位股票代码
        start_date: 开始日期，支持 '2025-03-18' / '20250318' / '3月24日' / '8.1号'，空=默认近90天
        end_date: 结束日期（同上），空=今天
        period: daily/weekly/monthly，默认 daily
        adjust: qfq(前复权)/hfq(后复权)/空串(不复权)，默认 qfq
    """
    symbol = _code(symbol)
    # 起始日期：支持相对区间（近一周/近一月/今年/近N天）与具体日期；默认近90天
    if start_date:
        s = parse_date_cn(start_date) or (_relative_start(start_date).isoformat() if _relative_start(start_date) else None)
    else:
        s = (date.today() - timedelta(days=90)).isoformat()
    if s is None:
        return f"无法解析起始日期: {start_date}（支持 '近一周/近一月/今年' 或 '2025-03-18' 等）"
    e = parse_date_cn(end_date) if end_date else _today_str()
    e = e or _today_str()
    s_compact, e_compact = s.replace("-", ""), e.replace("-", "")

    key = f"hist_{symbol}_{period}_{adjust}_{s_compact}_{e_compact}"
    ok, data = cached_ak(key, TTL_HIST, ak.stock_zh_a_hist, symbol=symbol, period=period,
             start_date=s_compact, end_date=e_compact, adjust=adjust)
    if not ok and period == "daily":
        # 回退数据源：新浪历史行情（finance.sina.com.cn，仅日线）
        sina_sym = to_ak_symbol(symbol)
        ok2, df2 = cached_ak(f"hist_sina_{symbol}_{adjust}_{s_compact}_{e_compact}", TTL_HIST,
                             ak.stock_zh_a_daily, symbol=sina_sym,
                             start_date=s_compact, end_date=e_compact, adjust=adjust)
        if ok2:
            df2 = df2.rename(columns={"date": "日期", "open": "开盘", "close": "收盘", "high": "最高",
                                      "low": "最低", "volume": "成交量", "amount": "成交额",
                                      "turnover": "换手率"})
            df2["涨跌幅"] = (df2["收盘"].pct_change() * 100).round(2)
            data = df2
        else:
            return data
    df = data
    records = frames_to_records(df)
    if not records:
        return f"股票 {symbol} 在 {s}~{e} 区间暂无行情数据"

    # 区间统计
    closes = [r["收盘"] for r in records if r.get("收盘") is not None]
    stats = {
        "区间最高价": max(closes) if closes else None,
        "区间最低价": min(closes) if closes else None,
        "区间首个交易日收盘": records[0].get("收盘"),
        "区间最后交易日收盘": records[-1].get("收盘"),
        "区间交易日数": len(records),
    }
    if len(closes) >= 2 and closes[0]:
        stats["区间涨跌幅"] = round((closes[-1] / closes[0] - 1) * 100, 2)
    # 收盘价 > 阈值的次数（如'华林证券收盘价超过16块钱的次数'）
    first_close = records[0].get("收盘")
    if isinstance(first_close, (int, float)):
        stats["区间收盘价平均值"] = round(sum(closes) / len(closes), 2)

    # 只回传最近 30 根 K 线，控制 token
    tail = records[-30:]
    return _json({
        "symbol": symbol,
        "data_source": "历史行情接口(akshare/东方财富)",
        "range": f"{s} ~ {e}",
        "period": period,
        "adjust": adjust,
        "stats": stats,
        "recent_kline": tail,
    })


# ================================================================ 工具 3：个股主力资金流向

@tool
def get_fund_flow(symbol: str, date: str = "") -> str:
    """查询个股主力资金流向（主力/超大单/大单/中单/小单 净流入额与净占比）。
    当用户询问某只股票的主力资金动向、资金流向、净流入、X月X日资金流向时调用。

    Args:
        symbol: 6位股票代码
        date: 查询日期，支持 '2025-03-24' / '3月24日' 等，空=最近交易日
    """
    symbol = _code(symbol)
    market = to_ak_market(symbol)
    # 个股资金流按股票缓存全量（同会话几十连问的关键），再本地过滤日期
    ok, data = cached_ak(f"flow_{symbol}", TTL_FLOW, ak.stock_individual_fund_flow, stock=symbol, market=market)
    if not ok:
        return data
    df = data
    target = (parse_date_cn(date) or _today_str()) if date else _today_str()
    ok2, result = pick_nearest_date(df, "日期", target)
    if not ok2:
        return result
    return _json({
        "symbol": symbol,
        "data_source": "主力资金流接口(akshare/东方财富)",
        "note": "与比赛脱敏财务数据不可比",
        "rows": result,
    })


# ================================================================ 工具 4：资金流排行

@tool
def get_fund_flow_rank(indicator: str = "今日", top_n: int = 20) -> str:
    """查询全市场主力资金净流入排行（今日/3日/5日/10日）。
    当用户询问哪些股票主力资金流入最多、今日资金净流入排名、连续流入的股票时调用。

    Args:
        indicator: 今日 / 3日 / 5日 / 10日，默认今日
        top_n: 返回条数上限，默认 20
    """
    if indicator not in ("今日", "3日", "5日", "10日"):
        return f"indicator 仅支持: 今日/3日/5日/10日，收到 {indicator}"
    ok, data = cached_ak(f"flow_rank_{indicator}", TTL_FLOW, ak.stock_individual_fund_flow_rank, indicator=indicator)
    src = "资金流排行接口(akshare/东方财富)"
    if not ok:
        # 回退数据源：同花顺资金流排行（data.10jqka.com.cn）
        ths_map = {"今日": "即时", "3日": "3日排行", "5日": "5日排行", "10日": "10日排行"}
        ok2, df2 = cached_ak(f"flow_rank_ths_{indicator}", TTL_FLOW,
                             ak.stock_fund_flow_individual, symbol=ths_map[indicator])
        if ok2:
            df2 = df2.sort_values("净额", ascending=False)
            keep = ["股票代码", "股票简称", "最新价", "涨跌幅", "净额", "换手率"]
            keep = [c for c in keep if c in df2.columns]
            return _json({
                "data_source": "资金流排行接口(akshare/同花顺)",
                "indicator": indicator,
                "count": min(top_n, len(df2)),
                "rows": frames_to_records(df2.head(top_n)[keep]),
            })
        return data
    df = data
    col = "今日主力净流入-净额" if indicator == "今日" else f"{indicator.replace('日','')}日主力净流入-净额"
    if col in df.columns:
        df = df.sort_values(col, ascending=False)
    keep = ["代码", "名称", "最新价", "今日涨跌幅", col, f"{indicator}超大单净流入-净额"]
    keep = [c for c in keep if c in df.columns]
    return _json({
        "data_source": "资金流排行接口(akshare/东方财富)",
        "indicator": indicator,
        "count": min(top_n, len(df)),
        "rows": frames_to_records(df.head(top_n)[keep]),
    })


# ================================================================ 工具 5：龙虎榜

@tool
def get_lhb(date: str = "", symbol: str = "") -> str:
    """查询龙虎榜上榜明细（上榜原因、龙虎榜净买额、机构席位等）。
    当用户询问某只股票是否上龙虎榜、今天的龙虎榜、龙虎榜机构净买时调用。

    Args:
        date: 查询日期，空=今天
        symbol: 可选，6位股票代码，只返回该股票的上榜记录
    """
    d = parse_date_cn(date) or _today_str()
    d_compact = d.replace("-", "")
    ok, data = cached_ak(f"lhb_{d_compact}", TTL_OTHER, ak.stock_lhb_detail_em, start_date=d_compact, end_date=d_compact)
    if not ok:
        return data
    df = data
    records = frames_to_records(df)
    if not records:
        return f"{d} 龙虎榜暂无数据（可能非交易日）"
    if symbol:
        code = _code(symbol)
        records = [r for r in records if str(r.get("代码", "")).zfill(6) == code]
        if not records:
            return f"{code} 在 {d} 未上龙虎榜"
    cols = ["代码", "名称", "上榜日", "收盘价", "涨跌幅", "龙虎榜净买额", "龙虎榜买入额",
            "龙虎榜卖出额", "换手率", "流通市值", "上榜原因", "解读"]
    cols = [c for c in cols if c in df.columns]
    if "龙虎榜净买额" in df.columns:
        records = sorted(records, key=lambda r: (r.get("龙虎榜净买额") or 0), reverse=True)
    return _json({
        "date": d,
        "data_source": "龙虎榜接口(akshare/东方财富)",
        "count": len(records),
        "rows": [{c: r.get(c) for c in cols} for r in records[:30]],
    })


# ================================================================ 工具 6：大宗交易

@tool
def get_block_trade(date: str = "") -> str:
    """查询大宗交易每日明细（成交价、折溢价率、成交额、买卖营业部）。
    当用户询问今天/某日的大宗交易情况、大宗交易成交额排名时调用。

    Args:
        date: 查询日期，空=今天
    """
    d = parse_date_cn(date) or _today_str()
    d_compact = d.replace("-", "")
    ok, data = cached_ak(f"dzjy_{d_compact}", TTL_OTHER, ak.stock_dzjy_mrmx, symbol="A股", start_date=d_compact, end_date=d_compact)
    if not ok:
        return data
    df = data
    records = frames_to_records(df)
    if not records:
        return f"{d} 大宗交易暂无数据（可能非交易日）"
    if "成交额" in df.columns:
        records = sorted(records, key=lambda r: (r.get("成交额") or 0), reverse=True)
    return _json({
        "date": d,
        "data_source": "大宗交易接口(akshare/东方财富)",
        "count": len(records),
        "rows": records[:20],
    })


# ================================================================ 工具 7：融资融券

@tool
def get_margin_balance(symbol: str, date: str = "") -> str:
    """查询个股融资融券余额（融资买入额、融资余额、融券卖出量、融券余量等）。
    当用户询问某只股票的融资余额、融资买入额、融券卖出量、两融数据时调用。

    Args:
        symbol: 6位股票代码
        date: 查询日期，空=最近有数据的交易日
    """
    symbol = _code(symbol)
    market = to_ak_market(symbol)
    d = parse_date_cn(date) or _today_str()
    d_compact = d.replace("-", "")

    # 上交所个股两融明细按日期查询；深交所按日期查询
    fn = ak.stock_margin_detail_sse if market == "sh" else ak.stock_margin_detail_szse
    ok, data = cached_ak(f"margin_{market}_{d_compact}", TTL_OTHER, fn, date=d_compact)
    if not ok:
        return data
    df = data
    code_col = "标的证券代码" if "标的证券代码" in df.columns else "证券代码"
    if code_col not in df.columns:
        return "两融接口返回格式异常，请稍后重试"
    sub = df[df[code_col].astype(str).str.zfill(6) == symbol]
    if sub.empty:
        return f"{symbol} 在 {d} 无融资融券数据（可能非交易日或非两融标的）"
    return _json({
        "symbol": symbol,
        "date": d,
        "data_source": "融资融券接口(akshare/交易所)",
        "note": "与比赛脱敏财务数据不可比",
        "rows": frames_to_records(sub),
    })


# ================================================================ 工具 8：公司基本信息

@tool
def get_stock_basic_info(symbol: str) -> str:
    """查询上市公司基本信息（上市时间、所属行业、总市值、流通市值、总股本、流通股本）。
    当用户询问某家公司的上市日期、属于哪个行业/板块、总股本、市值时调用。

    Args:
        symbol: 6位股票代码
    """
    symbol = _code(symbol)
    ok, data = cached_ak(f"basic_{symbol}", TTL_OTHER, ak.stock_individual_info_em, symbol=symbol)
    if not ok:
        return data
    df = data
    kv = {str(r.get("item")): r.get("value") for _, r in df.iterrows()}
    return _json({
        "symbol": symbol,
        "data_source": "公司基本信息接口(akshare/东方财富)",
        "note": "首发价格未收录，如需请询问上市时间/行业/股本/市值",
        "info": kv,
    })


# ================================================================ 工具 9：涨停池

@tool
def get_zt_pool(date: str = "") -> str:
    """查询涨停股票池（涨停股列表、连板数、封板资金、所属行业、涨停统计）。
    当用户询问今天/某日有哪些股票涨停、涨停板、连板股时调用。

    Args:
        date: 查询日期，空=最近交易日
    """
    d = parse_date_cn(date) or _today_str()
    d_compact = d.replace("-", "")
    ok, data = cached_ak(f"zt_{d_compact}", TTL_OTHER, ak.stock_zt_pool_em, date=d_compact)
    if not ok:
        return data
    df = data
    records = frames_to_records(df)
    if not records:
        return f"{d} 涨停池暂无数据（可能非交易日）"
    if "连板数" in df.columns:
        records = sorted(records, key=lambda r: (r.get("连板数") or 0), reverse=True)
    keep = ["代码", "名称", "涨跌幅", "最新价", "封板资金", "首次封板时间", "最后封板时间",
            "炸板次数", "涨停统计", "连板数", "所属行业"]
    keep = [c for c in keep if c in df.columns]
    return _json({
        "date": d,
        "data_source": "涨停池接口(akshare/东方财富)",
        "count": len(records),
        "rows": [{c: r.get(c) for c in keep} for r in records[:30]],
    })


# ================================================================ 工具 10：板块行情

@tool
def get_board_rank() -> str:
    """查询行业板块行情排名（板块涨跌幅、上涨/下跌家数、领涨股）。
    当用户询问某板块今天涨跌幅、板块涨幅排名、哪些板块涨得好时调用。

    Args: 无
    """
    ok, data = cached_ak("board_industry", TTL_SPOT, ak.stock_board_industry_name_em)
    if not ok:
        return data
    df = data
    records = frames_to_records(df)
    if not records:
        return "行业板块行情暂无数据"
    if "涨跌幅" in df.columns:
        records = sorted(records, key=lambda r: (r.get("涨跌幅") or 0), reverse=True)
    keep = ["板块名称", "涨跌幅", "总市值", "上涨家数", "下跌家数", "领涨股票", "领涨股票-涨跌幅"]
    keep = [c for c in keep if c in df.columns]
    return _json({
        "data_source": "行业板块行情接口(akshare/东方财富)",
        "count": len(records),
        "rows": [{c: r.get(c) for c in keep} for r in records],
    })


if __name__ == "__main__":
    # 离线冒烟：无网络时应返回友好错误而非崩溃
    print("=== 行情快照（离线应报数据源不可用） ===")
    print(get_market_snapshot.invoke({"symbols": "600519"}))
