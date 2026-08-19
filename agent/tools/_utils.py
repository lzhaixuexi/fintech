"""_utils.py —— 工具层共享工具函数
包含：代码格式转换、自然语言日期解析、磁盘缓存、akshare 调用节流/兜底。
"""
from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = ROOT / "data" / "market_cache"

# ---------------------------------------------------------------- 代码转换

def to_wind_code(symbol: str) -> str:
    """6位代码转 wind 格式: 002242 -> 002242.SZ；3/0 开头深市、6 开头沪市、其余北交所"""
    symbol = str(symbol).strip().zfill(6)
    if symbol.startswith(("0", "3")):
        return f"{symbol}.SZ"
    if symbol.startswith("6"):
        return f"{symbol}.SH"
    return f"{symbol}.BJ"


def to_ak_market(symbol: str) -> str:
    """6位代码 -> akshare 市场前缀: 6xx->sh, 0/3xx->sz, 8/4xx->bj（自动归一化）"""
    symbol = normalize_symbol(symbol)
    if symbol.startswith("6"):
        return "sh"
    if symbol.startswith(("0", "3")):
        return "sz"
    return "bj"


def to_ak_symbol(symbol: str) -> str:
    """6位代码 -> akshare 完整代码（如 600519 -> sh600519；自动归一化）"""
    symbol = normalize_symbol(symbol)
    return f"{to_ak_market(symbol)}{symbol}"


def normalize_symbol(symbol: str) -> str:
    """容忍 '002742.SZ' / 'sz002742' / '002742' 等写法，统一返回 6 位代码"""
    s = re.sub(r"[^0-9]", "", str(symbol))
    return s.zfill(6) if s else ""


# ---------------------------------------------------------------- 日期解析

_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
           "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12}

def _cn_num_to_int(s: str) -> int:
    s = s.strip()
    if s in _CN_NUM:
        return _CN_NUM[s]
    if "十" in s:  # 二十 / 十三 / 二十一
        a, _, b = s.partition("十")
        tens = _CN_NUM.get(a, 1)
        ones = _CN_NUM.get(b, 0)
        return tens * 10 + ones
    return 0


def parse_date_cn(text: str) -> str | None:
    """把常见中文日期写法解析成 'YYYY-MM-DD'。
    支持: '2025年3月18日' '3月24日' '8.1号' '20250318' '2025-03-18' '今天' '昨天'
    返回 None 表示解析失败（调用方自行决定默认值）。
    """
    if not text:
        return None
    t = str(text).strip()

    # 今天 / 昨天 / 前天 / 明天
    if t in ("今天", "今日", "当日"):
        return date.today().isoformat()
    if t in ("昨天", "昨日"):
        return (date.today() - timedelta(days=1)).isoformat()
    if t in ("前天",):
        return (date.today() - timedelta(days=2)).isoformat()

    # 2025年3月18日 / 3月24日 / 2025年3月24号
    m = re.search(r"(?P<y>\d{4})年(?P<mo>\d{1,2})月(?P<d>\d{1,2})[日号]", t)
    if m:
        return f"{m.group('y')}-{int(m.group('mo')):02d}-{int(m.group('d')):02d}"
    m = re.search(r"(?P<mo>\d{1,2})月(?P<d>\d{1,2})[日号]", t)
    if m:
        y = date.today().year
        return f"{y}-{int(m.group('mo')):02d}-{int(m.group('d')):02d}"

    # 8.1号 / 3.24 / 8月1日(无年)
    m = re.search(r"(?P<mo>\d{1,2})[.月](?P<d>\d{1,2})[号日]?", t)
    if m:
        y = date.today().year
        return f"{y}-{int(m.group('mo')):02d}-{int(m.group('d')):02d}"

    # 中文数字：三月二十四日 / 三月24日
    m = re.search(r"(?P<mo>[\d一二三四五六七八九十十一]{1,3})月(?P<d>[\d一二三四五六七八九十]{1,3})[日号]", t)
    if m:
        mo = int(m.group("mo")) if m.group("mo").isdigit() else _cn_num_to_int(m.group("mo"))
        d = int(m.group("d")) if m.group("d").isdigit() else _cn_num_to_int(m.group("d"))
        if mo and d:
            return f"{date.today().year}-{mo:02d}-{d:02d}"

    # 20250318 / 2025-03-18 / 2025/03/18
    m = re.search(r"(?P<y>\d{4})[-/]?(?P<mo>\d{1,2})[-/]?(?P<d>\d{1,2})", t)
    if m and len(t.replace("-", "").replace("/", "")) in (6, 8):
        return f"{m.group('y')}-{int(m.group('mo')):02d}-{int(m.group('d')):02d}"
    return None


def nearest_past_date(d: date, max_back: int = 10) -> date:
    """向前找最近一个工作日（跳过周末；节假日无法精确判断，由数据层兜底）"""
    cur = d
    for _ in range(max_back):
        if cur.weekday() < 5:
            return cur
        cur -= timedelta(days=1)
    return d


# ---------------------------------------------------------------- 磁盘缓存

def disk_cache(key: str, ttl: int, producer: Callable[[], Any]) -> Any:
    """带 TTL 的 JSON 磁盘缓存：data/market_cache/{key}.json
    命中返回缓存；未命中/过期/损坏则调用 producer 重建。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    try:
        if path.exists() and (time.time() - path.stat().st_mtime) < ttl:
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    try:
        val = producer()
    except Exception:
        raise
    try:
        path.write_text(json.dumps(val, ensure_ascii=False, default=str), encoding="utf-8")
    except Exception:
        pass
    return val


def frames_to_records(df) -> list[dict]:
    """DataFrame -> records 列表（空表返回 []）"""
    if df is None or df.empty:
        return []
    df = df.where(df.notna(), None)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------- akshare 兜底

def ak_call(fn: Callable, *args, min_interval: float = 0.4, **kwargs):
    """统一调用 akshare 接口：节流 + 异常转友好信息。
    返回 (ok: bool, data) —— ok=False 时 data 为错误说明字符串。
    """
    time.sleep(min_interval)
    try:
        df = fn(*args, **kwargs)
    except Exception as e:
        return False, f"行情数据源调用失败: {type(e).__name__}: {e}"
    if df is None:
        return False, "行情数据源返回为空"
    if hasattr(df, "empty") and df.empty:
        return False, "该日/该股票暂无可用的行情数据"
    return True, df


def cached_ak(key: str, ttl: int, fn: Callable, *args, min_interval: float = 0.4, **kwargs):
    """akshare 调用 + 磁盘缓存（自动 DataFrame <-> records 序列化，失败不缓存）。
    返回 (ok, data)：ok=True 时 data 为 pandas.DataFrame；ok=False 时 data 为错误说明字符串。
    """
    import pandas as pd
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    # 命中缓存
    try:
        if path.exists() and (time.time() - path.stat().st_mtime) < ttl:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(cached, list) and len(cached) == 2 and cached[0] is True:
                return True, pd.DataFrame(cached[1])
    except Exception:
        pass
    # 未命中：真实调用（成功才缓存）
    ok, data = ak_call(fn, *args, min_interval=min_interval, **kwargs)
    if not ok:
        return False, data
    try:
        path.write_text(json.dumps([True, frames_to_records(data)],
                                   ensure_ascii=False, default=str), encoding="utf-8")
    except Exception:
        pass
    return True, data


def pick_nearest_date(df, date_col: str, target: str) -> tuple[bool, list[dict]]:
    """在 df 中找 target 日期；找不到则取 target 之前最近的日期。
    返回 (ok, records)，ok=False 时 records 为错误说明。
    """
    import pandas as pd
    records = frames_to_records(df)
    if not records:
        return False, "该日期暂无可用的行情数据"
    try:
        dates = sorted({str(r[date_col])[:10] for r in records})
    except Exception:
        return True, records[-1:]
    if target in dates:
        return True, [r for r in records if str(r[date_col])[:10] == target]
    prev = [d for d in dates if d <= target]
    if prev:
        got = prev[-1]
        return True, [r for r in records if str(r[date_col])[:10] == got] + \
               [{"_note": f"目标日期 {target} 无数据（非交易日或未披露），已返回最近交易日 {got}"}]
    return True, records[-1:] + [{"_note": f"目标日期 {target} 早于数据范围，已返回最早记录"}]
