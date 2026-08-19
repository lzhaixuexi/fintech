"""fetch_stock_list.py —— 生成 data/market/stock_list.csv（全A股 代码→名称 映射）

数据源优先级：
  ① 交易所官方源（上交所/深交所/北交所）——权威、信息全
  ② 新浪全市场列表（ak.stock_info_a_code_name）——补齐交易所遗漏（ST/退市整理等）
  ③ 比赛研报数据 race/5（离线兜底，无网也能生成）

用途：
  - build_graph.py：全市场代码名称映射（补 ST/重整股名字 + 扩大穿透面）
  - app.py / 各工具的股票名称解析
"""
from __future__ import annotations

from pathlib import Path

import akshare as ak
import pandas as pd

OUT = Path(__file__).parent.parent / "data" / "market" / "stock_list.csv"
R5 = Path(__file__).parent.parent / "data" / "race" / "5" / "rr_main_202605281537.csv"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名 + 代码 zfill(6) + 名称去空格（含全角）+ 去重"""
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["name"] = df["name"].astype(str).str.replace(" ", "", regex=False) \
                                    .str.replace("\u3000", "", regex=False)
    return df.drop_duplicates("code").reset_index(drop=True)


def fetch_exchange() -> pd.DataFrame | None:
    """① 交易所官方源：上交所 + 深交所 + 北交所"""
    frames = []
    for name, fn, cols in [
        ("上交所", lambda: ak.stock_info_sh_name_code(), ("证券代码", "证券简称")),
        ("深交所", lambda: ak.stock_info_sz_name_code(), ("A股代码", "A股简称")),
        ("北交所", lambda: ak.stock_info_bj_name_code(), ("证券代码", "证券简称")),
    ]:
        try:
            df = fn()
            sub = df[list(cols)].rename(columns={cols[0]: "code", cols[1]: "name"})
            frames.append(sub)
            print(f"  [交易所-{name}] {len(sub)} 只")
        except Exception as e:
            print(f"  [交易所-{name}] 失败: {type(e).__name__}: {str(e)[:100]}")
    if not frames:
        return None
    return _clean(pd.concat(frames, ignore_index=True))


def fetch_sina() -> pd.DataFrame:
    """② 新浪全市场列表"""
    df = ak.stock_info_a_code_name()
    return _clean(df[["code", "name"]])


def fetch_race5() -> pd.DataFrame:
    """③ 比赛研报数据离线兜底（sec_code -> sec_name）"""
    df = pd.read_csv(R5, usecols=["sec_code", "sec_name"])
    df = df.dropna(subset=["sec_code", "sec_name"])
    df["code"] = df["sec_code"].astype(str).str.split(".").str[0]
    df["name"] = df["sec_name"]
    return _clean(df[["code", "name"]])


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("[1/3] 交易所官方源...")
    df = fetch_exchange()
    if df is not None:
        print(f"  ✅ 交易所源: {len(df)} 只")
    else:
        print("  ⚠️ 交易所源全部失败")

    print("[2/3] 新浪源补齐（交易所优先合并）...")
    try:
        sina = fetch_sina()
        print(f"  新浪源: {len(sina)} 只")
        if df is None:
            df = sina
        else:
            merged = pd.concat([df, sina], ignore_index=True)
            df = merged.drop_duplicates("code", keep="first")  # 交易所优先
            print(f"  合并后: {len(df)} 只")
    except Exception as e:
        print(f"  新浪源失败: {type(e).__name__}: {str(e)[:100]}")

    if df is None or df.empty:
        print("[2.5/3] 网络源全部失败，用比赛研报数据离线兜底...")
        df = fetch_race5()
        print(f"  离线兜底: {len(df)} 只")

    df = _clean(df)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"[3/3] 已写入 {OUT}: {len(df)} 只")
    print(f"  文件大小: {OUT.stat().st_size / 1024:.0f} KB")

    print("\n抽查验证:")
    for c in ["002742", "002242", "600519", "000858", "601318"]:
        hit = df.loc[df["code"] == c, "name"].tolist()
        print(f"  {c} -> {hit[0] if hit else '未收录'}")


if __name__ == "__main__":
    main()
