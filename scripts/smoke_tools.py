"""smoke_tools.py —— 工具层冒烟测试
用法: uv run python scripts/smoke_tools.py
验证: 所有 @tool 可被 invoke、不崩溃；有网络时返回真实数据，无网络时优雅降级。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    from agent.tools.market_data import (get_market_snapshot, get_stock_history, get_fund_flow,
        get_fund_flow_rank, get_lhb, get_block_trade, get_margin_balance, get_stock_basic_info,
        get_zt_pool, get_board_rank)
    from agent.tools.financial_data import get_financials, detect_financial_fraud
    from agent.tools.get_event_clusters import get_event_clusters

    cases = [
        ("get_financials(年报)", get_financials, {"symbol": "600519"}),
        ("get_financials(季报)", get_financials, {"symbol": "600519", "period": "all"}),
        ("detect_financial_fraud", detect_financial_fraud, {"symbol": "002742"}),
        ("get_event_clusters", get_event_clusters, {"symbol": "002742"}),
        ("get_market_snapshot", get_market_snapshot, {"symbols": "600519,000858"}),
        ("get_stock_history", get_stock_history, {"symbol": "600519", "start_date": "近一月"}),
        ("get_fund_flow", get_fund_flow, {"symbol": "002742", "date": "2025-03-24"}),
        ("get_fund_flow_rank", get_fund_flow_rank, {}),
        ("get_lhb", get_lhb, {}),
        ("get_block_trade", get_block_trade, {}),
        ("get_margin_balance", get_margin_balance, {"symbol": "600519"}),
        ("get_stock_basic_info", get_stock_basic_info, {"symbol": "600519"}),
        ("get_zt_pool", get_zt_pool, {}),
        ("get_board_rank", get_board_rank, {}),
    ]
    real, degraded, crashed = 0, 0, 0
    for name, fn, kw in cases:
        try:
            r = fn.invoke(kw)
            try:
                j = json.loads(r)
                n = j.get("count", len(j.get("rows", [])) if isinstance(j.get("rows"), list) else "?")
                print(f"[OK  ] {name}: 真实数据 {n} 条")
                real += 1
            except Exception:
                print(f"[降级] {name}: {str(r)[:80]}")
                degraded += 1
        except Exception as e:
            print(f"[崩溃] {name}: {type(e).__name__}: {e}")
            crashed += 1
    print(f"\n=== 结果: 真实数据 {real} / 降级 {degraded} / 崩溃 {crashed} ===")
    return 1 if crashed else 0


if __name__ == "__main__":
    sys.exit(main())
