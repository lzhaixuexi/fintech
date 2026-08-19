import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from agent.tools._utils import to_wind_code

RACE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "race" / "4"

# 报告期后缀 -> 报告类型标签
PERIOD_TAG = {"0331": "一季报", "0630": "中报", "0930": "三季报", "1231": "年报"}

# 派生指标里属于"比率"的（输出时不除以 1e8）
RATIO_INDS = {"毛利率", "净利率", "ROE", "ROA", "资产负债率", "资产周转率",
              "存货周转率", "存货/营收比", "费用率"}
# 每股类指标（单位：元，不除以 1e8）
SHARE_INDS = {"基本每股收益", "每股净资产"}

# 基础字段清单：文件 -> (指标中文名 -> CSV 列名)
BASE_FIELDS = {
    "ashareincome_202605261519.csv": {
        "营业总收入": "tot_oper_rev",
        "营业收入": "oper_rev",
        "归母净利润": "net_profit_excl_min_int_inc",
        "营业成本": "less_oper_cost",
        "销售费用": "less_selling_dist_exp",
        "管理费用": "less_gerl_admin_exp",
        "财务费用": "less_fin_exp",
        "研发费用": "rd_expense",
        "其他收益": "other_income",
        "营业外收入": "plus_non_oper_rev",
        "营业外支出": "less_non_oper_exp",
        "营业利润": "oper_profit",
        "资产减值损失": "less_impair_loss_assets",
        "信用减值损失": "credit_impairment_loss",
        "基本每股收益": "s_fa_eps_basic",
        "利息收入(财务费用中)": "fin_exp_int_inc",
    },
    "asharebalancesheet_202605261517.csv": {
        "货币资金": "monetary_cap",
        "应收账款": "acct_rcv",
        "应收票据": "notes_rcv",
        "应收票据及账款": "accounts_receivable_bill",
        "预付账款": "prepay",
        "存货": "inventories",
        "合同资产": "contractual_assets",
        "商誉": "goodwill",
        "在建工程": "const_in_prog",
        "固定资产": "fix_assets",
        "总资产": "tot_assets",
        "总负债": "tot_liab",
        "归母股东权益": "tot_shrhldr_eqy_excl_min_int",
        "短期借款": "st_borrow",
        "长期借款": "lt_borrow",
        "应付债券": "bonds_payable",
        "一年内到期非流动负债": "non_cur_liab_due_within_1y",
        "总股本": "tot_shr",
    },
    "asharecashflow_202605261518.csv": {
        "经营现金流量净额": "net_cash_flows_oper_act",
        "投资现金流量净额": "net_cash_flows_inv_act",
        "筹资现金流量净额": "net_cash_flows_fnc_act",
        "自由现金流": "free_cash_flow",
    },
}


def _period_label(rp) -> str:
    """20250331 -> '2025-03-31(一季报)'"""
    rp = str(rp)
    tag = PERIOD_TAG.get(rp[4:8], "")
    return f"{rp[:4]}-{rp[4:6]}-{rp[6:8]}({tag})"


def _read_series(fname: str, col: str, wind_code: str,
                 periods: str = "annual", tail: int = 5) -> pd.Series:
    """读某文件的某列，过滤出目标公司，返回 报告期 -> 数值 的 Series（原始单位）。
    periods: 'annual' 只留年报(向后兼容)；'all' 保留全部报告期，取最近 tail 期。
    """
    df = pd.read_csv(RACE_DIR / fname, usecols=["s_info_windcode", "report_period", col])
    df = df[df["s_info_windcode"] == wind_code]
    if periods == "annual":
        df = df[df["report_period"].astype(str).str.endswith("1231")]
    df = df.sort_values("report_period")
    s = df.set_index("report_period")[col].apply(pd.to_numeric, errors="coerce")
    s = s.dropna()
    return s.tail(tail)


def load_race_financial(symbol: str, periods: str = "annual", tail: int = 5) -> pd.DataFrame:
    """返回 指标×报告期 的 DataFrame（原始单位：元 / 股）。
    periods='annual'：列名为年份字符串（如 '2024'），保持旧接口兼容；
    periods='all'：列名为完整报告期标签（如 '2025-09-30(三季报)'）。
    """
    wind_code = to_wind_code(symbol)
    rows = {}
    for fname, fields in BASE_FIELDS.items():
        for zh, col in fields.items():
            rows[zh] = _read_series(fname, col, wind_code, periods=periods, tail=tail)
    df = pd.DataFrame(rows).T  # 指标 × 报告期

    if periods == "all":
        # 列名换成可读标签，并保持时间顺序
        df.columns = [_period_label(c) for c in df.columns]
        df = _add_derived_all(df)
    else:
        df.columns = [str(y)[:4] for y in df.columns]  # 20241231 -> 2024
        df = _add_derived_annual(df)
    return df


def _add_derived_annual(df: pd.DataFrame) -> pd.DataFrame:
    """年报模式下补充派生指标行（单位：比率/元；货币类仍为元，输出时统一转亿）"""
    rev, cost = "营业总收入", "营业成本"
    if rev in df.index and cost in df.index:
        df.loc["毛利率"] = (df.loc[rev] - df.loc[cost]) / df.loc[rev].replace(0, pd.NA)
    if rev in df.index and "归母净利润" in df.index:
        df.loc["净利率"] = df.loc["归母净利润"] / df.loc[rev].replace(0, pd.NA)
    if "归母净利润" in df.index and "归母股东权益" in df.index:
        df.loc["ROE"] = df.loc["归母净利润"] / df.loc["归母股东权益"].replace(0, pd.NA)
    if "归母净利润" in df.index and "总资产" in df.index:
        df.loc["ROA"] = df.loc["归母净利润"] / df.loc["总资产"].replace(0, pd.NA)
    if "总负债" in df.index and "总资产" in df.index:
        df.loc["资产负债率"] = df.loc["总负债"] / df.loc["总资产"].replace(0, pd.NA)
    if rev in df.index and "总资产" in df.index:
        df.loc["资产周转率"] = df.loc[rev] / df.loc["总资产"].replace(0, pd.NA)
    if rev in df.index and "存货" in df.index:
        df.loc["存货/营收比"] = df.loc["存货"] / df.loc[rev].replace(0, pd.NA)
    if cost in df.index and "存货" in df.index:
        df.loc["存货周转率"] = df.loc[cost] / df.loc["存货"].replace(0, pd.NA)
    if "归母股东权益" in df.index and "总股本" in df.index:
        df.loc["每股净资产"] = df.loc["归母股东权益"] / df.loc["总股本"].replace(0, pd.NA)
    # 有息负债 = 短借 + 长借 + 应付债券 + 一年内到期非流动负债（缺项按 0 计）
    debt_parts = ["短期借款", "长期借款", "应付债券", "一年内到期非流动负债"]
    if any(d in df.index for d in debt_parts):
        df.loc["有息负债"] = sum(df.loc[d].fillna(0) for d in debt_parts if d in df.index)
    # 期间费用率 = (销售+管理+财务) / 营业总收入
    if all(x in df.index for x in ["销售费用", "管理费用", "财务费用"]) and rev in df.index:
        df.loc["费用率"] = (df.loc["销售费用"] + df.loc["管理费用"] + df.loc["财务费用"]) / df.loc[rev].replace(0, pd.NA)
    return df


def _add_derived_all(df: pd.DataFrame) -> pd.DataFrame:
    """全报告期模式下补充派生指标（含单季度差分）"""
    df = _add_derived_annual(df)
    rev, profit = "营业总收入", "归母净利润"
    if rev in df.index:
        df.loc["单季度营业收入"] = df.loc[rev].diff()
    if profit in df.index:
        df.loc["单季度归母净利润"] = df.loc[profit].diff()
    return df


def _fmt_row(row: pd.Series, unit_map: dict) -> dict:
    """把一行指标格式化成 {报告期: 数值}，货币类转亿元保留 2 位"""
    out = {}
    for period, v in row.items():
        if pd.isna(v):
            continue
        v = float(v)
        kind = unit_map.get(row.name, "亿")
        if kind == "亿":
            out[str(period)] = round(v / 1e8, 2)
        elif kind == "比率":
            out[str(period)] = round(v, 4)
        else:  # 元
            out[str(period)] = round(v, 4)
    return out


def build_unit_map(df: pd.DataFrame) -> dict:
    return {ind: ("比率" if ind in RATIO_INDS else "元" if ind in SHARE_INDS else "亿")
            for ind in df.index}


def get_financials_json(symbol: str, period: str = "annual", tail: int = 5) -> str:
    """核心查询逻辑（供 @tool 包装）。
    period: 'annual' 近5年年报 | 'latest' 最近一期 | 'all' 近 tail 期全部报告期
    """
    try:
        if period == "annual":
            df = load_race_financial(symbol, periods="annual", tail=tail)
        else:
            df = load_race_financial(symbol, periods="all", tail=tail)
            if period == "latest":
                df = df.iloc[:, [-1]]
    except FileNotFoundError:
        return f"未找到股票 {symbol} 的财务数据文件"

    if df.empty or df.shape[1] == 0:
        return json.dumps({"symbol": symbol, "note": "该股票暂无可用的财务数据"}, ensure_ascii=False)

    unit_map = build_unit_map(df)
    result = {ind: _fmt_row(df.loc[ind], unit_map) for ind in df.index}
    result = {k: v for k, v in result.items() if v}  # 去掉完全无数据的指标
    return json.dumps({
        "symbol": symbol,
        "data_source": "race/4 三大财务报表（母公司报表，比赛脱敏数据；绝对值为母公司口径，与真实行情/研报不可比）",
        "period_mode": period,
        "indicators": result,
    }, ensure_ascii=False)


@tool
def get_financials(symbol: str, period: str = "annual") -> str:
    """查询上市公司财务数据（近5年核心指标，支持季报）。
    指标包括：营业总收入、营业收入、归母净利润、经营现金流量净额、货币资金、存货、
    应收账款、预付账款、商誉、有息负债、总资产、总负债、股东权益、每股收益、每股净资产，
    以及派生指标：毛利率、净利率、ROE、ROA、资产负债率、资产周转率、存货周转率、费用率等。
    当用户询问某家公司的财务数据、营收、利润、现金流、毛利率、ROE、负债率、存货、
    应收账款、单季度业绩、季报/中报/年报时调用。

    Args:
        symbol: 6位股票代码，如"002242"
        period: 报告期范围，'annual'=近5年年报(默认)，'latest'=最近一期报告，
                'all'=近5期全部报告期（含季报/中报，附单季度值）
    """
    return get_financials_json(symbol, period=period)


@tool
def detect_financial_fraud(symbol: str) -> str:
    """检测上市公司近5年财报是否存在造假风险，基于财务勾稽规则分析
    存货增速、现金流质量、应收账款异常、存贷双高、商誉、预付账款、费用率等指标。
    当用户询问某家公司有没有财务造假、财报风险、暴雷风险、扫雷时调用。

    Args:
        symbol: 6位股票代码，如"002242"
    """
    try:
        df = load_race_financial(symbol)  # 年报口径，原始单位（元/股）
    except FileNotFoundError:
        return f"未找到股票 {symbol} 的财务数据文件"

    warnings = []

    def add(rule, severity, year, warning_point, data_comparison, fraud_pattern):
        warnings.append({
            "rule": rule, "severity": severity, "year": str(year),
            "warning_point": warning_point,
            "data_comparison": data_comparison,
            "fraud_pattern": fraud_pattern,
        })

    # === 规则1：存货激增（存货增速 >> 营收增速） ===
    if "存货" in df.index and "营业总收入" in df.index:
        inv_growth = df.loc["存货"].pct_change().dropna()
        rev_growth = df.loc["营业总收入"].pct_change().dropna()
        for year in inv_growth.index:
            if year in rev_growth.index:
                ig, rg = inv_growth[year], rev_growth[year]
                if ig > rg * 2 and ig > 0.3 and rg > -0.2:
                    add("存货激增", "high", year, "存货增速远超营收增速，库存与销售脱节",
                        f"存货同比{ig:+.0%} vs 营收同比{rg:+.0%}",
                        "可能虚增存货虚增利润，或滞销面临跌价风险")

    # === 规则2：利润质量差（经营现金流 < 净利润的50%） ===
    if "经营现金流量净额" in df.index and "归母净利润" in df.index:
        ocf = df.loc["经营现金流量净额"]
        profit = df.loc["归母净利润"]
        for year in ocf.index:
            if year in profit.index and profit[year] > 0:
                ratio = ocf[year] / profit[year]
                if ratio < -0.3 and profit[year] > 2e8:
                    severity = "high"
                elif 0.5e8 < profit[year] <= 2e8 and ratio < -0.5:
                    severity = "medium"
                else:
                    continue
                add("利润质量差", severity, year, "经营现金流为负而净利润为正，利润无现金支撑",
                    f"经营现金流{ocf[year]/1e8:.2f}亿 vs 净利润{profit[year]/1e8:.2f}亿，比率{ratio:.0%}",
                    "可能是纸面利润（虚增收入/应收挂账），或回款周期拉长的业务风险")

    # === 规则3：应收账款异常增长（增速 > 营收增速的1.5倍且超过20%） ===
    if "应收账款" in df.index and "营业总收入" in df.index:
        ar_growth = df.loc["应收账款"].pct_change().dropna()
        rev_growth = df.loc["营业总收入"].pct_change().dropna()
        for year in ar_growth.index:
            if year in rev_growth.index:
                ag, rg = ar_growth[year], rev_growth[year]
                if ag > rg * 1.5 and ag > 0.2:
                    add("应收账款异常增长", "medium", year, "应收账款增速远超营收，收入确认激进",
                        f"应收同比{ag:+.0%} vs 营收同比{rg:+.0%}",
                        "可能放宽信用政策提前确认收入，后续有坏账暴雷风险")

    # === 规则4：存货/营收比异动 ===
    if "存货" in df.index and "营业总收入" in df.index:
        inv_rev = df.loc["存货"] / df.loc["营业总收入"]
        inv_rev_growth = inv_rev.pct_change().dropna()
        for year in inv_rev_growth.index:
            if inv_rev[year] > 0.3 and inv_rev_growth[year] > 0.5:
                add("存货/营收比异动", "medium", year, "存货占营收比重异常跳升",
                    f"存货/营收比{inv_rev[year]:.2f}，同比{inv_rev_growth[year]:+.0%}",
                    "备货与销售脱节，可能虚增存货或面临跌价")

    # === 规则5：异常财务费用 ===
    if "财务费用" in df.index and "总资产" in df.index:
        fe_ratio = df.loc["财务费用"] / df.loc["总资产"]
        fe_growth = fe_ratio.pct_change().dropna()
        for year in fe_growth.index:
            if df.loc["财务费用", year] > 0 and fe_ratio[year] > 0.05 and fe_growth[year] > 0.5:
                add("异常财务费用", "medium", year, "财务费用占资产比重异常抬升",
                    f"财务费用/总资产{fe_ratio[year]:.1%}，同比{fe_growth[year]:+.0%}",
                    "融资成本急剧上升，可能资金链紧张或依赖高息借款")

    # === 规则6：利润结构异常（非经常性收益占比过高） ===
    if "归母净利润" in df.index and "其他收益" in df.index:
        profit = df.loc["归母净利润"]
        non_oper = (df.loc["营业外收入"].fillna(0) - df.loc["营业外支出"].fillna(0)
                    + df.loc["其他收益"].fillna(0))
        ratio = non_oper / profit
        for year in ratio.index:
            if profit[year] > 0 and ratio[year] > 0.5:
                add("利润结构异常", "medium", year, "非经常性收益占比过高，利润结构失衡",
                    f"非经常性收益{non_oper[year]/1e8:.2f}亿，占净利润{ratio[year]:.0%}",
                    "利润依赖一次性收益，主业盈利薄弱，利润含金量低")

    # === 规则7：存贷双高（货币资金与有息负债同时高企，且利息收入偏低） ===
    if all(x in df.index for x in ["货币资金", "有息负债", "总资产"]) and "利息收入(财务费用中)" in df.index:
        cash_ratio = df.loc["货币资金"] / df.loc["总资产"].replace(0, pd.NA)
        debt_ratio = df.loc["有息负债"] / df.loc["总资产"].replace(0, pd.NA)
        int_inc = df.loc["利息收入(财务费用中)"]
        for year in cash_ratio.index:
            if year not in debt_ratio.index:
                continue
            cash = df.loc["货币资金", year]
            if not (cash > 0 and cash_ratio[year] > 0.25 and debt_ratio[year] > 0.25):
                continue
            int_yield = float(int_inc.get(year, 0) or 0) / cash
            if int_yield >= 0.02:
                continue
            add("存贷双高", "high", year, "账面货币资金与有息负债同时高企，资金真实性与用途存疑",
                f"货币资金占总资产{cash_ratio[year]:.0%}，有息负债占总资产{debt_ratio[year]:.0%}，利息收入率仅{int_yield:.2%}",
                "典型资金占用/体外循环特征，资金可能被大股东占用或虚构")

    # === 规则8：商誉减值风险（商誉占净资产高 + 利润下滑） ===
    if all(x in df.index for x in ["商誉", "归母股东权益"]) and "归母净利润" in df.index:
        gw_ratio = df.loc["商誉"] / df.loc["归母股东权益"].replace(0, pd.NA)
        profit_chg = df.loc["归母净利润"].pct_change()
        for year in gw_ratio.index:
            if gw_ratio[year] > 0.3 and year in profit_chg.index and profit_chg[year] < -0.1:
                add("商誉减值风险", "medium", year, "商誉占净资产比重高且利润下滑，减值压力大",
                    f"商誉/净资产{gw_ratio[year]:.0%}，归母净利润同比{profit_chg[year]:+.0%}",
                    "并购标的业绩不及预期时商誉减值将直接吞噬利润")

    # === 规则9：预付账款异常增长 ===
    if "预付账款" in df.index and "营业总收入" in df.index:
        prepay_growth = df.loc["预付账款"].pct_change().dropna()
        rev_growth = df.loc["营业总收入"].pct_change().dropna()
        for year in prepay_growth.index:
            if year in rev_growth.index:
                pg, rg = prepay_growth[year], rev_growth[year]
                if pg > rg * 1.5 and pg > 0.3:
                    add("预付账款异常增长", "medium", year, "预付账款增速远超营收，资金被占用",
                        f"预付同比{pg:+.0%} vs 营收同比{rg:+.0%}",
                        "可能通过预付形式转移资金、虚增采购成本")

    # === 规则10：期间费用率异常抬升 ===
    if "费用率" in df.index:
        fe_ratio = df.loc["费用率"]
        fe_growth = fe_ratio.pct_change().dropna()
        for year in fe_growth.index:
            if fe_ratio[year] > 0.15 and fe_growth[year] > 0.2:
                add("费用率异常抬升", "medium", year, "期间费用率同比明显跳升，利润被侵蚀",
                    f"期间费用率{fe_ratio[year]:.1%}，同比{fe_growth[year]:+.0%}",
                    "收入增长乏力或费用失控，盈利能力恶化")

    # === 输出层：多维风险评分 + 诊断逻辑链（维度权重保持原五维不变） ===
    DIM_WEIGHTS = {"利润质量": 0.25, "存货风险": 0.20, "应收风险": 0.20,
                   "利润结构": 0.20, "融资风险": 0.15}
    DIM_RULES = {
        "利润质量": ["利润质量差", "费用率异常抬升"],
        "存货风险": ["存货激增", "存货/营收比异动"],
        "应收风险": ["应收账款异常增长", "预付账款异常增长"],
        "利润结构": ["利润结构异常", "商誉减值风险"],
        "融资风险": ["异常财务费用", "存贷双高"],
    }
    SEV_SCORE = {"high": 70, "medium": 40}

    dim_scores = {}
    for dim, rules in DIM_RULES.items():
        best = 0
        for w in warnings:
            if w["rule"] in rules:
                best = max(best, SEV_SCORE.get(w["severity"], 0))
        dim_scores[dim] = best

    weighted = sum(dim_scores[d] * DIM_WEIGHTS[d] for d in DIM_WEIGHTS)
    max_sev = max((SEV_SCORE.get(w["severity"], 0) for w in warnings), default=0)
    score = max(weighted, max_sev)  # 保底：单项 high 不会被权重稀释

    risk_level = "high" if score >= 60 else "medium" if score >= 40 else "low"
    return json.dumps({
        "company": symbol,
        "risk_level": risk_level,
        "risk_score": round(score, 1),
        "dimensions": dim_scores,
        "warnings": warnings,
        "conclusion": "未触发预警" if not warnings else f"触发{len(warnings)}项预警"
    }, ensure_ascii=False)


if __name__ == "__main__":
    print("===== 年报模式 002742 =====")
    print(get_financials_json("002742", period="annual"))
    print()
    print("===== 全报告期模式 002742（含季报+单季度） =====")
    print(get_financials_json("002742", period="all", tail=6))
    print()
    print("===== 最新一期 002742 =====")
    print(get_financials_json("002742", period="latest"))
    print()
    print("===== 反欺诈 002742 =====")
    print(detect_financial_fraud.invoke({"symbol": "002742"}))
    print()
    print("===== 反欺诈 300355 =====")
    print(detect_financial_fraud.invoke({"symbol": "300355"}))
