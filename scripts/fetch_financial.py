from pathlib import Path

import akshare as ak


DATA_DIR = Path(__file__).parent.parent / "data" / "financial"

def fetch_balance(stock:str,symbol:str):
    df = ak.stock_financial_report_sina(stock=stock,symbol="资产负债表")
    path = DATA_DIR / f"{symbol}_balance.csv"
    df.to_csv(path, index=False,encoding="utf-8-sig")
    print(f"已保存{path},{df.shape[0]}行 * {df.shape[1]}列")

def fetch_abstract(symbol:str):
    df = ak.stock_financial_abstract(symbol=symbol)
    path = DATA_DIR / f"{symbol}_abstract.csv"
    df.to_csv(path, index=False,encoding="utf-8-sig")
    print(f"已保存 {path}，{df.shape[0]} 行 × {df.shape[1]} 列")


if __name__ == '__main__':
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    companies = [("sz002242","002242"),("sz300308","300308")]
    for stock,symbol in companies:
        fetch_abstract(symbol)
        fetch_balance(stock,symbol)