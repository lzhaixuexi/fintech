import pandas as pd

df = pd.read_excel(
    r"D:\programbiancheng\python\fintech\data\race\3\clean.xlsx",
    sheet_name="Sheet2",
    usecols=["s_info_windcode", "ann_dt", "n_info_title", "n_info_fcode"],
)
df["code"] = df["s_info_windcode"].map(lambda x: str(x).split(".")[0])
sub = df[df["code"] == "002742"].sort_values("ann_dt")
print("总数:", len(sub))
for _, r in sub.iterrows():
    print(r["ann_dt"].strftime("%Y-%m-%d"), "|", str(r["n_info_title"])[:70], "|", str(r["n_info_fcode"])[:30])
