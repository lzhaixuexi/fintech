# 金融智能 Agent —— 面向财报反欺诈、股权穿透与舆情事件分析

> 第五届中国研究生金融科技创新大赛 · 东吴证券赛题参赛项目
>
> 一个基于 **LangGraph + DeepSeek** 的 Agentic AI 金融分析系统：用户用自然语言提问，Agent 自动选择合适的工具，完成财务异象甄别、股权穿透、舆情事件归类、行情/资金流查询等专业分析。
>
> 本 README 面向**零基础部署**：按顺序做完"环境准备 → 数据准备 → 运行"三节，你就能跑起来。

---

## 一、这个系统能干什么

| 能力 | 一句话说明 | 需要的数据/服务 |
|---|---|---|
| **智能问答** | 自然语言提问，Agent 自动选工具、给带证据链的回答 | DeepSeek API |
| **财务反欺诈检测** | 输入股票代码，跑 10 条排雷规则（存货激增、现金流倒挂、存贷双高等），输出五维风险评分 + 每条预警的"预警点/数据对比/造假模式"三件套 | `data/race/4` 财报 CSV |
| **财务数据查询** | 近 5 年财务指标（营收/净利/毛利率/ROE/负债率/单季度业绩等 30+ 项），支持季报/中报/年报 | `data/race/4` 财报 CSV |
| **股权穿透** | 查询实际控制人及完整持股链路（最多 5 跳） | Neo4j + `data/race/2` 股东数据 |
| **舆情事件簇** | 公司公告自动归类（财务造假/监管处罚/股权变动等 6+1 簇），带时间轴 | `data/race/3` 公告数据 |
| **研报语义检索** | 5.5 万篇券商研报按语义搜索 | `data/chroma` 向量库（已灌好） |
| **行情/资金流** | 实时股价、历史 K 线、主力资金流向、龙虎榜、大宗交易、融资融券、涨停池、板块行情 | 联网（akshare），自动降级 |
| **可视化界面** | Streamlit 双 Tab：智能问答 + 事件时间轴（AI 生成事件脉络叙事） | 同上 |

---

## 二、项目结构（每个文件是干什么的）

> ★ 越多越重要。**不想深究的部署者可以直接跳到第三节**，这里给你当"说明书"查。

```
fintech/
├── app.py                      # ★ Streamlit 可视化界面（两个 Tab：💬智能问答 / 📅事件时间轴）
│
├── agent/                      # ★★ Agent 核心（比赛主要代码在这）
│   ├── orchestrator.py         # ★★★ 主 Agent：LangGraph 状态图
│   │                           #   · 15 个工具注册 + 自动选择（LLM 决定调哪个）
│   │                           #   · 多轮对话记忆（thread_id + 摘要压缩）
│   │                           #   · 工具报错自动兜底（把错误喂回 LLM 自纠错）
│   │                           #   · 命令行对话入口（python agent/orchestrator.py）
│   └── tools/                  # ★ 15 个工具，每个文件一类功能
│       ├── __init__.py         #   空文件，让 tools 成为 Python 包
│       ├── _utils.py           #   共享工具函数：代码格式转换、中文日期解析（"3月24日"→2025-03-24）、磁盘缓存
│       ├── financial_data.py   # ★★ 财务工具×2：
│       │                       #   · get_financials         查询财务数据（30+ 指标，支持季报）
│       │                       #   · detect_financial_fraud 反欺诈检测（10 条规则 + 五维评分）
│       ├── market_data.py      # ★ 行情工具×10：股价快照 / 历史K线 / 主力资金流向 / 资金流排行
│       │                       #   / 龙虎榜 / 大宗交易 / 融资融券 / 公司基本信息 / 涨停池 / 板块行情
│       │                       #   （全部带磁盘缓存 + 多数据源回退 + 断网降级）
│       ├── shareholder_graph.py# 股权穿透 trace_shareholder（连 Neo4j 查持股链路）
│       ├── get_event_clusters.py# 公告事件簇 get_event_clusters（公告按主题归类）
│       └── search_reports.py   # 研报语义检索 search_reports（Chroma 向量库）
│
├── scripts/                    # 数据准备/工具脚本（不是每次运行都要跑）
│   ├── fetch_stock_list.py     # 生成 data/market/stock_list.csv（全 A 股 代码→名称，多源回退）
│   ├── build_graph.py          # 灌 Neo4j 股权图谱（读 data/race/2 股东数据，跑一次，约 3~5 分钟）
│   ├── build_chroma.py         # 灌 Chroma 研报向量库（读 data/race/5 研报，跑一次，约 10~15 分钟）
│   ├── fetch_financial.py      # 备用：用 akshare 拉真实财务数据存 CSV（目前主用比赛数据，可不管）
│   └── smoke_tools.py          # ★ 冒烟测试：一键验证 15 个工具是否正常（部署后建议先跑这个）
│
├── data/                       # 所有数据
│   ├── race/                   # 比赛官方数据（5 份，见第四节"数据说明"）
│   │   ├── 1/clean.xlsx        #   多轮对话问句集（1410 条，35 个会话）
│   │   ├── 2/clean.xlsx        #   前十大股东数据（64.6 万行，Neo4j 图谱的数据源）
│   │   ├── 3/clean.xlsx        #   公司公告数据（7311 条，事件簇的数据源）
│   │   ├── 4/                  #   三大财务报表 CSV×3 + 字段说明 dict（财务工具的数据源）
│   │   └── 5/rr_main_*.csv     #   券商研报 5.5 万条（Chroma 的数据源）
│   ├── market/stock_list.csv   # 全 A 股 代码→名称（fetch_stock_list.py 生成，已内置）
│   ├── models/bge-small-zh-v1.5# 本地中文 embedding 模型（已下载，离线可用，勿删）
│   ├── chroma/                 # Chroma 研报向量库（已灌好，约 1.1GB，勿删）
│   └── market_cache/           # 行情数据磁盘缓存（工具自动生成，无需手动管理；删了会自动重建）
│
├── test/                       # 测试脚本（akshare 拉数 / Neo4j 连接 / 公告数据抽查）
├── .env                        # 你的 API Key（见第三节第 4 步，不要提交 git）
├── .gitignore                  # git 忽略规则
├── pyproject.toml              # Python 依赖清单（uv 按它装依赖）
├── uv.lock                     # 依赖版本锁定文件（uv sync 自动使用，别手动改）
└── README.md                   # 本文件
```

---

## 三、环境准备（从零到跑起来）

> 全程大约 20 分钟。任何一步卡住，看"第七节 常见问题"。

### 第 1 步：安装 Python（≥ 3.11）

- 官网下载：https://www.python.org/downloads/
- 安装时**务必勾选 "Add Python to PATH"**
- 验证：打开终端（Windows 用 PowerShell），输入 `python --version`，能显示 3.11+ 即可

### 第 2 步：安装 uv（Python 包管理器）

uv 用来一键安装所有依赖，比 pip 快很多。

```bash
# Windows PowerShell：
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux：
curl -LsSf https://astral.sh/uv/install.sh | sh
```

验证：`uv --version`

### 第 3 步：安装 Neo4j（可选，只影响"股权穿透"功能）

> 不做股权穿透可以跳过，其他功能照常能用。

1. 下载社区版：https://neo4j.com/download-center/
2. 安装后启动数据库，记下密码（项目里默认密码写的是 `12345678`）
3. 如果密码不是 `12345678`，改这两处文件的顶部：
   - `agent/tools/shareholder_graph.py` 第 15 行
   - `scripts/build_graph.py` 第 11 行

### 第 4 步：配置 API Key

在项目根目录建一个 `.env` 文件（项目里已有一份模板），内容：

```
DEEPSEEK_API_KEY=你的key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

- Key 在 https://platform.deepseek.com 注册获取（很便宜）
- `.env` 是敏感文件，已被 `.gitignore` 忽略，不会提交

### 第 5 步：安装 Python 依赖

在项目根目录（有 `pyproject.toml` 的文件夹）执行：

```bash
uv sync
```

它会自动创建虚拟环境并安装全部依赖（akshare、langgraph、pandas、streamlit 等）。
**国内网络慢的话**，`pyproject.toml` 里已配置阿里云镜像，不用额外设置。

---

## 四、数据准备

### 第 1 步：放置比赛数据

把比赛官方数据放到 `data/race/` 下，目录结构见"第二节"的 `data/race/` 说明：

```
data/race/1/clean.xlsx         ← 问句集（可选，测试长对话用）
data/race/2/clean.xlsx         ← 股东数据（Neo4j 图谱用）
data/race/3/clean.xlsx         ← 公告数据（事件簇用）
data/race/4/*.csv 和 *dict.txt ← 三大财报（财务工具用）
data/race/5/rr_main_*.csv      ← 研报（Chroma 用）
```

> 项目已内置 `data/models/`（embedding 模型）和 `data/chroma/`（研报向量库），**这两个不用你准备**。
> 如果 `data/race/4` 或 `data/race/5` 缺失，对应功能（财务/研报）会提示"数据未覆盖"，其他功能不受影响。

### 第 2 步：生成股票代码表（已内置，可跳过）

`data/market/stock_list.csv`（全 A 股代码→名称）已随项目提供。想重新生成（比如更新新股）就运行：

```bash
uv run python scripts/fetch_stock_list.py
```

### 第 3 步（可选）：灌 Neo4j 股权图谱（跑一次）

> 需要 Neo4j 已启动（Neo4j Browser 能打开 http://localhost:7474）。

```bash
uv run python scripts/build_graph.py
```

大约 3~5 分钟，会写入约 6000 家公司 / 6 万条持股关系。脚本是"清空重建"，重跑不会重复。

### 第 4 步（可选）：灌 Chroma 研报向量库（跑一次）

> **项目已内置灌好的 `data/chroma/`（1.1GB），正常情况下跳过这一步。**
> 只有你删掉了 `data/chroma/` 才需要重新灌：

```bash
# Windows PowerShell：
$env:HF_ENDPOINT = "https://hf-mirror.com"
uv run python scripts/build_chroma.py

# Git Bash / Linux / Mac：
export HF_ENDPOINT=https://hf-mirror.com
uv run python scripts/build_chroma.py
```

> `HF_ENDPOINT` 只是让模型下载走国内镜像。项目用的是本地模型 `data/models/bge-small-zh-v1.5`，模型已存在，一般不会再触发下载。

---

## 五、运行

### 方式 A：命令行对话（最快验证）

```bash
uv run python agent/orchestrator.py
```

看到 `你:` 提示符就能聊天了。试试：

```
你: 002742 有什么风险事件？
你: 帮我检测一下 300355 的财报造假风险
你: 002742 的股权结构是什么样的？
你: 600519 最近五年的毛利率和 ROE 怎么样？
你: 金奥博3月24日主力资金流向
你: 600519 今天股价多少？
你: 今天有哪些股票涨停了？
你: 贵州茅台上市日期是什么时候？
```

输入 `quit` 或 `exit` 退出。

### 方式 B：可视化界面（Streamlit）

```bash
uv run streamlit run app.py
```

浏览器会自动打开 http://localhost:8501，两个 Tab：
- **💬 智能问答**：和命令行一样的 Agent，流式输出，结束显示调用了哪些工具
- **📅 事件时间轴**：输入 6 位代码 → 事件簇卡片 + 时间轴 + AI 生成事件脉络叙事

### 方式 C：工具冒烟测试（部署后建议先跑）

```bash
uv run python scripts/smoke_tools.py
```

会逐个调用 15 个工具并打印"真实数据 / 降级 / 崩溃"。**所有工具 0 崩溃**即为正常；
行情类工具显示"降级"是网络受限的表现（见第七节 FAQ），不是代码坏了。

---

## 六、15 个工具速查表

Agent 会自动选工具，但你了解它们能更好提问：

| 工具 | 功能 | 典型问题 |
|---|---|---|
| `get_financials` | 财务数据（30+ 指标，支持季报） | "600519 最近的毛利率和 ROE" |
| `detect_financial_fraud` | 财报造假风险（10 条规则+评分） | "002742 有什么财务风险/扫雷" |
| `trace_shareholder` | 股权穿透（Neo4j，最多 5 跳） | "XX 的实际控制人是谁" |
| `get_event_clusters` | 公告事件簇 | "XX 最近发生了什么大事" |
| `search_reports` | 研报语义检索 | "东吴证券的最新研报" |
| `get_market_snapshot` | 实时行情快照 | "600519 今天股价/市盈率/市值" |
| `get_stock_history` | 历史 K 线+区间统计 | "近一个月最低收盘价 / 3月24日收盘价" |
| `get_fund_flow` | 个股主力资金流向 | "金奥博3月24日主力资金流向" |
| `get_fund_flow_rank` | 资金净流入排行 | "今天哪些股票主力资金流入最多" |
| `get_lhb` | 龙虎榜 | "今天龙虎榜 / XX 上龙虎榜了吗" |
| `get_block_trade` | 大宗交易 | "今天大宗交易情况" |
| `get_margin_balance` | 融资融券余额 | "XX 的融资余额/融券卖出量" |
| `get_stock_basic_info` | 公司基本信息 | "贵州茅台上市日期/所属行业" |
| `get_zt_pool` | 涨停池 | "今天有哪些涨停/连板股" |
| `get_board_rank` | 板块行情排名 | "今天哪些板块涨得好" |

---

## 七、常见问题

**Q：运行时报 Neo4j 连接失败（WinError 10061 / ConnectionRefusedError）**
→ Neo4j 服务没启动，或密码不对。先启动数据库，再核对 `shareholder_graph.py` 里的密码。

**Q：行情类工具返回"数据源调用失败 / 暂无可用的行情数据"**
→ 行情数据来自联网接口（akshare）。可能原因：① 没网；② 数据源域名被网络环境封锁（公司代理/防火墙常见，表现是 ProxyError）。系统会如实告知并给出替代建议，**不会编造数字**。换个网络环境通常就能解决；同一会话内查询过一次的数据会走磁盘缓存，不受影响。

**Q：Agent 说"该股票暂未收录"**
→ 正常。公告数据只覆盖 2585 家公司、财报覆盖约 6400 家，不在覆盖范围内的公司 Agent 会如实告知。

**Q：财务预警触发率是不是太高了？**
→ 规则偏灵敏是设计选择：预警 ≠ 定罪，规则负责"把可疑的都捞出来"，Agent 会结合公告事件做二次研判。阈值都写在 `financial_data.py` 里，可调。

**Q：灌 Chroma 时卡在下载模型**
→ 项目用的是本地模型 `data/models/`，正常不会下载。若触发下载，先设 `HF_ENDPOINT=https://hf-mirror.com` 再跑。

**Q：重跑灌库脚本报 id 重复 / 想清空重来**
→ Chroma：删掉 `data/chroma/` 目录再跑；Neo4j：脚本本身"清空重建"，直接重跑即可。

**Q：为什么财报数据和行情数据的数值对不上？**
→ **正常且故意的**。财务数据来自比赛脱敏数据集（母公司报表口径），行情/研报来自真实市场，两者绝对值不可比。系统提示词已强制禁止交叉对账。

**Q：`uv sync` 很慢或报错**
→ 确认 `pyproject.toml` 里阿里云镜像配置还在；报错就删掉 `.venv` 和 `uv.lock` 后重试。

---

## 八、已知限制（口径说明）

1. 财报数据是**母公司报表**（非合并报表），部分指标口径与合并报表不同，规则设计时已做适配
2. 股权数据只含**前十大股东**（一阶），多层穿透依赖股东恰为上市公司，自然人/非上市壳公司链路暂无法穿透
3. 公告数据为**风险事件导向样本**（监管处罚类占比高），不代表全市场公告
4. 行情类功能**依赖联网**，现场断网时只能走磁盘缓存（建议赛前用工具预热缓存 `data/market_cache/`）
5. 研报检索为语义召回，未做重排序与相似度阈值过滤

---

## 九、给参赛同学的三句话

1. **代码入口**：`agent/orchestrator.py`（主 Agent）→ `agent/tools/`（15 个工具）→ `app.py`（界面）
2. **赛题对应**：任务 1（记忆+自纠错）在 orchestrator 的记忆/摘要与工具错误兜底；任务 2（股权/事件）在 shareholder_graph 与 get_event_clusters；任务 3（财务反欺诈）在 financial_data
3. **演示之前**：先跑 `uv run python scripts/smoke_tools.py` 确认工具正常，再启动 `uv run streamlit run app.py`
