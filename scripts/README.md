# scripts 使用说明

只含三部分：目录约定、工作流程、可直接执行的命令。所有命令都在 `/data3/paper_analysis`
下执行；`<批次>` 是占位符，本文示例批次为 `catch_20260915`。

## 1. 目录约定

```text
human_notes/Catch_Paper_Urls.md            监控的 GitHub 论文列表 + 兴趣主题
paper_catch/YYYYMMDD_HHMMSS.md             paper_catch 常规报告（含 sidecar，下次增量的 baseline）
paper_catch/backfill_<sourceId>/           单来源历史回填（独立 .state/.runs）
paper_catch/<run>_merged.md                合并报告
paper_catch/<run>_merged.titles.md         供 paper_download.py --file 的标题列表
papers_pdf/paper_<批次>/                    PDF（results.json 为下载记录；slides/ 子目录会被忽略）
papers_md/md_<批次>/<论文>/                 Marker Markdown + _meta.json + 导出图片（*.jpeg）
paper_secs/secs_<批次>/<论文>/              按一级标题拆分的章节 + 图片
paper_extract_checkpoints/<批次>/           run_all_papers 的 logs/ status/ progress.json
repos/repo_<批次>/{experiment,idea,knowledge}_repo/   结构化分析汇总
experiment_notes/ idea_notes/ knowledge_notes/        repo_mdsplit_batch 拆出的独立笔记（Obsidian vault）
learning_outputs_codex/ experiment_outputs_codex/     Simple Semantic Loop / Direction Experiment Loop 的 run
review_notes/                              Idea Review 结果
```

脚本一览：

| 脚本 | 作用 |
|---|---|
| `paper_catch.ts` | 监控 Git 论文列表增量，Claude CLI（默认 `claude-sonnet-5`，`--provider codex` 可切）语义筛选 |
| `paper_catch_backfill.ts` | 对单个来源 URL 从 `--since` 起做一次性历史回填 |
| `paper_catch_merge.ts` | 合并多个 paper catch 报告并生成下载用标题列表 |
| `paper_download.py` | 按标题/标题文件下载 PDF（不调用模型） |
| `pdf_to_md.py` | Marker PDF → Markdown（本地 OCR） |
| `md_image_ocr_injector.py` | 对 Markdown 引用的图片逐张 OCR，文字原位插入图片前 |
| `paper_mdsplit_batch.py` | 按一级标题拆分 Markdown 并复制图片 |
| `run_all_papers.py` | 逐篇执行 `paper-experiment-idea` + `paper-knowledge`（Claude CLI） |
| `run_all_papers_multi_launch*.py` | 并行版，配置写在脚本顶部 `CONFIGS`，无 dry-run |
| `repo_mdsplit_batch.py` | 把 repo 汇总按 `##` 拆成独立笔记 |
| `simple_semantic_loop.ts` | Learning Workflow（Codex）；详见 [simple_semantic_loop/README.md](simple_semantic_loop/README.md) |
| `direction_experiment_loop.ts` | Direction 实验深化；详见 [direction_experiment_loop/README.md](direction_experiment_loop/README.md) |
| `learning_scheduler.ts` / `monitor_progress.sh` | Legacy 四阶段学习调度与进度面板 |
| `idea_review_orchestrator.ts` | QA/AA 双会话盲评 Idea |
| `obsidian_api.md` / `obsidian_service/` | Obsidian REST API / MCP 用法与 systemd 托管，见 [obsidian_api.md](obsidian_api.md) |

## 2. 工作流程

```text
Catch_Paper_Urls.md
  → paper_catch.ts run（增量）+ paper_catch_backfill.ts（历史回填）
  → paper_catch_merge.ts                → <run>_merged.md / .titles.md
  → paper_download.py --file            → papers_pdf/paper_<批次>/
  → pdf_to_md.py batch（保留图片）       → papers_md/md_<批次>/
  → md_image_ocr_injector.py            → 图片 OCR 文字原位注入（回写同目录，留 .bak）
  → paper_mdsplit_batch.py              → paper_secs/secs_<批次>/
  → run_all_papers.py --model claude-sonnet-5 → repos/repo_<批次>/
  → repo_mdsplit_batch.py               → *_notes/（Obsidian 立即可查）
  → simple_semantic_loop.ts / direction_experiment_loop.ts / idea_review_orchestrator.ts
```

规则：

- 每个批次用独立的 `paper_<批次>` / `md_<批次>` / `secs_<批次>` / `repo_<批次>` 目录。
- `pdf_to_md.py` **不要**加 `--disable_image_extraction`，否则图片 OCR 注入无事可做；`--force_ocr` 非默认项。
- `md_image_ocr_injector.py` 在拆分前执行；可重复运行，已注入的图片自动跳过。
- `run_all_papers.py` 会启动付费 Agent，先 `--dry-run`；`progress.json` 里 `done` 的论文重跑时跳过。
- Obsidian 必须在线才能用 `obsidian` MCP：`systemctl --user status obsidian.service`。

## 3. 可执行命令

### 3.1 论文抓取 → 合并

```bash
cd /data3/paper_analysis

node scripts/paper_catch.ts doctor
node scripts/paper_catch.ts scan                      # 只抓 Git、冻结候选，不调模型
node scripts/paper_catch.ts run                       # 前台增量 run（Claude sonnet 5）
node scripts/paper_catch.ts start && node scripts/paper_catch.ts status   # 后台 run
node scripts/paper_catch.ts run --provider codex      # 切回 Codex
node scripts/paper_catch.ts validate

# 新来源历史回填（示例：LLM-for-Systems 小节，2024 年起）
node scripts/paper_catch_backfill.ts run \
  --url "https://github.com/AmberLJC/LLMSys-PaperList#llm-for-systems" --since 2024-01-01

# 合并：第一个目录为主，重复标题以其为准；同时产出 .titles.md
node scripts/paper_catch_merge.ts paper_catch paper_catch/backfill_llm_for_systems

node --test scripts/paper_catch/tests/*.test.ts
```

环境变量默认：`PAPER_CATCH_PROVIDER`、`PAPER_CATCH_MODEL`、`PAPER_CATCH_CLAUDE_BIN`、
`PAPER_CATCH_CODEX_BIN`；更多开关见 `node scripts/paper_catch.ts` 的 usage 和
[paper_catch/README.md](paper_catch/README.md)。

### 3.2 下载 PDF（`paper_download.py`）

功能：按标题或标题文件下载公开可访问的论文 PDF，不调用模型。它复用
`/data3/Projects/agent_research/download_papers.py`（可用 `--backend PATH` 或
`PAPER_DOWNLOAD_BACKEND` 覆盖）。每篇论文按顺序尝试：

1. 标题文件里给出的链接（`source_link`）：arXiv abs/pdf、usenix、mlsys 等直接可解析 PDF 的地址
2. 探测来源页面（`source_page`）：从链接页面找 PDF 地址（`--skip-source-discovery` 关闭）
3. arXiv 标题检索（`--no-arxiv-fallback` 关闭）
4. OpenAlex 按 DOI/标题检索开放版本（`--no-oa-fallback` 关闭）
5. DBLP 检索（`--no-dblp-fallback` 关闭）
6. 已记录的公开来源表（`--no-known-public-fallback` 关闭）

标题文件格式（`--file`，可重复）：

- 含 Markdown 链接的行按 `[标题](链接)` 解析，链接作为首选来源——`paper_catch_merge.ts`
  生成的 `<run>_merged.titles.md` 就是这种形式
- 纯 Markdown 标题行（`#` 开头且无链接）跳过；表格分隔行、代码围栏跳过
- 其余非空行整行视为论文标题（去掉列表符号、`*_` 等修饰）
- 合并报告 `<run>_merged.md` **不能**直接作为标题文件（条目行会被当作标题）

```bash
cd /data3/paper_analysis

# 从合并结果批量下载：先 --dry-run 看解析出的标题与来源，再正式下载
python3 scripts/paper_download.py \
  --file paper_catch/20260915_223659_merged.titles.md \
  --output papers_pdf/paper_catch_20260915 --dry-run
python3 scripts/paper_download.py \
  --file paper_catch/20260915_223659_merged.titles.md \
  --output papers_pdf/paper_catch_20260915 --delay 0

# 单篇 / 多篇按标题下载（--title 可重复）
python3 scripts/paper_download.py \
  --title "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" \
  --output papers_pdf/paper_catch_20260915

# 只信任标题文件里的链接，不做来源页探测与检索回退（更快、更可控）
python3 scripts/paper_download.py \
  --file paper_catch/20260915_223659_merged.titles.md \
  --output papers_pdf/paper_catch_20260915 \
  --skip-source-discovery --no-arxiv-fallback --no-oa-fallback --no-dblp-fallback

# 网络慢时放宽超时（秒）与间隔
python3 scripts/paper_download.py --file <标题文件> --output <目录> --max-time 120 --delay 1.5
```

输出：PDF 命名为去掉标点的标题（`AccelOpt Self-Improving LLM Agentic System for Kernel Optimization.pdf`），
同目录写 `results.json`：顶层 `total / success / downloaded / exists / failed`，`results[]`
每条含 `title`、`status`（`downloaded` / `exists` / `failed`）、`pdf_source`、`pdf_url`、
`reason`（如 `no_pdf_url_found`、`response_is_not_pdf_or_too_small`）。重复运行会跳过已存在文件。

检查与补漏：

```bash
# 列出失败项及原因
python3 -c "import json;[print(x['status'],x.get('reason'),'|',x['title']) for x in json.load(open('papers_pdf/paper_catch_20260915/results.json'))['results'] if x['status']!='downloaded']"

# 逐个确认拿到的是论文而不是幻灯片/网页（页数、页面尺寸、首页文字）
for f in papers_pdf/paper_catch_20260915/*.pdf; do echo "== $f"; pdfinfo "$f" | grep -E '^Pages|^Page size'; done
```

- `failed` 且原因为 `no_pdf_url_found` / 非 PDF 响应：多为 ACM DL（对非浏览器客户端返回 403）
  或只有网页版的报告。用浏览器打开 DOI 链接下载，按上述命名规则存入同目录；
  也可先在 arXiv / NSF PAR / Hugging Face 找开放副本，再用 `--title` 或直接 `curl -o` 保存。
- 会议页（如 `mlsys.org/.../Slides/*.pdf`）常被当作论文下载下来，实际是幻灯片：
  移到 `papers_pdf/paper_<批次>/slides/`，再用 `--title` 或 arXiv 链接补正文。
  `pdf_to_md.py batch` 只处理目录直下的 PDF，子目录不会被转换。
- 手工补下的来源记在 `papers_pdf/paper_<批次>/manual_fetch.json`（自由格式），`results.json` 保持下载器原始记录。

### 3.3 PDF → Markdown → 图片 OCR 注入 → 章节拆分

```bash
# 1. 转 Markdown（保留图片）
python3 scripts/pdf_to_md.py batch papers_pdf/paper_catch_20260915 \
  --output papers_md/md_catch_20260915 --workers 2 --skip-existing

# 2. 图片 OCR 原位注入（两张 4090，每卡 2 个持久 worker）
python3 scripts/md_image_ocr_injector.py \
  --path papers_md/md_catch_20260915 \
  --torch-device cuda --cuda-devices 0,1 --workers-per-gpu 2

# 3. 拆章节
python3 scripts/paper_mdsplit_batch.py papers_md/md_catch_20260915 paper_secs/secs_catch_20260915

# 数量核对
find papers_pdf/paper_catch_20260915 -maxdepth 1 -iname '*.pdf' | wc -l
find papers_md/md_catch_20260915 -mindepth 1 -maxdepth 1 -type d | wc -l
find paper_secs/secs_catch_20260915 -mindepth 1 -maxdepth 1 -type d | wc -l
grep -l '图片提取文字' papers_md/md_catch_20260915/*/*.md | wc -l
```

单篇转换：`python3 scripts/pdf_to_md.py single "papers_pdf/paper_catch_20260915/<文件>.pdf" --output papers_md/md_catch_20260915`。
Marker 位置可用 `MARKER_ROOT` / `MARKER_PYTHON` 覆盖（默认 `/data3/Projects/marker`、`~/miniconda3/bin/python3`）。

### 3.4 结构化分析 → 笔记

`run_all_papers.py` 通过裸 `claude` 命令逐篇启动 Agent，模型来自 `--model`，不传时用脚本
第 38 行的默认值 `deepseek-v4-flash[1m]`。两种模型来源：

| 模型 | 走哪里 | 前提 |
|---|---|---|
| `claude-sonnet-5` 等官方模型 | Anthropic 官方账号（已 `/login`） | 无 |
| `deepseek-v4-flash[1m]` / `deepseek-v4-pro[1m]` | 本地代理 `~/deepseek-local-proxy/proxy.py`（`127.0.0.1:8787`）转发到 DeepSeek | 代理在跑 **且** 当前命令带 `ANTHROPIC_BASE_URL` 等变量 |

`.bashrc` **不会**导出这些变量（登录时反而 `unset ANTHROPIC_*`，保证默认走官方）；它只定义了
`deepseek-proxy-ensure`（拉起代理）和 `cc-deepseek`（仅对那一次 `claude` 注入变量）。
`run_all_papers.py` 不经过 `cc-deepseek`，所以用 DeepSeek 时必须在**这条命令上手动注入**，
否则请求发到官方 API，日志出现 `unrecognized_model` / `terminal_reason: api_error`，每篇立即失败。

#### 3.4.1 用 DeepSeek（默认模型）

```bash
cd /data3/paper_analysis
deepseek-proxy-ensure                                   # .bashrc 函数；已在跑则直接返回
ss -ltn | grep ':8787' && tail -2 /tmp/deepseek-proxy.log

# token 直接取 .bashrc 里 cc-deepseek 用的那一个，不要复制到别处
DS_TOKEN=$(grep -oP 'ANTHROPIC_AUTH_TOKEN="\K[^"]+' ~/.bashrc)

ANTHROPIC_BASE_URL="http://127.0.0.1:8787" \
ANTHROPIC_AUTH_TOKEN="$DS_TOKEN" \
ANTHROPIC_MODEL="deepseek-v4-flash[1m]" \
ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-pro[1m]" \
ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash[1m]" \
ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash[1m]" \
CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash[1m]" \
python3 scripts/run_all_papers.py \
  --paper-base-dir paper_secs/secs_catch_20260915 \
  --checkpoint-dir paper_extract_checkpoints/catch_20260915 \
  --output-repo-dir repos/repo_catch_20260915
```

要用 `deepseek-v4-pro[1m]`，加 `--model "deepseek-v4-pro[1m]"`。变量只作用于这一条命令；
若想在当前 shell 连续跑多个批次，把上面 7 个赋值改成 `export ...`，跑完 `unset` 回官方。

#### 3.4.2 用官方 Claude 模型

```bash
python3 scripts/run_all_papers.py \
  --paper-base-dir paper_secs/secs_catch_20260915 \
  --checkpoint-dir paper_extract_checkpoints/catch_20260915 \
  --output-repo-dir repos/repo_catch_20260915 \
  --model claude-sonnet-5
```

#### 3.4.3 通用

```bash
# 先 dry-run 核对路径和 prompt（不启动 Agent、不写 progress）；DeepSeek 时同样带上 3.4.1 的变量
python3 scripts/run_all_papers.py --paper-base-dir ... --checkpoint-dir ... --output-repo-dir ... --dry-run

# 只跑一篇 / 前 N 篇
python3 scripts/run_all_papers.py ... --title "<论文子目录名>"
python3 scripts/run_all_papers.py ... --limit 3

# 进度与确认：progress.json 只跳过 done，failed 会在重跑时重试
python3 -c "import json;p=json.load(open('paper_extract_checkpoints/catch_20260915/progress.json'));print('done',len(p['done']),'failed',len(p['failed']),p['last_updated'])"
grep -o '"terminal_reason":"[^"]*"' paper_extract_checkpoints/catch_20260915/logs/001_*.jsonl | tail -1   # 期望 completed
grep -l 'unrecognized_model' paper_extract_checkpoints/catch_20260915/logs/*.jsonl                        # 有输出 = 没走代理

# 拆成独立笔记，写入 vault 根目录下的 experiment_notes / idea_notes / knowledge_notes
python3 scripts/repo_mdsplit_batch.py repos/repo_catch_20260915 --notes-base /data3/paper_analysis
```

### 3.5 Learning / Direction / Idea Review

```bash
# Simple Semantic Loop（Codex）
node scripts/simple_semantic_loop.ts doctor
node scripts/simple_semantic_loop.ts init --topic '<主题>' --objective '<目标>' --acceptance '<标准>' \
  --max-rounds 8 --max-exp-goals 5 --work-dir learning_outputs_codex/<run>
node scripts/simple_semantic_loop.ts run --yolo --work-dir learning_outputs_codex/<run>
node scripts/simple_semantic_loop.ts resume --additional-rounds 4 --work-dir learning_outputs_codex/<run>

# Direction Experiment Loop
node scripts/direction_experiment_loop.ts init \
  --direction-result learning_outputs_codex/<source_run>/results/<direction_turn>.json \
  --max-cycles 5 --work-dir experiment_outputs_codex/<direction_run>
node scripts/direction_experiment_loop.ts run --yolo --work-dir experiment_outputs_codex/<direction_run>

# Idea Review（写入 review_notes/）
npx tsx scripts/idea_review_orchestrator.ts \
  --idea-note "idea_notes/<Idea note>.md" \
  --work-dir ".claude/idea-review-runs/<短名>" --max-rounds 8 --max-budget-usd 100
npx tsx scripts/idea_review_orchestrator.ts ... --resume
npx tsx scripts/idea_review_orchestrator.test.ts

# Legacy 学习调度
npx tsx scripts/learning_scheduler.ts --work-dir learning_outputs --user-input "<研究问题>"
watch -n 5 -c scripts/monitor_progress.sh <run 目录>
```

### 3.6 Obsidian（`obsidian` MCP 后端）

```bash
systemctl --user status obsidian.service            # 窗口/无头模式、重启次数
systemctl --user restart obsidian.service           # 任意 shell/SSH 拉起或重新判定模式
journalctl --user -u obsidian.service -f -o cat
curl -s http://127.0.0.1:27123/ | head -3           # REST API 健康
```

安装、REST API curl 用法与排障见 [obsidian_api.md](obsidian_api.md)。

### 3.7 环境检查

```bash
python3 -m py_compile scripts/*.py
mdsplit --help      # 报 ModuleNotFoundError 时：/usr/bin/python3 -m pip install --user --break-system-packages mdsplit
npx tsx --version
node --test scripts/paper_catch/tests/*.test.ts
bash -n scripts/monitor_progress.sh
```
