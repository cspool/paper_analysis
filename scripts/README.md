# scripts 使用说明

本目录包含论文下载、PDF 转 Markdown、论文分析、笔记拆分、学习工作流和 Idea Review 编排脚本。

建议先区分三类操作：

- **纯本地处理**：`paper_mdsplit_batch.py`、`repo_mdsplit_batch.py`、`monitor_progress.sh`。
- **联网或模型处理**：`paper_download.py`、`pdf_to_md.py`。
- **会启动付费 Agent**：`run_all_papers*.py`、`simple_semantic_loop.ts`、`direction_experiment_loop.ts`、`learning_scheduler.ts`、`idea_review_orchestrator.ts`。

## 正式目录约定

当前项目的正式论文处理链路是：

```text
/data3/paper_analysis/papers_pdf/<论文批次>
  --pdf_to_md.py-->
/data3/paper_analysis/papers_md/<Markdown 批次>
  --paper_mdsplit_batch.py-->
/data3/paper_analysis/paper_secs/<章节批次>
  --run_all_papers.py-->
/data3/paper_analysis/repos/<repo 批次>
  --repo_mdsplit_batch.py-->
/data3/paper_analysis/{experiment_notes,idea_notes,knowledge_notes}
```

各根目录职责：

- `/data3/paper_analysis/papers_pdf`：批量下载 PDF 的目标根目录，也是 `pdf_to_md.py` 的来源。
- `/data3/paper_analysis/papers_md`：批量 PDF 转 Markdown 后的目标根目录，也是 `paper_mdsplit_batch.py` 的来源。
- `/data3/paper_analysis/paper_secs`：`paper_mdsplit_batch.py` 的拆分后目录，也是 `run_all_papers.py` 的来源。
- `/data3/paper_analysis/repos`：`run_all_papers.py` 的输出根目录；每个批次应使用一个独立 `repo_*` 子目录。
- 项目根目录下的 `experiment_notes`、`idea_notes`、`knowledge_notes`：`repo_mdsplit_batch.py` 的正式输出目录。

`paper_mdsplit_batch.py` 和 `run_all_papers.py` 都读取 Markdown 目录，不能直接将 `papers_pdf` 作为输入。

临时或测试示例与正式路径的对应关系：

| 临时/测试路径 | 正式路径 |
|---|---|
| `paper_tool_test/SGLang*/pdf` | `/data3/paper_analysis/papers_pdf/paper_2026` |
| `paper_tool_test/SGLang*/markdown` | `/data3/paper_analysis/papers_md/md_2026` |
| `temp/script_single_paper_test/paper_raw` | `/data3/paper_analysis/papers_md/md_2026` |
| `temp/script_single_paper_test/paper_split` | `/data3/paper_analysis/paper_secs/secs_2026` |
| `temp/script_single_paper_test/checkpoints` | `/data3/paper_analysis/paper_extract_checkpoints/2026` |
| `temp/script_single_paper_test/repo_output` | `/data3/paper_analysis/repos/repo_2026` |
| `temp/script_single_paper_test/notes_split` | `/data3/paper_analysis`，由脚本写入根目录下的 `*_notes` |
| `temp/learning_runs` | `/data3/paper_analysis/learning_outputs` |
| `temp/idea_review_runs/example` | `/data3/paper_analysis/.claude/idea-review-runs/<论文短名>` |

## 脚本索引

| 文件 | 用途 | 主要输入 | 主要输出 |
|---|---|---|---|
| `paper_download.py` | 按标题或标题文件下载公开可访问论文 | `--title` 或 `--file` | PDF、`results.json` |
| `pdf_to_md.py` | 复用 Marker 单篇或批量 PDF 转 Markdown/OCR | PDF 或 PDF 目录 | 每篇论文一个 Markdown 子目录 |
| `paper_mdsplit_batch.py` | 按一级标题拆分每篇 Marker Markdown，并复制 JPEG | 每篇论文一个子目录的根目录 | 可供分析 Agent 使用的论文子目录 |
| `run_all_papers.py` | 顺序运行 `paper-experiment-idea` 和 `paper-knowledge` | 论文子目录根目录 | experiment/idea/knowledge 汇总与 checkpoint |
| `run_all_papers_multi_launch.py` | 并行运行 effAttn、rtrans_ssm、video_image 等硬编码配置 | 脚本内 `CONFIGS` | 多组 repo 与进度文件 |
| `run_all_papers_multi_launch_2nd_process.py` | 并行运行 MoE、多模态 kernel 等第二组硬编码配置 | 脚本内 `CONFIGS` | 多组 repo 与进度文件 |
| `repo_mdsplit_batch.py` | 将汇总 repo 按 `##` 标题拆成独立笔记 | `experiment_repo/idea_repo/knowledge_repo` | experiment/idea/knowledge notes |
| `simple_semantic_loop.ts` | 当前 Learning Workflow format v8：最小核心字段 gate 的确定性文件型 Controller 编排 Decision、Worker、Reviewer fresh Turn，并可由 Decision 按需启动持久 EXP Goal；EXP 后先走原子 Reviewer→Decision，按 Anchor 索引经审阅负结果并反馈后续搜索；支持从 FINISHED 继续，或从稳定 PAUSED 进展不可变分支并重置授权 | topic、objective、acceptance criteria，可选 EXP Goal 数量和 timeout，或 FINISHED/PAUSED source run；EXP 不设置 token budget | G01/T01/D01/W01/R01/E01、实验 Goal task/result/workspace、负 EXP Ref 索引、continuation provenance、core control 投影、Context snapshots、trajectory/memory/checkpoint、授权/recovery 审计和 `final/report.md` |
| `direction_experiment_loop.ts` | 当前 format v7：对一个已审阅 Direction 做实验深化；fresh Decision 每轮冻结一个可执行原子合同，持久 Lab 以 Stop Gate、分片和 checkpoint/result 原子提交执行，fresh Judge 独立审查早停和证据范围；支持 timeout/blocked 后结果接管、同线程 invocation 恢复与锁外实时暂停 | Learning run 中一个明确的 Direction WORK_RESULT 路径 | run-local Skill 快照、合同/Cycle 绑定、invocation/checkpoint/result、Decision/Experiment/Judgment 轨迹、带 scope 的最终报告与 Learning handoff |
| `learning_scheduler.ts` | 四阶段学习调度：问题、回答、横向总结、纵向总结 | 自然语言研究问题 | `learning_outputs` run |
| `monitor_progress.sh` | 只读显示 learning scheduler 进度 | learning run 目录 | 终端进度面板 |
| `idea_review_orchestrator.ts` | QA/AA 双会话盲评 Idea | Idea note 路径或论文标题 | review、运行日志与 checkpoint |
| `idea_review_orchestrator.test.ts` | Idea Review marker/protocol 单元测试 | 无 | 测试结果 |
| `tmp_titles.md` | 单标题下载测试输入，不是可执行脚本 | 一行或多行论文标题 | 供 `paper_download.py --file` 使用 |

## Learning Workflow 状态

- **当前 Codex 工作流**：`simple_semantic_loop.ts` 与
  `simple_semantic_loop/`，通信以
  `workflow_goal.json`、`turn_task.json`、`decision_context.json` 和
  W01/R01/E01 为边界。完整使用说明见
  [Simple Semantic Loop README](simple_semantic_loop/README.md)。
  `maxRounds` 是初始授权窗口；耗尽时会保存已准备好的下一 Round 并返回
  `PAUSED`。使用 `resume --additional-rounds N` 可一次授权多轮，省略 N 时默认
  再授权初始 `maxRounds` 轮。
  修改前的正式 v6 运行与实现快照保存在
  [official_runs/multimodal_inference_latency_first_v6_20260801_round12](../archive/learning_workflow/official_runs/multimodal_inference_latency_first_v6_20260801_round12/OFFICIAL_SNAPSHOT.md)。
- **已退役工作流**：旧四角色 SQLite/Stage/Gate Simple Semantic Loop、旧
  Codex 入口、模块、测试、旧 Skill，以及更早的
  standalone Layered Exploration Python workflow 和单体 Skill，均已统一
  迁入 `archive/learning_workflow/`；`scripts/` 和活动 Skill 目录不再保留
  这些入口。
- **更早的 legacy/provenance 工作流**：`learning_scheduler.ts` 与配套的
  `monitor_progress.sh`。它们是 Claude 四阶段实现，仍保留独立命令说明，但不
  参与当前 Simple Semantic Loop。

旧版 Codex 工作流的设计、Skill 和运行时归档索引见
[archive/learning_workflow/README.md](../archive/learning_workflow/README.md)。

## Direction Experiment Loop

当 Learning Flow 已经给出一个具体 Direction，需要围绕其 baseline 做环境部署、
实现、A/B 和消融时，使用独立的
[Direction Experiment Loop README](direction_experiment_loop/README.md)。该 Flow
不接收 Topic，也不创建新的 Anchor。它从一个不可变来源 Direction 开始，按
`Experiment Decision → Direction Lab Goal → Evidence Judge → Experiment Decision`
闭环：Decision 冻结同一因果主张的实验合同及允许弱化，Lab 只执行，Judge 只评判；
若核心研究主张必须改变则回交 Learning Flow。弱化代理、真实单卡、模拟器和原论文
环境的证据边界由 `evidenceScope` 写入最终 handoff。

## 环境检查

```bash
# Python 脚本语法
python3 -m py_compile scripts/*.py

# Markdown 拆分工具
mdsplit --help

# TypeScript runner
npx tsx --version

# Idea Review 协议测试
npx tsx scripts/idea_review_orchestrator.test.ts

# Shell 脚本语法
bash -n scripts/monitor_progress.sh
```

## 论文下载与 PDF 转 Markdown

这两个接口脚本只从当前仓库调用已有工具，不迁移环境或复制实现：

- `paper_download.py` 复用 `/data3/agent_research/download_papers.py`
- `pdf_to_md.py` 复用 `/home/descfly/Desktop/marker` 和现有 Miniconda Marker 环境

所有相对输出路径均以运行命令时的当前目录为基准。

## 按标题下载论文

直接提供标题：

```bash
python3 scripts/paper_download.py \
  --title "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" \
  --output /data3/paper_analysis/papers_pdf/paper_2026
```

批量读取标题文件。文件中的 Markdown 标题行会跳过，其余非空行视为论文标题；也支持 Markdown 论文链接：

```bash
python3 scripts/paper_download.py \
  --file /data3/paper_analysis/papers_pdf/paper_titles_2026.md \
  --output /data3/paper_analysis/papers_pdf/paper_2026
```

可重复使用 `--title` 或 `--file`。大批量下载前可先检查解析结果：

```bash
python3 scripts/paper_download.py \
  --file /data3/paper_analysis/papers_pdf/paper_titles_2026.md \
  --output /data3/paper_analysis/papers_pdf/paper_2026 \
  --dry-run

# 当前 scripts 下的一行标题测试文件
python3 scripts/paper_download.py --file scripts/tmp_titles.md --dry-run
```

下载器依次尝试已有脚本提供的来源页、arXiv、OpenAlex、DBLP 和已记录公开来源，并在输出目录写入 `results.json`。

## PDF 转 Markdown

转换单篇论文：

```bash
python3 scripts/pdf_to_md.py single \
  "/data3/paper_analysis/papers_pdf/paper_2026/<论文文件名>.pdf" \
  --output /data3/paper_analysis/papers_md/md_2026
```

对整篇论文强制执行 OCR：

```bash
python3 scripts/pdf_to_md.py single \
  "/data3/paper_analysis/papers_pdf/paper_2026/<论文文件名>.pdf" \
  --output /data3/paper_analysis/papers_md/md_2026 \
  --force_ocr
```

批量转换目录中直接包含的全部 PDF：

```bash
python3 scripts/pdf_to_md.py batch \
  /data3/paper_analysis/papers_pdf/paper_2026 \
  --output /data3/paper_analysis/papers_md/md_2026 \
  --workers 2 \
  --skip-existing
```

批量接口只向 Marker 提交 PDF，因此会自动忽略下载目录中的 `results.json` 等文件。未被接口识别的参数会继续传给 Marker，例如：

```bash
python3 scripts/pdf_to_md.py single \
  "/data3/paper_analysis/papers_pdf/paper_2026/<论文文件名>.pdf" \
  --output /data3/paper_analysis/papers_md/md_2026 \
  --page_range 0-5 \
  --disable_image_extraction
```

只检查将要执行的 Marker 命令：

```bash
python3 scripts/pdf_to_md.py batch \
  /data3/paper_analysis/papers_pdf/paper_2026 \
  --output /data3/paper_analysis/papers_md/md_2026 \
  --dry-run
```

如源工具位置以后发生变化，可临时覆盖：

```bash
PAPER_DOWNLOAD_BACKEND=/path/to/download_papers.py python3 scripts/paper_download.py ...
MARKER_ROOT=/path/to/marker MARKER_PYTHON=/path/to/python python3 scripts/pdf_to_md.py ...
```

## 已验证单样本：SGLang

以下命令已于 2026-06-14 在当前仓库实际运行，完成了按标题下载和整篇强制 OCR 转换。
测试产物目录被清理后，可使用相同命令重新生成。

下载论文：

```bash
python3 scripts/paper_download.py \
  --title "SGLang: Efficient Execution of Structured Language Model Programs" \
  --output paper_tool_test/SGLang/pdf \
  --delay 0
```

正式目录参考：

```bash
python3 scripts/paper_download.py \
  --title "SGLang: Efficient Execution of Structured Language Model Programs" \
  --output /data3/paper_analysis/papers_pdf/paper_2026 \
  --delay 0
```

使用 Marker 对整篇论文强制执行 OCR 并转成 Markdown：

```bash
python3 scripts/pdf_to_md.py single \
  "paper_tool_test/SGLang/pdf/SGLang Efficient Execution of Structured Language Model Programs.pdf" \
  --output paper_tool_test/SGLang/markdown \
  --force_ocr \
  --disable_image_extraction
```

正式目录参考：

```bash
python3 scripts/pdf_to_md.py single \
  "/data3/paper_analysis/papers_pdf/paper_2026/SGLang Efficient Execution of Structured Language Model Programs.pdf" \
  --output /data3/paper_analysis/papers_md/md_2026 \
  --force_ocr \
  --disable_image_extraction
```

`--force_ocr` 强制所有页面走 OCR；`--disable_image_extraction` 只关闭图片文件导出，不会关闭 OCR。

验证结果：

- 下载来源：`https://arxiv.org/pdf/2312.07104.pdf`
- PDF：20 页，1,383,463 字节
- Marker 元数据记录全部 20 页的 `text_extraction_method` 均为 `surya`
- OCR 转换耗时约 85.48 秒
- 输出 Markdown：76,416 字节，11,165 个单词
- 抽检可识别标题、Abstract、RadixAttention 正文和附录；双栏代码、公式与图表附近仍可能出现 OCR 排版或数值误识别，需要人工校对
- 下载记录：`paper_tool_test/SGLang/pdf/results.json`
- Markdown 与元数据：`paper_tool_test/SGLang/markdown/SGLang Efficient Execution of Structured Language Model Programs/`

## 已验证批处理：临时标题文件 + SGLang

以下命令已于 2026-06-14 在当前仓库实际运行，用只包含一个论文标题的临时 Markdown 验证了 `--file` 批量下载和 `batch` 批量强制 OCR 转换。
测试产物目录被清理后，可使用相同命令重新生成。

建立临时标题文件：

```bash
install -d paper_tool_test/SGLang_batch
printf '%s\n' \
  'SGLang: Efficient Execution of Structured Language Model Programs' \
  > paper_tool_test/SGLang_batch/tmp_titles.md
```

正式目录参考；以下命令建立一个单论文标题文件，不覆盖已有的批量标题文件：

```bash
install -d /data3/paper_analysis/papers_pdf
printf '%s\n' \
  'SGLang: Efficient Execution of Structured Language Model Programs' \
  > /data3/paper_analysis/papers_pdf/paper_titles_single.md
```

当前仓库还已有批量标题文件：

```text
/data3/paper_analysis/papers_pdf/paper_titles_2026.md
```

通过标题文件批量下载：

```bash
python3 scripts/paper_download.py \
  --file paper_tool_test/SGLang_batch/tmp_titles.md \
  --output paper_tool_test/SGLang_batch/pdf \
  --delay 0
```

正式目录参考：

```bash
python3 scripts/paper_download.py \
  --file /data3/paper_analysis/papers_pdf/paper_titles_2026.md \
  --output /data3/paper_analysis/papers_pdf/paper_2026 \
  --delay 0
```

对下载目录中的全部 PDF 批量执行强制 OCR：

```bash
python3 scripts/pdf_to_md.py batch \
  paper_tool_test/SGLang_batch/pdf \
  --output paper_tool_test/SGLang_batch/markdown \
  --workers 1 \
  --force_ocr \
  --disable_image_extraction
```

正式目录参考：

```bash
python3 scripts/pdf_to_md.py batch \
  /data3/paper_analysis/papers_pdf/paper_2026 \
  --output /data3/paper_analysis/papers_md/md_2026 \
  --workers 2 \
  --skip-existing \
  --force_ocr \
  --disable_image_extraction
```

验证结果：

- 临时 Markdown 只包含 1 行标题，下载接口解析出 1 个唯一标题
- 批量下载成功获取 1 个 PDF，并在同目录生成 `results.json`
- 批量转换接口只选择了 1 个 PDF，自动忽略同目录的 `results.json`
- Marker 批处理成功转换 20 页，吞吐约 0.20 页/秒，总耗时约 97.85 秒
- Marker 元数据记录全部 20 页的 `text_extraction_method` 均为 `surya`
- 输出 Markdown：76,419 字节，11,166 个单词
- 临时标题文件：`paper_tool_test/SGLang_batch/tmp_titles.md`
- 下载记录：`paper_tool_test/SGLang_batch/pdf/results.json`
- Markdown 与元数据：`paper_tool_test/SGLang_batch/markdown/SGLang Efficient Execution of Structured Language Model Programs/`

## 论文 Markdown 拆分：paper_mdsplit_batch.py

输入目录要求：

```text
<path1>/
└── <论文标题>/
    ├── <论文 Markdown>.md
    └── *.jpg / *.jpeg
```

脚本遍历 `<path1>` 的直接子目录，每篇论文选择排序后的第一个 `.md` 文件，调用 `mdsplit --max-level 1 --force` 按一级标题拆分，并将伴随 JPEG 复制到对应输出子目录。

```bash
python3 scripts/paper_mdsplit_batch.py <path1> <path2>
```

正式目录示例：

```bash
python3 scripts/paper_mdsplit_batch.py \
  /data3/paper_analysis/papers_md/md_2026 \
  /data3/paper_analysis/paper_secs/secs_2026
```

注意：

- 每个论文子目录应只保留一个需要拆分的主 Markdown。
- 只复制 `.jpg/.jpeg`；不会复制 Marker 的 `_meta.json`。
- 输出仍保持“一篇论文一个子目录”，可直接作为 `run_all_papers.py --paper-base-dir`。

## 顺序论文分析：run_all_papers.py

该脚本按论文子目录顺序处理，每篇论文在一个 Claude context 中依次执行：

1. `paper-experiment-idea`
2. `paper-knowledge`

无参数运行使用 model quant 正式配置：

```bash
python3 scripts/run_all_papers.py
```

其默认路径为：

```text
输入：/data3/paper_analysis/paper_secs/secs_model_quant
checkpoint：/data3/paper_analysis/paper_extract_checkpoints
输出：/data3/paper_analysis/repos/repo_model_quant
```

可配置参数：

```text
--paper-base-dir   直接子目录名为论文标题的输入根目录
--checkpoint-dir   写入 logs/、status/、progress.json
--output-repo-dir  写入 experiment_repo/、idea_repo/、knowledge_repo/
--title            只处理完全匹配的一个论文子目录
--limit            最多处理前 N 篇
--model            覆盖 Claude model
--dry-run          只验证选择和 prompt，不启动 Claude、不写 checkpoint
```

单论文 temp dry-run：

```bash
python3 scripts/run_all_papers.py \
  --paper-base-dir temp/script_single_paper_test/paper_split \
  --checkpoint-dir temp/script_single_paper_test/checkpoints \
  --output-repo-dir temp/script_single_paper_test/repo_output \
  --title "SGLang__Efficient_Execution_of_Structured_Language_Model_Programs" \
  --dry-run
```

正式目录参考：

```bash
python3 scripts/run_all_papers.py \
  --paper-base-dir /data3/paper_analysis/paper_secs/secs_2026 \
  --checkpoint-dir /data3/paper_analysis/paper_extract_checkpoints/2026 \
  --output-repo-dir /data3/paper_analysis/repos/repo_2026 \
  --title "1-Towards High-Goodput LLM Serving with Prefill-decode Multiplexing" \
  --dry-run
```

确认 prompt 中全部路径都位于 temp 后，移除 `--dry-run` 才会真正启动 Agent：

```bash
python3 scripts/run_all_papers.py \
  --paper-base-dir temp/script_single_paper_test/paper_split \
  --checkpoint-dir temp/script_single_paper_test/checkpoints \
  --output-repo-dir temp/script_single_paper_test/repo_output \
  --title "SGLang__Efficient_Execution_of_Structured_Language_Model_Programs"
```

正式目录参考：

```bash
python3 scripts/run_all_papers.py \
  --paper-base-dir /data3/paper_analysis/paper_secs/secs_2026 \
  --checkpoint-dir /data3/paper_analysis/paper_extract_checkpoints/2026 \
  --output-repo-dir /data3/paper_analysis/repos/repo_2026 \
  --title "1-Towards High-Goodput LLM Serving with Prefill-decode Multiplexing"
```

真实执行会产生模型调用费用。`progress.json` 用于恢复；已在 `done` 中的论文会跳过。

## 并行论文分析启动器

`run_all_papers_multi_launch.py` 和 `run_all_papers_multi_launch_2nd_process.py` 将不同论文集合分配给不同进程；每个集合内部仍顺序处理论文。

```bash
python3 scripts/run_all_papers_multi_launch.py
python3 scripts/run_all_papers_multi_launch_2nd_process.py
```

运行前必须检查脚本顶部的：

- `CONFIGS`：每组论文输入、repo 输出、日志和 progress 路径。
- `MAX_WORKERS`：并行配置数量。
- `STOP_ON_FAILURE`、`MODEL_NAME`。

这两个脚本当前没有 CLI 覆盖或 dry-run，不适合用于单论文 temp 测试；单论文验证应使用 `run_all_papers.py`。

## Repo 汇总拆分：repo_mdsplit_batch.py

输入目录要求：

```text
<root_repo_path>/
├── experiment_repo/*.md
├── idea_repo/*.md
└── knowledge_repo/*.md
```

脚本按每个汇总 Markdown 中的 `##` 标题拆分独立笔记。

正式输出到当前 vault：

```bash
python3 scripts/repo_mdsplit_batch.py \
  /data3/paper_analysis/repos/repo_2026 \
  --notes-base /data3/paper_analysis
```

隔离输出到 temp：

```bash
python3 scripts/repo_mdsplit_batch.py \
  temp/script_single_paper_test/repo_output \
  --notes-base temp/script_single_paper_test/notes_split
```

正式目录参考即上面的 `repo_2026 -> /data3/paper_analysis` 命令；输出会进入项目根目录下的 `experiment_notes`、`idea_notes` 和 `knowledge_notes`。

映射规则：

- `idea_repo/*.md` -> `idea_notes/`
- `experiment_repo/实验_编译框架.md` 等固定文件名 -> 对应实验笔记目录
- `knowledge_repo/知识库_系统架构.md` 等固定文件名 -> 对应知识笔记目录
- 未识别的 experiment/knowledge 文件名会警告并跳过

## temp 单论文处理测试链

本测试于 2026-06-14 实际运行，所有输入与输出都位于：

```text
temp/script_single_paper_test/
```

测试链：

```bash
# 1. 将单论文 Markdown 按一级标题拆分
python3 scripts/paper_mdsplit_batch.py \
  temp/script_single_paper_test/paper_raw \
  temp/script_single_paper_test/paper_split

# 2. 验证单论文分析的输入、输出和 checkpoint 路径，不启动 Claude
python3 scripts/run_all_papers.py \
  --paper-base-dir temp/script_single_paper_test/paper_split \
  --checkpoint-dir temp/script_single_paper_test/checkpoints \
  --output-repo-dir temp/script_single_paper_test/repo_output \
  --title "SGLang__Efficient_Execution_of_Structured_Language_Model_Programs" \
  --dry-run

# 3. 将模拟 repo 汇总拆分为独立笔记
python3 scripts/repo_mdsplit_batch.py \
  temp/script_single_paper_test/repo_output \
  --notes-base temp/script_single_paper_test/notes_split
```

对应的正式目录完整处理链：

```bash
# 1. 下载标题文件中的 PDF
python3 scripts/paper_download.py \
  --file /data3/paper_analysis/papers_pdf/paper_titles_2026.md \
  --output /data3/paper_analysis/papers_pdf/paper_2026

# 2. 将 PDF 批量转换为 Markdown
python3 scripts/pdf_to_md.py batch \
  /data3/paper_analysis/papers_pdf/paper_2026 \
  --output /data3/paper_analysis/papers_md/md_2026 \
  --workers 2 \
  --skip-existing

# 3. 将每篇 Markdown 按一级标题拆分
python3 scripts/paper_mdsplit_batch.py \
  /data3/paper_analysis/papers_md/md_2026 \
  /data3/paper_analysis/paper_secs/secs_2026

# 4. 顺序分析拆分后的论文
python3 scripts/run_all_papers.py \
  --paper-base-dir /data3/paper_analysis/paper_secs/secs_2026 \
  --checkpoint-dir /data3/paper_analysis/paper_extract_checkpoints/2026 \
  --output-repo-dir /data3/paper_analysis/repos/repo_2026

# 5. 将 repo 汇总拆分到项目根目录的 *_notes
python3 scripts/repo_mdsplit_batch.py \
  /data3/paper_analysis/repos/repo_2026 \
  --notes-base /data3/paper_analysis
```

实际测试结果：

- `paper_mdsplit_batch.py`：拆出 3 个一级章节，并复制 1 个 JPEG。
- `run_all_papers.py --dry-run`：只选中 1 篇论文，所有注入路径均指向 temp，未启动 Claude，未写 `progress.json`。
- `repo_mdsplit_batch.py`：处理 3 个 repo 汇总文件，拆出 4 条独立笔记。

## Legacy Learning Scheduler（独立保留）

启动新的四阶段学习任务：

```bash
npx tsx scripts/learning_scheduler.ts \
  --work-dir temp/learning_runs \
  --user-input "研究 MoE 在单 GPU 上的多算子并发，侧重 Kernel 调度"
```

正式目录参考：

```bash
npx tsx scripts/learning_scheduler.ts \
  --work-dir /data3/paper_analysis/learning_outputs \
  --user-input "研究 MoE 在单 GPU 上的多算子并发，侧重 Kernel 调度"
```

`--work-dir` 若不是已有 run，会在其下创建带时间戳的子目录；若目录中已有 `dispatch.json`，则恢复该 run。该脚本会启动多个 Agent 并产生模型调用费用。

监控已有 run：

```bash
scripts/monitor_progress.sh <具体 run 目录>

# 或持续刷新
watch -n 5 -c scripts/monitor_progress.sh <具体 run 目录>
```

`monitor_progress.sh` 只读文件系统，不修改 scheduler 或运行中进程。

## Idea Review

通过 Idea note 路径启动盲 QA Review：

```bash
npx tsx scripts/idea_review_orchestrator.ts \
  --idea-note idea_notes/example.md \
  --work-dir temp/idea_review_runs/example \
  --max-rounds 8 \
  --max-budget-usd 100
```

正式目录参考：

```bash
npx tsx scripts/idea_review_orchestrator.ts \
  --idea-note "/data3/paper_analysis/idea_notes/FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving.md" \
  --work-dir "/data3/paper_analysis/.claude/idea-review-runs/FlashInfer" \
  --max-rounds 8 \
  --max-budget-usd 100
```

也可将 `--idea-note` 设为论文标题，让脚本在 `idea_notes/` 中自动匹配。恢复中断任务：

```bash
npx tsx scripts/idea_review_orchestrator.ts \
  --idea-note idea_notes/example.md \
  --work-dir temp/idea_review_runs/example \
  --max-rounds 8 \
  --max-budget-usd 100 \
  --resume
```

正式目录参考：

```bash
npx tsx scripts/idea_review_orchestrator.ts \
  --idea-note "/data3/paper_analysis/idea_notes/FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving.md" \
  --work-dir "/data3/paper_analysis/.claude/idea-review-runs/FlashInfer" \
  --max-rounds 8 \
  --max-budget-usd 100 \
  --resume
```

`--work-dir` 控制运行日志与 checkpoint；最终 review 仍写入正式 `review_notes/`。该流程会启动 QA/AA Agent 并产生模型调用费用。

只测试 marker/protocol 解析，不启动 Review Agent：

```bash
npx tsx scripts/idea_review_orchestrator.test.ts
```
