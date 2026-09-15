# Paper Catch Git/Claude(Codex) Batch Loop

该流程由人工按需要定时调用，但实际抓取和语义筛选（默认 Claude CLI，可切 Codex）可在后台执行。它读取：

```text
/data3/paper_analysis/human_notes/Catch_Paper_Urls.md
```

并把成功完成的报告原子发布为：

```text
/data3/paper_analysis/paper_catch/YYYYMMDD_HHMMSS.md
```

## 工作流

```text
最近成功报告/HEAD 快照
  → fetch 所有 Git URL
  → 统计 baseline..HEAD 的全部 Markdown 更新
  → 对 URL fragment 限定到对应 README 章节
  → 提取、合并、去重新论文标题
  → 固定 batch
  → 每个 batch 启动一个 fresh `claude -p`（或 `codex exec`）会话
  → output-schema + 候选全覆盖校验
  → checkpoint 已完成 batch
  → Script 确定性汇总所有 batch Result
  → 按 PAPER_ENTRY_TEMPLATE.md 原子发布报告与 HEAD sidecar
```

最近报告存在时，下一次运行优先使用其 sidecar 中每个来源的 Git HEAD，能够覆盖
“旧 commit 被晚推送”的情况；sidecar 缺失时才按报告文件名时间推导 baseline。
首次运行没有报告，默认回看 7 天，可用 `--lookback-days` 修改。

## 人工启动后台任务

```bash
cd /data3/paper_analysis

node scripts/paper_catch.ts doctor
node scripts/paper_catch.ts start
node scripts/paper_catch.ts status
```

`start` 立即返回 PID 和日志路径。后台 run 使用全局文件锁；重复启动时只有一个进程
能继续。若某个 Codex batch 超时、运行失败或输出不满足合同，本次不发布报告，run
进入 `PAUSED`。下次人工执行 `start` 或前台 `run` 会从首个未完成 batch 恢复，已
完成 Result 不会重复调用。

## 前台运行与仅扫描

```bash
# 前台运行，便于观察 Codex stderr
node scripts/paper_catch.ts run

# 只抓 Git、统计更新并冻结候选 batch，不调用 Codex、不发布报告
node scripts/paper_catch.ts scan
```

常用参数：

```bash
node scripts/paper_catch.ts start \
  --batch-size 20 \
  --lookback-days 7 \
  --codex-timeout-ms 900000
```

默认每个 batch 最多尝试两个 fresh 会话，并启用 live web search 用于核查候选语义和
开源链接。使用 `--no-search` 可关闭联网核查；`--model` 可覆盖模型。

## 筛选后端：Claude CLI（默认）或 Codex CLI

```bash
# 默认：claude -p，模型 claude-sonnet-5，结构化输出直接按 batch_result.schema.json 校验
node scripts/paper_catch.ts run
node scripts/paper_catch.ts run --model claude-opus-5

# 切回 Codex（读 ~/.codex/config.toml 的默认模型，或 --model 覆盖）
node scripts/paper_catch.ts run --provider codex

# 用环境变量做全局默认（cron / shell profile 里切换后端，不改命令）
export PAPER_CATCH_PROVIDER=codex        # 或 claude
export PAPER_CATCH_MODEL=claude-sonnet-5
export PAPER_CATCH_CLAUDE_BIN=/path/to/claude
export PAPER_CATCH_CODEX_BIN=/path/to/codex
```

`node scripts/paper_catch.ts doctor` 会打印当前生效的 provider、模型、二进制和 `PAPER_CATCH_*`
环境变量。CLI 参数优先于环境变量。

各后端的细粒度开关（都可通过 `paper_catch_backfill.ts` 透传）：

| 开关 | 作用 | 默认 |
|---|---|---|
| `--claude-effort LEVEL` | `claude --effort` | CLI 默认（或 `$PAPER_CATCH_CLAUDE_EFFORT`） |
| `--claude-max-budget-usd N` | `claude --max-budget-usd` | 不限（或 `$PAPER_CATCH_CLAUDE_MAX_BUDGET_USD`） |
| `--claude-tools LIST` | 覆盖 `--tools/--allowedTools` | `Read,WebSearch,WebFetch`；`--no-search` 时 `Read` |
| `--claude-setting-sources S` | `claude --setting-sources` | `user` |
| `--claude-arg ARG`（可重复） | 追加原始参数 | 无 |
| `--codex-reasoning-effort L` | `model_reasoning_effort` | `high` |
| `--codex-arg ARG`（可重复） | 在 `exec` 前插入原始参数 | 无 |

Claude 路径的会话参数固定为：`--output-format json --json-schema <合同>`、只开放
`Read,WebSearch,WebFetch`（`--no-search` 时仅 `Read`）、`--add-dir` 限定到本 run 的
batch 与 inputs 目录、`--strict-mcp-config`、`--no-session-persistence`、
`--setting-sources user`。stdout 的 JSON envelope 原样保存在 `provider_raw.jsonl`，
其 `structured_output` 提取为 `output.json` 后走与 Codex 相同的候选全覆盖校验。
Codex 路径保持 sandbox `read-only`、approval `never`、`--output-schema` 不变。
`--codex-timeout-ms` 对两种后端都生效。

## 历史回填与结果合并

新增来源（或为已有来源加新的 README `#fragment`）时，常规 run 只会从最近报告时间
起算增量。要一次性补齐更早的条目，用回填脚本在独立目录跑一遍，再与常规报告合并：

```bash
# 1. 只含该来源、回看到 2024-01-01 的独立 run（scan 可先预览候选）
node scripts/paper_catch_backfill.ts scan --url "https://github.com/AmberLJC/LLMSys-PaperList#llm-for-systems" --since 2024-01-01
node scripts/paper_catch_backfill.ts run  --url "https://github.com/AmberLJC/LLMSys-PaperList#llm-for-systems" --since 2024-01-01

# 2. 常规增量 run
node scripts/paper_catch.ts run

# 3. 合并：第一个目录为主，重复标题保留主目录的判断
node scripts/paper_catch_merge.ts paper_catch paper_catch/backfill_<sourceId>
```

回填目录默认是 `paper_catch/backfill_<sourceId>/`，内含自动生成的 `config.md`、复制的
`PAPER_ENTRY_TEMPLATE.md` 以及自己的 `.state/`、`.runs/`；主 `paper_catch/.state` 的
各来源 HEAD sidecar 不受影响。回填 run 同样支持 PAUSED 后重复执行恢复。合并报告写为
`<主目录>/<主 run id>_merged.md`，文件名不匹配 `YYYYMMDD_HHMMSS.md`，不会被当作
下一次 run 的 baseline。

## 固定条目模板

人类可编辑的展示合同位于：

```text
/data3/paper_analysis/paper_catch/PAPER_ENTRY_TEMPLATE.md
```

新 run 会冻结该模板到 `.runs/<run-id>/inputs/`。每个 Batch Task 都引用这份冻结
模板；Codex 输出字段与模板逐项对应，最终 Markdown 只由 Script 渲染。

## 审计与恢复目录

```text
paper_catch/
├── YYYYMMDD_HHMMSS.md
├── PAPER_ENTRY_TEMPLATE.md
├── .state/
│   ├── cache/                 # bare Git mirrors
│   ├── reports/               # report → source HEAD sidecar
│   ├── active_run.json
│   └── run.lock
├── .runs/<run-id>/
│   ├── run.json
│   ├── events.jsonl
│   ├── inputs/                # frozen config/template/schema
│   ├── sources/source_snapshots.json
│   ├── candidates.json
│   ├── batches/<batch-id>/
│   │   ├── task.json
│   │   ├── attempt-NN/{prompt,provider_raw,stderr,output,attempt}.json[l]
│   │   └── result.json
│   ├── aggregate.json
│   └── final_report.md
└── logs/
```

最终自检：

```bash
node scripts/paper_catch.ts validate
node --test scripts/paper_catch/tests/*.test.ts
```
