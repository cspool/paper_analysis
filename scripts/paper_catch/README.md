# Paper Catch Git/Codex Batch Loop

该流程由人工按需要定时调用，但实际抓取和 Codex 筛选可在后台执行。它读取：

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
  → 每个 batch 启动一个 fresh `codex exec` 会话
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

默认每个 batch 最多尝试两个 fresh ephemeral Codex 会话，sandbox 固定为
`read-only`、approval 固定为 `never`，并启用 live web search 用于核查候选语义和
开源链接。使用 `--no-search` 可关闭联网核查；`--model` 可覆盖当前 Codex 默认模型。

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
