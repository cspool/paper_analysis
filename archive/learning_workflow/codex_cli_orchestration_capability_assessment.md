# Codex CLI 持久 Agent 与脚本编排能力结论

> 归档状态：旧版持久 Agent 方案的支撑性能力评估，仅供设计追溯。

> 调研日期：2026-07-27  
> 本机版本：`codex-cli 0.144.1`  
> 文档性质：能力审计与实测记录，不修改任何现有脚本或 skill

## 1. 核心结论

Codex CLI 可以实现与 Claude CLI `--input-format stream-json` 相同类别的“持续输入、多轮持久 Agent”语义，但入口不是 `codex exec`，而是：

```text
codex app-server --listen stdio://
```

需要准确区分：

1. `codex exec`
   - 一次读取一个 prompt；
   - `--json` 是流式输出，不是持续流式输入；
   - 可通过 `codex exec resume <thread-id>` 在新进程中继续历史 thread；
   - 适合单次任务、流水线步骤和短暂 Agent。
2. `codex app-server`
   - stdin/stdout 保持打开；
   - 使用双向 JSONL/JSON-RPC 消息；
   - 支持一个进程内创建、恢复和驱动多个 thread；
   - 支持连续 `turn/start`、运行中的 `turn/steer`、`turn/interrupt` 和增量事件；
   - 最接近现有 `idea_review_orchestrator.ts` 所用的 Claude stream-json 会话模型。
3. Codex SDK
   - TypeScript SDK 和 beta Python SDK 对 thread、turn 和 resume 作了更高层封装；
   - 官方建议普通自动化优先使用 SDK；
   - 本项目需要直接控制 Marker、LOOP、原始流、工具事件和多 thread 路由，因此新实现优先使用 App Server 的显式协议。

因此：

> Codex 支持 stream 风格的持久输入，但不能把现有命令中的 `claude` 字符串直接替换成 `codex`。命令参数、输入 envelope、输出事件和 Session 生命周期协议均不同，必须增加 Codex 专用 runtime。

## 2. App Server 的持久会话语义

最小生命周期为：

```text
启动 codex app-server
  → initialize
  → initialized
  → thread/start
  → turn/start
  → item/... 增量通知
  → turn/completed
  → turn/start（同一 thread）
  → ...
```

进程重启后的恢复流程为：

```text
启动新的 codex app-server
  → initialize
  → initialized
  → thread/resume(threadId)
  → turn/start
```

App Server 的基本对象是：

```text
Thread：一个 Agent 的持久对话
Turn：脚本发送给该 Agent 的一次输入及其执行
Item：消息、推理、工具调用、命令、文件修改等原子事件
```

对本项目而言，正确映射是：

```text
Agent 身份 = Codex thread
Agent 的一次推进 = Codex turn
Agent runtime = 一个由脚本管理的 app-server 进程
Workflow 状态 = 本地 canonical state，不是 thread 记忆
```

## 3. App Server 可用于编排的能力

| 能力 | 接口 | 对新工作流的用途 |
|---|---|---|
| 创建持久 Agent | `thread/start` | 创建 Stage Controller、Direction Planner、Reviewer |
| 恢复 Agent | `thread/resume` | 脚本重启后恢复持久角色 |
| 创建短暂 Agent | `thread/start(ephemeral=true)` | 单次 Evidence/Curator Worker |
| 开始一轮 | `turn/start` | 输入当前状态、上次输出和新任务结果 |
| 运行中追加信息 | `turn/steer` | 只用于明确需要中途补充的少数场景 |
| 中止超时轮次 | `turn/interrupt` | 超时、预算或脚本取消 |
| 流式文本 | `item/agentMessage/delta` | 实时日志和超时活动检测 |
| 完成判定 | `turn/completed` | 获取 completed/interrupted/failed |
| 分支历史 | `thread/fork` | 仅用于显式对照实验，不作为默认流程 |
| 显式 skill 输入 | `type=skill` 输入项 | 确保一个 thread 只执行指定 role skill |
| 每轮模型覆盖 | `turn/start.model` | 固定 `gpt-5.6-sol` |
| 每轮推理强度 | `turn/start.effort` | 按 Agent 类型选择 effort |
| 输出约束 | `turn/start.outputSchema` | 可选；本工作流仍以 Marker/LOOP 控制流为主 |
| 沙箱与审批 | `sandboxPolicy`、`approvalPolicy` | 只读 Evidence 和无工具 Controller 的隔离基础 |
| 模型能力探测 | `model/list` | 启动前核验模型及 effort |
| 协议 schema 生成 | `generate-ts` / `generate-json-schema` | 将 App Server 协议与 CLI 版本绑定 |

官方文档：

- [Codex App Server](https://developers.openai.com/codex/app-server/)
- [Codex SDK](https://developers.openai.com/codex/sdk/)
- [Codex 非交互模式](https://developers.openai.com/codex/noninteractive/)
- [Codex CLI Reference](https://developers.openai.com/codex/cli/reference/)

## 4. 本机实测

### 4.1 环境

```text
Codex CLI: codex-cli 0.144.1
认证状态: Logged in using ChatGPT
```

此次测试使用现有 ChatGPT/Codex 登录。它会消耗账户的 Codex 用量，但 CLI JSONL 没有给出可核验的美元费用，因此不能把测试描述为已确认的 Platform API Key 账单。

### 4.2 `codex exec resume` 跨进程记忆

测试过程：

1. 首个 `codex exec --json` turn 要求记忆随机标记；
2. 保存 `thread.started.thread_id`；
3. 启动新进程执行 `codex exec resume <thread-id>`；
4. 第二个进程正确返回首轮标记。

结果：

```text
跨进程 resume：成功
首轮 usage：input 20546，output 18
续接轮 usage：input 41142，cached input 20224，output 28
```

这证明 `exec resume` 可以实现语义持久化，但每轮都会启动新进程，不是持续 stdin。

### 4.3 App Server 同进程持续输入

测试过程：

1. 启动一个 `codex app-server --listen stdio://`；
2. 完成 initialize handshake；
3. `thread/start` 创建一个 thread；
4. 第一个 `turn/start` 要求记忆随机标记；
5. 第二个 `turn/start` 在同一 thread 中询问该标记。

结果：

```text
同一 App Server 进程：成功
同一 thread 连续两个 turn：成功
第二轮恢复首轮语义：成功
```

这证明 App Server 可以承担持久 Controller、Planner 和 Reviewer 的 transport。

### 4.4 文件产物与完成 Marker

在 `workspace-write` 模式下执行了一个临时测试，Codex 成功写入指定文件并保留完成 Marker。这证明旧 scheduler 使用的“产物文件 + DONE Marker”模式在技术上可迁移，但新版工作流应由脚本写 canonical 数据，Agent 只返回协议结果。

### 4.5 模型目录探测

当前 Codex 官方手册明确列出：

```text
gpt-5.6-sol
gpt-5.6-terra
gpt-5.6-luna
```

并说明 CLI reasoning effort 至少包括：

```text
low | medium | high | xhigh
```

本轮尝试通过 App Server `model/list` 读取当前账户的实际模型目录时，本地 Codex state DB 正处于 backfill，启动等待超时。因此：

- `gpt-5.6-sol` 是当前官方文档中的有效模型标识；
- 当前账户是否开放全部目标 effort，必须由新脚本的 `doctor` 在正式付费调用前再次核验；
- 不允许在探测失败时静默换模型或降低 effort。

这个失败也暴露出一个运行约束：正式编排器应长期复用一个 App Server 进程，启动时处理 state DB migration/backfill 等待，而不是为每个 Agent 反复启动 App Server。

## 5. 对现有三个 Claude 脚本的兼容判断

### 5.1 `scripts/learning_scheduler.ts`

现状：

```text
claude -p
  --output-format stream-json
  --permission-mode acceptEdits
  --add-dir ...
```

它的大多数任务是一次性 Agent，主要通过输出文件和 DONE Marker 判断完成。

判断：

- 能用 `codex exec --json` 或 App Server ephemeral thread 实现；
- Claude 参数不能原样复用；
- 需要改写事件解析和权限参数；
- skill 正文目前被嵌入 prompt，这部分语义容易迁移；
- Answer skill 依赖 Obsidian MCP，Codex runtime 必须启用同一只读 MCP。

兼容度：高，但不是命令行直接替换。

### 5.2 `scripts/run_all_papers.py`

现状：

- 每篇论文一次 Claude 子进程；
- parser 识别 Claude 的 `system`、`assistant`、`stream_event`、`result`；
- prompt 通过名称调用 `.claude/skills`。

判断：

- 可使用 `codex exec --json`、Python Codex SDK 或 ephemeral App Server thread；
- 必须把 parser 改成 Codex 的 `thread.started`、`item.*`、`turn.completed`；
- `.claude/skills` 不会自动成为 Codex skill；
- skill 名称也存在 `paper-knowledge` 与 `paper-knowledge-base` 的差异；
- 需要显式传入 Codex skill，不能只替换二进制。

兼容度：中等。

### 5.3 `scripts/idea_review_orchestrator.ts`

现状：

- 维护两个持续 Claude 子进程；
- 通过 stdin 反复写入 Claude stream-json user message；
- QA/AA 各自持有 Session；
- 脚本解析 Marker、LOOP、result，并进行路由、repair、checkpoint。

判断：

- `codex exec` 不能直接替代这个持续 stdin 协议；
- `codex app-server` 可以在架构上完整表达；
- QA、AA 可分别映射成两个 thread；
- Claude `session-id/resume` 映射为 Codex `thread/start/thread/resume`；
- Claude `result` 映射为 `turn/completed`；
- 现有 Marker/LOOP 和业务状态机可以保留其思想；
- transport、事件解析、权限配置和 thread ID 管理需要全新实现。

兼容度：语义能力高，直接命令兼容度低。

## 6. 不能忽略的限制

### 6.1 不是 Claude wire protocol 的 drop-in replacement

以下内容不兼容：

- CLI 参数；
- stdin message envelope；
- stdout event type；
- Session ID 创建方式；
- tool allow/disallow 参数；
- skill 搜索路径和显式调用方式；
- usage/cost telemetry。

应建立 provider/runtime adapter，不应在旧脚本中散布条件分支。

### 6.2 App Server 仍标记为 experimental

正式实现必须：

1. 固定最低/已验证的 Codex CLI 版本；
2. 在 `doctor` 中检查实际版本；
3. 使用当前二进制生成 TypeScript/JSON schema；
4. 将生成 schema 的版本和哈希写入 run config；
5. schema 或版本变化时拒绝 resume，先完成兼容迁移。

### 6.3 Codex 没有 Claude `--tools ""` 的直接等价参数

可采用：

- `approvalPolicy=never`；
- `sandboxPolicy=readOnly`；
- 空的 Agent cwd；
- 对无工具角色禁用 MCP、Web 和 dynamic tools；
- 对事件流使用严格 allowlist；
- 一旦无工具角色产生 command/MCP/Web/file/subagent 事件，拒收该 turn。

但“拒收结果”不等于“模型从未看见文件”。若必须保证盲评角色物理上无法接触知识库，还需要独立进程级沙箱或容器，只挂载空工作目录和必要状态。

### 6.4 禁止使用 Ultra

当前设计明确禁止 Agent 启动或管理其他 Agent。`ultra` 可能启用主动 subagent 行为，因此新工作流应：

```text
只使用 low / medium / high / xhigh
不使用 ultra
不启用 collaborationMode
拒收任何 subagent/collaboration tool event
```

### 6.5 Session 记忆不是 canonical state

即使 thread 可以持久化，每轮仍必须由脚本回注：

```text
当前规范状态
上次规范化输出
本轮新增结果
本次允许执行的任务块
```

thread 历史只用于语义连续性。恢复、审计和最终输出必须只依赖本地 canonical state。

### 6.6 固定上下文开销需要控制

最小实测的首轮输入已达到约 20.5k tokens，并出现全局 skill 描述被截断的提示。正式实现需要：

- 显式传入唯一 role skill；
- 限制每个 thread 可见的无关 skill；
- 每个 Agent 只承担一个简单角色；
- 记录每 turn usage；
- 在到达配置阈值前轮换 thread，并从 canonical checkpoint 重建；
- 禁止让一个持久 Agent 跨越整个 Stage 1 和全部 Stage 2。

## 7. 对新版 Learning Workflow 的最终选型

新版不修改或 import 现有脚本/skill，而采用并行实现：

```text
TypeScript deterministic orchestrator
  → 一个长期运行的 codex app-server
  → 一个 Agent 对应一个 Codex thread
  → 一个 Skill 对应一个 Agent role
  → 持久角色使用 persisted thread
  → 短暂角色使用 ephemeral thread
  → gpt-5.6-sol + role-specific effort
  → Marker/LOOP 控制流
  → JSON 只承载语义 payload
  → canonical JSON/JSONL 由脚本校验后写入
```

`codex exec` 仅用于：

- 安装/认证诊断；
- 简单 smoke test；
- App Server 不可用时的明确失败诊断；
- 不作为新版持久 Agent 的主 runtime，也不作为静默 fallback。
