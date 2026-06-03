# HATB mPLUG-Owl3 对话记录

- 导出时间：2026-06-03 17:33 CST
- 来源：当前 Codex session 可用对话上下文
- 保存路径：/data3/paper_analysis/human_notes/HATB_mPLUG-Owl3_对话记录.md
- 范围：仅用户输入与 Codex 最终输出
- 说明：运行环境上下文、工具调用、中间状态更新未保存；本次导出完成后的 Codex 回复不包含在本次记录内。

## 对话记录

### 001 User
# Context from my IDE setup:

## Open tabs:
- L1_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L1_horizon_summary.md

## My request for Codex:
教我:'多算子并发与调度	HATB (mPLUG-Owl3)	Self-Attn ‖ Cross-Attn 并行 + Adaptive Gating 融合；共享 Q 投影, K/V 独立	4/28 layers optimal; cross-attn 延迟隐藏'

### 002 Codex Final
我按你的本地笔记和 mPLUG-Owl3 论文笔记来讲。核心一句话：

**HATB = 在少数 LLM 层里，把文本 self-attention 和视觉 cross-attention 设计成两个并行分支：共用文本 Query，各自用文本/视觉 K,V，最后用 Adaptive Gating 把两路结果融合。**

**1. 它解决什么问题**
传统多模态模型常把视觉 token 直接拼进文本序列：

```text
[text tokens + image tokens] -> LLM self-attention
```

问题是 attention 长度变长，计算和 KV cache 都膨胀，而且长图像序列/视频会很快吃掉上下文窗口。普通 Cross-Attn 虽然避免拼接，但通常是串行插入：

```text
Self-Attn -> Cross-Attn -> FFN
```

这样 cross-attn 的延迟直接加到路径上。

HATB 的思路是：

```text
             -> Self-Attn(text Q, text K/V)  ->
Text hidden -|                                  -> Adaptive Gate -> FFN
             -> Cross-Attn(text Q, image K/V) ->
```

**2. “共享 Q，K/V 独立”是什么意思**
在 self-attn 里：

$$
Q_t = H_t W_Q,\quad K_t = H_t W_K,\quad V_t = H_t W_V
$$

在 cross-attn 里，HATB 不重新造一套 query，而是复用文本 query：

$$
Q = Q_t
$$

但视觉侧有自己的 K/V：

$$
K_v,V_v = H_{img} W^{KV}_{img}
$$

所以 self-attn 负责“文本内部推理”，cross-attn 负责“文本问题去视觉特征里取信息”。共享 Q 的直觉是：**当前文本 token 想问什么，就用同一个 query 同时问文本上下文和视觉上下文。**

**3. Adaptive Gating 怎么融合**
论文里的融合大意是：

$$
g = \operatorname{Sigmoid}(W_{gate}^{T}\hat{H}^{i})
$$

$$
H_{fused}^{i} = \bar{H}^{i} \cdot g + \hat{H}^{i} \cdot (1-g)
$$

可以这样理解：

- $\hat{H}^{i}$：self-attn 分支输出，偏文本语义。
- $\bar{H}^{i}$：cross-attn 分支输出，带视觉补充信息。
- $g$：每个 token 自适应决定“要多少视觉信息”。

例如问“图中车是什么颜色？”时，`车`、`颜色`相关 token 的视觉门控应更高；纯语言连接词的视觉门控可以低一些。

**4. 为什么叫“cross-attn 延迟隐藏”**
如果串行做：

$$
T_{serial}=T_{self}+T_{cross}+T_{gate}
$$

HATB 并行做：

$$
T_{parallel}=\max(T_{self},T_{cross})+T_{gate}
$$

所以可隐藏的 cross-attn 延迟近似是：

$$
T_{hidden}=\min(T_{self},T_{cross})
$$

如果 cross-attn 比 self-attn 短，它几乎被完全藏住；如果 cross-attn 更长，只暴露多出来的尾巴。这就是你笔记里的“cross-attn 延迟隐藏”。

**5. 为什么只放 4 层，不是每层都放**
论文实验比较了不同 HATB 层数/位置：2 层、4 层、8 层。4 层策略 `[1, 9, 17, 25]` 整体最均衡；正文也提到对 Qwen2 采用均匀扩展少数层，类似 `[0, 9, 17, 25]`，这里主要是 0/1-based 编号差异。

不是越多越好，因为：

- 每加一层 HATB 都增加视觉 K/V projection、cross-attn、gate 的计算和参数。
- 视觉信息注入太密，会扰动预训练 LLM 的 hidden states。
- 训练数据规模不够时，8 层 HATB 反而泛化更差。
- 4 层均匀插入浅层/中层/深层，已经能让模型多次补充视觉信息。

所以 `4/28 layers optimal` 的准确理解是：**在约 28 层 LLM 里，只把 4 层替换成 HATB，获得了性能、参数、显存、延迟之间的较好平衡。**

**记忆版**
HATB 不是简单“多开一个 cross-attn”，而是一个很巧的并发 DAG：

```text
共享文本 Q
   |---------------- self-attn: Q x text K/V
   |---------------- cross-attn: Q x image K/V
                         |
                  adaptive gating
                         |
                        FFN
```

它的系统价值在于：把原本可能串行的跨模态融合，改造成可并行调度的两个 attention 分支；它的模型价值在于：不把海量视觉 token 塞进 LLM 主序列，而是让文本 token 按需从视觉特征里取信息。

依据笔记：  
[1-Introduction.md](/data3/paper_analysis/paper_secs/secs_video_image/mPLUG-Owl3__Towards_Long_Image-Sequence_Understanding_in_Multi-Modal_Large_Language_Models/1-Introduction.md)  
[4-Experiments.md](/data3/paper_analysis/paper_secs/secs_video_image/mPLUG-Owl3__Towards_Long_Image-Sequence_Understanding_in_Multi-Modal_Large_Language_Models/4-Experiments.md)  
[L1_horizon_summary.md](/data3/paper_analysis/learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L1_horizon_summary.md)

### 003 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L1_horizon_summary.md

## Open tabs:
- L1_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L1_horizon_summary.md

# Selected text:

## Selection 1
3. Adaptive Gating 怎么融合论文里的融合大意是：\[
g = \operatorname{Sigmoid}(W_{gate}^{T}\hat{H}^{i})
\]\[
H_{fused}^{i} = \bar{H}^{i} \cdot g + \hat{H}^{i} \cdot (1-g)
\]可以这样理解：$\hat{H}^{i}$：self-attn 分支输出，偏文本语义。$\bar{H}^{i}$：cross-attn 分支输出，带视觉补充信息。$g$：每个 token 自适应决定“要多少视觉信息”。例如问“图中车是什么颜色？”时，车、颜色相关 token 的视觉门控应更高；纯语言连接词的视觉门控可以低一些。

## My request for Codex:
详细点

### 004 Codex Final
更细一点看，Adaptive Gating 不是“把 self-attn 和 cross-attn 简单相加”，而是在每个文本 token 上学一个开关：**这个 token 当前更应该相信语言上下文，还是更应该从图像里补信息。**

**1. 先看 HATB 里的两条分支**
在第 $i$ 层，输入主要有两类 hidden states：

$$
H_{\text{text}}^i \in \mathbb{R}^{B \times L_t \times D}
$$

$$
H_{\text{img}} \in \mathbb{R}^{B \times L_v \times D}
$$

其中 $L_t$ 是文本 token 数，$L_v$ 是视觉 token 数，$D$ 是 hidden dim。

HATB 同时做两件事：

$$
\hat{H}^i = \operatorname{SelfAttn}(Q_{\text{text}}, K_{\text{text}}, V_{\text{text}})
$$

$$
\bar{H}^i = \operatorname{CrossAttn}(Q_{\text{text}}, K_{\text{img}}, V_{\text{img}})
$$

这里的关键是：**两个分支共享文本 Query**。

$$
Q_{\text{text}} = H_{\text{text}}^i W_Q
$$

self-attn 用文本自己的 K/V：

$$
K_{\text{text}}, V_{\text{text}} = H_{\text{text}}^i W_{\text{text}}^{KV}
$$

cross-attn 用图像的 K/V：

$$
K_{\text{img}}, V_{\text{img}} = H_{\text{img}} W_{\text{img}}^{KV}
$$

Annotations:

- $\hat{H}^i$：文本 self-attn 输出，表示“文本内部上下文理解之后的表示”。
- $\bar{H}^i$：视觉 cross-attn 输出，表示“当前文本 query 从图像 token 中取回来的视觉信息”。
- 共享 $Q_{\text{text}}$ 的含义是：同一个文本意图，同时去问文本上下文和视觉上下文。
- $K/V$ 独立是因为文本和图像来自不同模态，不能直接共用同一套 key/value 投影。

**2. Adaptive Gate 到底在算什么**
论文公式是：

$$
g = \operatorname{Sigmoid}(W_{\text{gate}}^T \hat{H}^i)
$$

$$
H_{\text{fused}}^i = \bar{H}^i \cdot g + \hat{H}^i \cdot (1-g)
$$

如果 $g$ 越大，融合结果越偏向 cross-attn，也就是越相信视觉补充；如果 $g$ 越小，结果越偏向 self-attn，也就是越保留语言模型原本的文本推理。

可以把每个 token 想成这样：

| Token | 语义 | 可能的 $g$ | 融合倾向 |
|---|---|---:|---|
| “图中” | 视觉指示 | 0.75 | 更多视觉 |
| “车” | 目标实体 | 0.85 | 更多视觉 |
| “是什么颜色” | 需要看图回答 | 0.90 | 更多视觉 |
| “请问” | 语言礼貌词 | 0.15 | 更多文本 |
| “的” | 功能词 | 0.05 | 几乎文本 |

所以对“图中车是什么颜色？”这个问题，模型不应该让所有 token 都等量吸收视觉信息。真正需要视觉证据的是“图中 / 车 / 颜色”这些 token。

**3. 为什么 gate 用 $\hat{H}^i$ 来算**
这是一个很重要的小设计。$\hat{H}^i$ 是 self-attn 后的文本表示，它已经知道当前 token 在句子里的语义角色。

也就是说，gate 不是从原始 token embedding 算的，而是从“语言上下文理解后的表示”算的：

```text
原始 token: “颜色”
经过 self-attn 后: “用户正在问图像中对象的颜色”
再算 gate: 这个 token 需要较多视觉信息
```

Annotations:

- 如果只用固定 scale，比如 $H = H_{\text{text}} + \alpha H_{\text{img}}$，那么所有 token 都用同一个视觉注入强度。
- Adaptive Gating 的优势是 token-adaptive：不同 token 有不同视觉需求。
- 它也保护 LLM：不让图像信息无脑冲进每个 hidden state，减少对预训练语言空间的扰动。

**4. 一个更具体的伪代码**
```python
def HATB(H_text, H_img):
    X_text = shared_layernorm(H_text)
    X_img = shared_layernorm(H_img)

    Q = X_text @ W_q

    K_text = X_text @ W_k_text
    V_text = X_text @ W_v_text

    K_img, V_img = X_img @ W_img_kv

    H_self = self_attention(Q, K_text, V_text)
    H_cross = cross_attention(Q, K_img, V_img)

    gate = sigmoid(H_self @ W_gate)

    H_fused = gate * H_cross + (1 - gate) * H_self

    return FFN(H_fused)
```

Annotations:

- `Q` 只算一次，然后被 self-attn 和 cross-attn 共用。
- `H_self` 对应公式里的 $\hat{H}^i$。
- `H_cross` 对应公式里的 $\bar{H}^i$。
- `gate` 通常可理解为 token-level gate；实现上也可以扩展成 channel-level gate。
- `H_fused` 才会进入 FFN 和下一层 Transformer。

**5. 从调度角度看它为什么能隐藏延迟**
普通串行 cross-attn 是：

```text
Self-Attn -> Cross-Attn -> Gate/Fuse -> FFN
```

HATB 是：

```text
          -> Self-Attn  -> Gate -----
Q shared -|                         -> Fuse -> FFN
          -> Cross-Attn ------------
```

所以关键路径从：

$$
T_{\text{serial}} = T_{\text{self}} + T_{\text{cross}} + T_{\text{fuse}}
$$

变成：

$$
T_{\text{HATB}} = \max(T_{\text{self}}, T_{\text{cross}}) + T_{\text{fuse}}
$$

Annotations:

- cross-attn 不再完全排在 self-attn 后面。
- 如果 cross-attn 比 self-attn 短，它几乎被完全隐藏。
- 如果 cross-attn 比 self-attn 长，只暴露多出来的尾部。
- gate 依赖 self-attn 输出，所以 self-attn 完成后可以先算 gate，同时等待 cross-attn 完成融合。

**6. 实验上 Adaptive Gating 有什么用**
mPLUG-Owl3 的消融里，基础 Hyper Attention 加上 Adaptive Gating 后，多项指标明显提升：

| 结构 | GQA | TextVQA | MvBench | VideoMME | NLVR2 | Mantis |
|---|---:|---:|---:|---:|---:|---:|
| 基础 HATB | 53.3 | 44.6 | 40.2 | 38.1 | 52.7 | 41.9 |
| + Adaptive Gating | 55.7 | 49.3 | 43.2 | 40.1 | 53.4 | 47.9 |

最直观的是 TextVQA 从 44.6 到 49.3，Mantis 从 41.9 到 47.9。说明 gate 确实帮助模型更有选择地吸收视觉信息，而不是把视觉分支当成固定噪声源加进去。

一句话收束：

**Adaptive Gating 的本质是：让文本 token 自己决定视觉信息的注入强度。HATB 先并行得到“文本理解”和“视觉补充”两份表示，再用语义相关的 gate 做动态混合。**

依据笔记：  
[1-Introduction.md](/data3/paper_analysis/paper_secs/secs_video_image/mPLUG-Owl3__Towards_Long_Image-Sequence_Understanding_in_Multi-Modal_Large_Language_Models/1-Introduction.md)  
[4-Experiments.md](/data3/paper_analysis/paper_secs/secs_video_image/mPLUG-Owl3__Towards_Long_Image-Sequence_Understanding_in_Multi-Modal_Large_Language_Models/4-Experiments.md)

### 005 User
# Context from my IDE setup:

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- L1_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L1_horizon_summary.md
- SKILL.md: .codex/skills/export-conversation-notes/SKILL.md
- Accelerating Sparse Transformer Inference on GPU (STOF).md: human_notes/Accelerating Sparse Transformer Inference on GPU (STOF).md
- Adaptive Draft Sequence Length_ Enhancing Speculative Decoding Throughput on PIM-Enabled Systems.md: human_notes/Adaptive Draft Sequence Length_ Enhancing Speculative Decoding Throughput on PIM-Enabled Systems.md

## My request for Codex:
[$export-conversation-notes](/data3/paper_analysis/.codex/skills/export-conversation-notes/SKILL.md) 

### 006 User
<skill>
<name>export-conversation-notes</name>
<path>/data3/paper_analysis/.codex/skills/export-conversation-notes/SKILL.md</path>
---
name: export-conversation-notes
description: Export the currently available Codex/chat session record into a target Markdown file, defaulting to human_notes/. Save only user inputs and Codex final outputs. Use when the user asks to save, archive, output, or record the current conversation, dialogue history, or analysis session in the paper_analysis workspace. If the target Markdown file already exists and is non-empty, append incrementally without modifying, deleting, reformatting, or replacing any existing content.
---

# Export Conversation Notes

## Overview

Save the current session's available user inputs and Codex final outputs to the requested target location. Default to `human_notes/` when no explicit directory is provided. Default to Chinese for headings and status text. Do not save process notes, status updates, tool calls, command outputs, hidden instructions, or intermediate work logs.

## Workflow

1. Identify the target path.
   - save under `human_notes/`.
   - If no filename can be identified, use `<session-name>.md` and <session-name> is the name of the current session.

2. Prepare the target file.
   - Work relative to the current workspace root.
   - Ensure the destination directory exists.
   - Sanitize only the filename portion by replacing filesystem-hostile characters such as `/`, `\`, `:`, `*`, `?`, `"`, `<`, `>`, and `|` with safe separators, then trim whitespace.
   - Save to the resolved target path.

3. Capture the available conversation record.
   - Output only user inputs and Codex final outputs in chronological order with speaker labels.
   - Include all available user messages exactly as written.
   - Include only Codex final responses that answer the user, not interim progress updates or tool-facing content.
   - Do not include tool calls, shell commands, command outputs, file edit logs, errors from tools, status updates, planning chatter, hidden system/developer/policy/runtime instructions, or intermediate reasoning.
   - Do not summarize, compress, paraphrase, normalize, or reorganize the saved user inputs and final outputs.
   - If earlier user inputs or final outputs are unavailable because context was compacted or not exposed to Codex, state this limitation briefly before the saved conversation.

4. Write the Markdown file.
   - If the file does not exist, create it with the new-file template.
   - If the file exists but is empty, write the new-file template.
   - If the file exists and is non-empty, enter incremental mode: append a new dated section only at the end of the file.
   - In incremental mode, never modify, delete, reorder, summarize, normalize, reformat, or replace any existing content, even if the existing note has typos, duplicate headings, stale metadata, or inconsistent formatting.
   - In incremental mode, use an append-only edit. With `apply_patch`, add only new lines after the existing final line.
   - Keep the saved record readable, but preserve the available user inputs and final outputs over brevity.
   - If the record is too long for one edit, append it in multiple consecutive chunks until all currently available user inputs and final outputs are saved.

## Markdown Template

For a new file, use this structure:

```md
# <session-or-paper-title>

- 导出时间：<YYYY-MM-DD HH:MM TZ>
- 来源：当前 Codex session 可用对话上下文
- 保存路径：<resolved-target-path>
- 范围：仅用户输入与 Codex 最终输出

## 对话记录

### 001 User
<用户消息原文>

### 002 Codex Final
<Codex 最终回复原文>
```

For appending to an existing non-empty file, add this block at the end of the file without changing earlier content:

```md
---

## 对话记录补充：<YYYY-MM-DD HH:MM TZ>

<continue the same chronological format, saving only user inputs and Codex final outputs>
```

## Completion Response

After saving, respond briefly with the output path and whether the file was created or appended. Mention any uncertainty about target inference or incomplete available conversation context.

</skill>
