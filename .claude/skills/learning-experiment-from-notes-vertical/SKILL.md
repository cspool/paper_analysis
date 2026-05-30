---
name: learning-experiment-from-notes-vertical
description: Vertical Summary Agent —— 接收所有层的 horizon summary 文件，对所有层次分类后的结果进行垂向梳理输出（模型负载→Serving→编译→Kernel 调度→硬件架构→芯片设计的不同组合下，负载定义到后端执行的全过程中，每层涉及的方法、实现和对应实验环境），输出 summary.md 后结束。由 scheduler.ts Phase 4 启动。
---

# Learning Experiment from Notes — Vertical Summary Agent

只读。用中文回答。不要人为缩短。每个结构化示例后须跟随「注解」节。

本 skill 是 **Vertical Summary Agent**，负责跨层垂向梳理。输入来自各层 Horizon Summary Agent 的输出。**不做单层分类。**

## 核心目标

对 6 层分类后的结果进行垂向梳理输出：在「模型负载 → Serving → 编译 → Kernel 调度 → 硬件架构 → 芯片设计」的不同组合下，从负载定义到后端执行的全过程中，每层涉及的方法、实现和对应实验环境是什么？

## 输入

scheduler.ts 在 prompt 中传递：
- 所有层的 horizon summary 文件路径列表
- 输出文件路径（`summary.md`）
- 用户输入要素（模型负载、后端平台、请求模式、计算场景、侧重标签）
- 侧重配置

## Workflow

### Step 1: 读取所有 horizon summary

逐文件读取所有 `<lid>_horizon_summary.md`，提取每层的分类方法列表和关系标注。

### Step 2: 识别垂向组合

从完成的层中识别可串联的垂向全栈组合：

```
组合 C-<n>: <具体模型负载> + <Serving框架> + <编译框架> + <Kernel方法> + <后端平台>
```

**识别原则**：
1. 选择各层笔记证据最充分的方法串联
2. 优先选兼容性明确的方法链（如 Triton kernel + Triton 编译链 + NVIDIA GPU 硬件）
3. 若某层笔记证据不充分，标记「该层缺失，以下为推断」
4. 覆盖多个后端平台组合（NVIDIA GPU、Ascend NPU、SN40L/MITA 等加速器）

### Step 3: 逐组合垂向全栈梳理

对每条垂向组合，输出从 **负载定义 → 后端执行** 的完整过程：

```markdown
## 垂向组合 C-<n>: <描述>

### 组合定义
- **模型负载**: <具体模型类型，如 Mixtral 8x7B MoE>
- **Serving 框架**: <如 SGLang + MuxWise>
- **编译框架**: <如 Triton + torch.compile>
- **Kernel 方法**: <如 FlashFuser inter-core fusion>
- **后端平台**: <如 H100 + LRM-GPU chiplet>

### 全栈执行路径
flowchart TD
    A["L1: <模型定义 + 推理计算流程>"] -->|"gate→expert selection"| B["L2: <Serving调度分解>"]
    B -->|"FX Graph / IR 导出"| C["L3: <编译融合和优化>"]
    C -->|"Triton IR → PTX/SASS"| D["L4: <Kernel 指令编排>"]
    D -->|"指令流 + TMA 访存"| E["L5: <硬件数据流 + 控制模块>"]
    E -->|"CoWoS / 芯片互联"| F["L6: <芯片级设计影响>"]

### 逐层：方法、实现、实验环境

#### L1: 算法 Pipeline
- **方法**: <从 horizon summary 提取的方法描述，保持伪代码/计算过程的具体程度>
- **实现**: <框架、工具链>
- **实验环境**: <硬件平台、benchmark、关键指标>
- **来源**: <horizon summary 文件>

#### L2: Serving 调度
- **方法**: <调度策略的框架运行模拟例子>
- **实现**: <Serving 框架>
- **实验环境**: ...
- **来源**: ...

#### L3: 编译框架
- **方法**: <编译流程模拟，IR 转换链>
- **实现**: <编译框架>
- **实验环境**: ...
- **来源**: ...

#### L4: Kernel 调度
- **方法**: <kernel 伪代码 + 指令 pipeline 编排>
- **实现**: <Kernel 库/框架>
- **实验环境**: ...
- **来源**: ...

#### L5: 硬件架构
- **方法**: <数据流设计 + 控制模块功能>
- **实现**: <硬件平台>
- **实验环境**: ...
- **来源**: ...

#### L6: 芯片设计
- **方法**: <芯片拓扑设计 + 评估数据>
- **实现**: <工艺/互联>
- **实验环境**: ...
- **来源**: ...

### 端到端数据流
一个 token/tensor 从输入到输出的完整路径：
1. [L1] hidden_states[t] [D] → Gate Linear → TopK(expert 0, expert 1)
2. [L2] MuxWise dispatcher 将 expert 0 分配到 SM[0..15], expert 1 到 SM[16..31]
3. [L3] Triton fusion: FC1+GeLU+FC2 → single fused kernel
4. [L4] TMA async load tile → Tensor Core MMA → inter-core buffer 直传
5. [L5] H100 warp scheduler latency hiding + TMA compute overlap
6. [L6] Chiplet 间 sync-val directory 追踪跨 die 同步

### 方法和实验环境对照表
| 层次 | 方法 | 实现 | 硬件平台 | Benchmark | 关键指标 | Vault 来源 |
|------|------|------|----------|-----------|----------|-----------|
| L1 | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |

### 组合不确定性
<证据链缺口>
```

### Step 4: 综合报告

#### 4.1 汇总
2-4 句：模型负载类型、发现的核心方法类别（MoE expert 并发、编译融合、chiplet 同步）、关键空白（如 wavelet-Diffusion 笔记覆盖低、NPU 编译链空白）。

#### 4.2 全栈关系图（Mermaid）
```mermaid
flowchart TD
    subgraph "L1 算法Pipeline"
        A1["<方法1>"] --> A2["<方法2>"]
    end
    subgraph "L2 Serving调度"
        B1["<方法1>"]
    end
    ...
    A1 -->|"gate→expert"| B1
    B1 -->|"FX Graph"| C1
```

**注解**: 箭头含义、方法间兼容性、数据格式转换。

#### 4.3 关键方法总结表

| 层次 | 方法数 | 笔记覆盖度 | 核心方法 | 主要空白 |
|------|--------|-----------|----------|----------|
| L1 | ... | 高/中/低 | ... | ... |
| ... | ... | ... | ... | ... |

#### 4.4 推荐学习路径（3-5 条）

```markdown
### P1: <路径名>
- **目标**: <学习目标>
- **涉及层次**: Lx, Ly, ...
- **推荐笔记**: <vault-path> (score: ...)
- **Web 补充**: <url>
```

#### 4.5 完整证据索引

| 层次 | 问题 ID | 方法 | Vault Path | Score |
|------|---------|------|------------|-------|
| ... | ... | ... | ... | ... |

### Step 5: 输出

写入 `summary.md`，末尾加 `[VERTICAL_SUMMARY_DONE]`。

## Mermaid 语法安全规则

1. 始终双引号节点文本和边标签
2. 禁止：`^`→`#Hat;`，`×`→`x`，`&`→`&amp;`，`<`/`>`→`&lt;`/`&gt;`
3. 节点 ID 仅字母数字，多行用 `<br/>`
4. 子图标题始终引号

## 公式指南

- 块级 `$$...$$` 单独行，行内 `$...$`
- ASCII 变量名，`\mathrm{Label}` 或 `\operatorname{name}`

## 质量自检

- [ ] 所有 horizon summary 文件完整读取
- [ ] 识别 ≥1 条垂向全栈组合
- [ ] 每条组合覆盖 L1→L6 的全过程（缺失层标记）
- [ ] 每层在组合中的描述保持原答案的具体程度（伪代码/数据流/实现框架/实验环境）
- [ ] 端到端数据流完整（token 输入到输出）
- [ ] 方法和实验环境对照表完整
- [ ] 全栈关系图覆盖所有已完成层
- [ ] 3-5 条推荐学习路径
- [ ] 完整证据索引
- [ ] Mermaid 语法检查
- [ ] `[VERTICAL_SUMMARY_DONE]` 在末尾
- [ ] 中文回答
