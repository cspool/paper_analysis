# MoE并行概念解释对话

- 导出时间：2026-06-03 17:28 CST
- 来源：当前 Codex session 可用对话上下文
- 保存路径：human_notes/MoE并行概念解释对话.md
- 范围：仅用户输入与 Codex 最终输出

## 对话记录

### 001 User
````text
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Active selection of the file:
### 方法3: DeepSpeed-MoE / DeepSpeed-FastGen — 混合并行 + 分层 All-to-All

**笔记证据**: `paper_secs/secs_moe/HAP.../I.-INTRODUCTION.md` (score: 3346.2); `knowledge_notes/kernel知识笔记/All-to-All Communication in MoE.md` (score: 461.4)

**方法细节**（L1 粒度 —— 四维并行 MoE 推理 Pipeline）:

```
DeepSpeed-MoE 多维并行推理 (8 GPUs, DP=2, TP=2, EP=2):
=========================================================
每 GPU 持有: 1/TP=1/2 attention 权重 + 1/EP=1/2 expert 权重

Layer ℓ MoE Forward:
  Input: h [batch_size, seq_len, hidden_dim] 分布在 DP group

  # Stage 1: Attention TP (重量级 AllReduce)
  h_local = Attention_TP(LayerNorm(h))        # Q/K/V/O 沿 hidden dim 切分
  h = AllReduce(h_local)                       # 聚合各 TP rank 的部分结果
    // 通信量: 2×batch×seqlen×hidden×sizeof(FP16)
    // Prefill (长序列): 通信瓶颈, 占总延迟 20-40%

  # Stage 2: MoE Gating (每个 DP rank 独立)
  gate_logits = Gating(h)                      # [B×S, E]
  selected = TopK(softmax(gate_logits), K=2)   # 每 token 选 top-2

  # Stage 3: 分层 All-to-All Dispatch
  // Phase 3a: Intra-node (NVLink 600 GB/s)
  for src_gpu in node_gpus:
      gather tokens → NCCL P2P Send to target GPU within node
  // Phase 3b: Inter-node (InfiniBand 200 Gbps)
  regroup by target node → NCCL P2P Send across nodes
  // 或 TensorRT-LLM 方式: NVLink one-sided alltoAll (替代 AllGather+ReduceScatter)

  # Stage 4: Expert FFN (EP)
  received_tokens = tokens_for_my_expert       # [n_e, hidden_dim]
  expert_out = ExpertFFN(received_tokens)      # GateProj→SiLU→UpProj→DownProj
    // 每 GPU 仅计算 1/EP 的 expert, 参数驻留本地 HBM

  # Stage 5: All-to-All Combine (反向通信)
  combined = AllToAll_Combine(expert_out)       # 通信量同 Dispatch

  # Stage 6: Residual
  output = h + combined
```

**注解**:
- **通信 vs 计算权衡**: EP=8 时每 GPU 通信量 ≈ 1/8 expert 参数加载，但 All-to-All 通信仍占 30-50% 延迟。笔记 "HAP 论文关键洞察：长 context prefill 时通信是瓶颈，EP 通信量低于 TP 的 AllReduce；decode 时通信量小但 EP 负载不均衡"
- **分层 All-to-All 原理**: Intra-node NVLink (600+ GB/s) 聚合 → Inter-node InfiniBand (25-50 GB/s per link) 传输，复杂度从 O(N²) 降至 O(N+G²)，G=每节点 GPU 数
- **DeepEP**: 字节跳动开源的非均匀 All-to-All 库，匹配 popular expert 链路高带宽需求，实现高吞吐低延迟

**实验环境** (HAP 论文):
- **硬件**: 4×A6000 (PCIe Gen4 ≤32 GB/s), 4×A100 (NVLink 600 GB/s), 8×V100 (PCIe), 8×H100
- **模型**: Mixtral-8x7B (46.7B, 8 experts), Qwen1.5-MoE-A2.7B (14.3B, 60 experts), Qwen2-57B-A14B (57.4B, 64 experts)
- **4 种推理场景**: 短/长 context (256/4096) × 短/长 generation (64/2048)
- **关键指标**: end-to-end latency speedup vs TP baseline: A6000 max 1.68×, A100 max 1.77×, V100 1.57×
- **硬件适配**: PCIe 低带宽下选 DP+EP (避免 AllReduce)，NVLink 高带宽下可选 TP 保持计算效率

---
## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- review_on_learning_outputs_20260529_185938.md: human_notes/review_on_learning_outputs_20260529_185938.md
- Q1.2_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.2_L1_answer.md
- Q1.1_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.1_L1_answer.md
- I.-INTRODUCTION.md: paper_secs/secs_moe/HAP Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference/I.-INTRODUCTION.md

## My request for Codex:
'### 方法3: DeepSpeed-MoE / DeepSpeed-FastGen — 混合并行 + 分层 All-to-All

**笔记证据**: `paper_secs/secs_moe/HAP.../I.-INTRODUCTION.md` (score: 3346.2); `knowledge_notes/kernel知识笔记/All-to-All Communication in MoE.md` (score: 461.4)

**方法细节**（L1 粒度 —— 四维并行 MoE 推理 Pipeline）:

```
DeepSpeed-MoE 多维并行推理 (8 GPUs, DP=2, TP=2, EP=2):
=========================================================
每 GPU 持有: 1/TP=1/2 attention 权重 + 1/EP=1/2 expert 权重

Layer ℓ MoE Forward:
  Input: h [batch_size, seq_len, hidden_dim] 分布在 DP group

  # Stage 1: Attention TP (重量级 AllReduce)
  h_local = Attention_TP(LayerNorm(h))        # Q/K/V/O 沿 hidden dim 切分
  h = AllReduce(h_local)                       # 聚合各 TP rank 的部分结果
    // 通信量: 2×batch×seqlen×hidden×sizeof(FP16)
    // Prefill (长序列): 通信瓶颈, 占总延迟 20-40%

  # Stage 2: MoE Gating (每个 DP rank 独立)
  gate_logits = Gating(h)                      # [B×S, E]
  selected = TopK(softmax(gate_logits), K=2)   # 每 token 选 top-2

  # Stage 3: 分层 All-to-All Dispatch
  // Phase 3a: Intra-node (NVLink 600 GB/s)
  for src_gpu in node_gpus:
      gather tokens → NCCL P2P Send to target GPU within node
  // Phase 3b: Inter-node (InfiniBand 200 Gbps)
  regroup by target node → NCCL P2P Send across nodes
  // 或 TensorRT-LLM 方式: NVLink one-sided alltoAll (替代 AllGather+ReduceScatter)

  # Stage 4: Expert FFN (EP)
  received_tokens = tokens_for_my_expert       # [n_e, hidden_dim]
  expert_out = ExpertFFN(received_tokens)      # GateProj→SiLU→UpProj→DownProj
    // 每 GPU 仅计算 1/EP 的 expert, 参数驻留本地 HBM

  # Stage 5: All-to-All Combine (反向通信)
  combined = AllToAll_Combine(expert_out)       # 通信量同 Dispatch

  # Stage 6: Residual
  output = h + combined
```

**注解**:
- **通信 vs 计算权衡**: EP=8 时每 GPU 通信量 ≈ 1/8 expert 参数加载，但 All-to-All 通信仍占 30-50% 延迟。笔记 "HAP 论文关键洞察：长 context prefill 时通信是瓶颈，EP 通信量低于 TP 的 AllReduce；decode 时通信量小但 EP 负载不均衡"
- **分层 All-to-All 原理**: Intra-node NVLink (600+ GB/s) 聚合 → Inter-node InfiniBand (25-50 GB/s per link) 传输，复杂度从 O(N²) 降至 O(N+G²)，G=每节点 GPU 数
- **DeepEP**: 字节跳动开源的非均匀 All-to-All 库，匹配 popular expert 链路高带宽需求，实现高吞吐低延迟

**实验环境** (HAP 论文):
- **硬件**: 4×A6000 (PCIe Gen4 ≤32 GB/s), 4×A100 (NVLink 600 GB/s), 8×V100 (PCIe), 8×H100
- **模型**: Mixtral-8x7B (46.7B, 8 experts), Qwen1.5-MoE-A2.7B (14.3B, 60 experts), Qwen2-57B-A14B (57.4B, 64 experts)
- **4 种推理场景**: 短/长 context (256/4096) × 短/长 generation (64/2048)
- **关键指标**: end-to-end latency speedup vs TP baseline: A6000 max 1.68×, A100 max 1.77×, V100 1.57×
- **硬件适配**: PCIe 低带宽下选 DP+EP (避免 AllReduce)，NVLink 高带宽下可选 TP 保持计算效率

---',教我,并解释伪代码
````

### 002 Codex Final
````text
我先给你一句总图景：

这段伪代码在讲 **一个 MoE Transformer 层在多 GPU 上怎么跑**：Attention 部分用 **TP** 切权重并 AllReduce；MoE Expert 部分用 **EP** 把专家分散到不同 GPU，然后用 **All-to-All** 把 token 发给对应专家，算完再发回来。

本地笔记依据主要是：
- `paper_secs/secs_moe/HAP Hybrid Adaptive Parallelism.../I.-INTRODUCTION.md`
- `knowledge_notes/kernel知识笔记/All-to-All Communication in MoE.md`

**先懂 4 个词**
| 词 | 一句话解释 |
|---|---|
| `MoE` | 把普通 FFN 换成多个 expert，每个 token 只激活 Top-K 个 expert。 |
| `TP` | Tensor Parallelism，把一个大矩阵/Attention 权重切到多个 GPU 上算。 |
| `EP` | Expert Parallelism，把不同 experts 放到不同 GPU 上。 |
| `All-to-All` | 每个 GPU 都可能给每个 GPU 发送不同 token，用于 token 到 expert 的路由。 |

注意一个小修正：你笔记里写 “DeepEP 字节跳动开源”，但 HAP 引用里对应的是 `deepseek-ai/DeepEP`。这里理解为“面向 MoE 的高性能 expert-parallel 通信库”即可。

**整体流程**
你可以把一层 MoE 看成：

```text
输入 token hidden states
    ↓
Attention 部分：TP 并行计算，需要 AllReduce 聚合
    ↓
Router/Gate：决定每个 token 去哪几个 expert
    ↓
All-to-All Dispatch：把 token 发到持有目标 expert 的 GPU
    ↓
Expert FFN：每个 GPU 只计算自己持有的 experts
    ↓
All-to-All Combine：把 expert 输出发回 token 原来的 GPU
    ↓
Residual：加回主干 hidden state
```

**伪代码逐段解释**
```text
DeepSpeed-MoE 多维并行推理 (8 GPUs, DP=2, TP=2, EP=2)
```

意思是 8 张 GPU 被组织成多种并行维度。可以粗略理解为：

```text
DP=2：有 2 份并行副本，处理不同 batch/request
TP=2：Attention/线性层权重切成 2 份
EP=2：Experts 切成 2 份，每组 GPU 只放一部分 experts
```

所以：

```text
每 GPU 持有: 1/TP=1/2 attention 权重 + 1/EP=1/2 expert 权重
```

意思不是每张卡只干一半事情，而是：Attention 大矩阵被拆了，Expert 集合也被拆了。这样单卡显存压力下降，但通信变多。

**Stage 1: Attention TP**
```text
h_local = Attention_TP(LayerNorm(h))
h = AllReduce(h_local)
```

Attention 权重被切在多个 GPU 上，所以每个 GPU 只能算出一部分结果，叫 `h_local`。

例如 TP=2：

```text
GPU0 算 attention 的前半部分输出
GPU1 算 attention 的后半部分输出
然后 AllReduce / 聚合，得到完整 h
```

为什么这里通信重？

因为 prefill 阶段输入序列很长，例如 `seq_len=4096`，通信张量大小近似和下面成正比：

```text
batch_size × seq_len × hidden_dim × dtype_size
```

所以长上下文时，TP 的 AllReduce 很容易变成瓶颈。

**Stage 2: MoE Gating**
```text
gate_logits = Gating(h)
selected = TopK(softmax(gate_logits), K=2)
```

这一步是 router。它给每个 token 打分，判断 token 应该送到哪些 experts。

例子：

```text
token_17:
  expert_3 score = 0.62
  expert_7 score = 0.31
  其他 expert 分数更低

TopK=2，所以 token_17 会被送到 expert_3 和 expert_7
```

MoE 的稀疏性就在这里：模型可能有 8、60、64 个 experts，但每个 token 只激活 1 或 2 个。

**Stage 3: All-to-All Dispatch**
```text
for src_gpu in node_gpus:
    gather tokens → NCCL P2P Send to target GPU within node

regroup by target node → NCCL P2P Send across nodes
```

这是 MoE 最关键的一步：**token 不一定在持有目标 expert 的 GPU 上**。

假设：

```text
token_17 当前在 GPU0
router 选择 expert_3 和 expert_7

expert_3 在 GPU5
expert_7 在 GPU2
```

那么 GPU0 要把 `token_17` 的 hidden state 发送给 GPU5 和 GPU2。

这就是 `Dispatch`：

```text
原始 GPU → 持有 expert 的 GPU
```

`All-to-All` 的含义是：

```text
GPU0 可能给 GPU1/GPU2/GPU3/... 发 token
GPU1 也可能给 GPU0/GPU2/GPU3/... 发 token
每个 GPU 发给每个 GPU 的数据量还可能不同
```

分层 All-to-All 的直觉是：

```text
同节点内：先用 NVLink / PCIe 做局部整理
跨节点时：再通过 InfiniBand / RoCE 发到其他节点
```

这样做的目的不是“神奇消灭通信”，而是减少低带宽跨节点链路上的碎片化传输，把快链路和慢链路区别对待。

如果只是单机 8 GPU，严格说就没有 inter-node 阶段；分层思想主要在多节点时更明显。

**Stage 4: Expert FFN**
```text
received_tokens = tokens_for_my_expert
expert_out = ExpertFFN(received_tokens)
```

每张 GPU 收到发给自己 experts 的 token 后，只计算本地 experts。

Expert FFN 通常类似：

```text
ExpertFFN(x):
    a = GateProj(x)
    b = UpProj(x)
    y = DownProj(SiLU(a) * b)
    return y
```

重点是：EP 让 expert 参数留在本地 HBM，不用每次搬 expert 权重；搬的是 token activation。

这是一种典型交换：

```text
少放/少加载 expert 权重
换来 token 的 All-to-All 通信
```

**Stage 5: All-to-All Combine**
```text
combined = AllToAll_Combine(expert_out)
```

专家算完以后，结果还要回到 token 原本所在的位置。

还是刚才的例子：

```text
token_17 从 GPU0 被发到 GPU5 和 GPU2
GPU5 算 expert_3(token_17)
GPU2 算 expert_7(token_17)

现在 GPU5 和 GPU2 要把结果发回 GPU0
GPU0 按 router 权重加权合并：
output_17 = 0.62 * expert_3_out + 0.31 * expert_7_out
```

所以 MoE 层通常有两次 All-to-All：

```text
Dispatch: token → expert
Combine: expert output → token owner
```

**Stage 6: Residual**
```text
output = h + combined
```

这是 Transformer 残差连接。简化理解：

```text
MoE 负责产生一个增量 combined
然后加回原来的 hidden state h
```

实际模型里可能还有 LayerNorm、shared expert、dropout 或不同 residual 位置，但伪代码表达的是主干逻辑。

**为什么 HAP 论文关心这个？**
HAP 的核心观察是：不同阶段适合不同并行策略。

| 场景 | 特征 | 更容易适合 |
|---|---|---|
| 长 context prefill | token 很多，通信量大 | 低通信策略，比如 DP/EP 组合 |
| decode | 每次只生成少量 token，通信量小但负载不均明显 | TP 往往更稳 |
| PCIe 机器 | 带宽低，AllReduce/All-to-All 都贵 | 更要减少通信量 |
| NVLink 机器 | 带宽高，通信压力缓解 | 可以更多考虑计算效率 |

所以你这段不是单纯讲 “DeepSpeed-MoE 怎么写代码”，而是在讲一个系统设计问题：

```text
MoE 推理 = Attention TP 通信 + Expert EP 路由通信 + 硬件带宽 + prefill/decode 阶段差异
```

最该记住的版本：

```text
TP：切矩阵，算完要 AllReduce。
EP：切 experts，token 要 All-to-All 跑去对应 expert。
prefill：token 多，通信容易爆。
decode：token 少，EP 容易负载不均。
HAP：按模块、阶段、硬件自动选 TP/EP/DP 的组合。
```
````

### 003 User
````text
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- review_on_learning_outputs_20260529_185938.md: human_notes/review_on_learning_outputs_20260529_185938.md
- Q1.2_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.2_L1_answer.md
- Q1.1_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.1_L1_answer.md
- I.-INTRODUCTION.md: paper_secs/secs_moe/HAP Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference/I.-INTRODUCTION.md

## My request for Codex:
'DP=2：有 2 份并行副本，处理不同 batch/request
TP=2：Attention/线性层权重切成 2 份
EP=2：Experts 切成 2 份，每组 GPU 只放一部分 experts',详细点,例子
````

### 004 Codex Final
````text
可以，把这三句话拆开讲。核心是：

```text
8 GPUs = DP 维度 × TP 维度 × EP 维度
       = 2 × 2 × 2
```

你可以把 8 张 GPU 看成这个三维表：

```text
DP0 逻辑模型副本:
          EP0              EP1
TP0       GPU0             GPU1
TP1       GPU2             GPU3

DP1 逻辑模型副本:
          EP0              EP1
TP0       GPU4             GPU5
TP1       GPU6             GPU7
```

**1. DP=2：两个“逻辑模型副本”处理不同请求**
DP 是 Data Parallelism。

假设现在有 4 个请求：

```text
Request A, B, C, D
```

DP=2 时可以这样分：

```text
DP0: GPU0-GPU3 处理 Request A, B
DP1: GPU4-GPU7 处理 Request C, D
```

重点：DP 复制的是“完整逻辑模型”，不是说每张 GPU 都有完整模型。

因为一个 DP 副本内部还有 TP 和 EP，所以：

```text
DP0 的完整模型 = GPU0 + GPU1 + GPU2 + GPU3 共同组成
DP1 的完整模型 = GPU4 + GPU5 + GPU6 + GPU7 共同组成
```

推理时，DP0 和 DP1 基本不需要互相通信。它们只是各自处理不同 batch/request。DP 主要提升吞吐量，不一定降低单个请求延迟。

**2. TP=2：Attention/线性层权重切成 2 份**
TP 是 Tensor Parallelism。

假设 Attention 有 32 个 heads，hidden size 是 4096。TP=2 可以理解为：

```text
TP0: 负责前 16 个 attention heads，或者部分输出通道
TP1: 负责后 16 个 attention heads，或者另一部分输出通道
```

在 DP0 里面：

```text
GPU0 和 GPU2 是一组 TP pair
GPU1 和 GPU3 也是一组 TP pair
```

比如看 `GPU0 + GPU2`：

```text
GPU0: 持有 Attention 权重的一半
GPU2: 持有 Attention 权重的另一半
```

输入 token 来了以后：

```text
GPU0 算一半 attention 结果
GPU2 算另一半 attention 结果
然后 AllReduce / 聚合，得到完整 hidden state
```

用一个简化矩阵例子：

```text
原始 W_qkv: [4096, 12288]

TP=2 后:
GPU0 持有 W_qkv_0: [4096, 6144]
GPU2 持有 W_qkv_1: [4096, 6144]
```

两个 GPU 各算一半 QKV，最后拼起来或规约成完整结果。

所以这句：

```text
TP=2：Attention/线性层权重切成 2 份
```

意思是：

```text
同一个请求的 Attention 不是一张卡独立算完，
而是两个 TP rank 合作算完。
```

代价是：TP rank 之间需要通信，比如 `AllReduce`。

**3. EP=2：Experts 切成 2 份**
EP 是 Expert Parallelism。

假设一个 MoE 层有 8 个 experts：

```text
Expert 0, 1, 2, 3, 4, 5, 6, 7
```

EP=2 时，可以这样切：

```text
EP0: Expert 0, 1, 2, 3
EP1: Expert 4, 5, 6, 7
```

在 DP0 里面：

```text
EP0 列: GPU0, GPU2 持有 Expert 0-3
EP1 列: GPU1, GPU3 持有 Expert 4-7
```

所以：

```text
GPU0: TP0 的 attention shard + EP0 的 experts
GPU1: TP0 的 attention shard + EP1 的 experts
GPU2: TP1 的 attention shard + EP0 的 experts
GPU3: TP1 的 attention shard + EP1 的 experts
```

这就是你笔记里那句：

```text
每 GPU 持有: 1/TP attention 权重 + 1/EP expert 权重
```

对于 `TP=2, EP=2`：

```text
每张 GPU 持有 1/2 Attention 权重
每张 GPU 持有 1/2 Expert 集合
```

**完整例子**
假设 DP0 正在处理 token `t0, t1, t2, t3`，Router/Gate 给出 Top-2 experts：

```text
t0 -> Expert 1, Expert 6
t1 -> Expert 2, Expert 3
t2 -> Expert 5, Expert 7
t3 -> Expert 0, Expert 4
```

专家分布是：

```text
EP0: Expert 0,1,2,3
EP1: Expert 4,5,6,7
```

那么 token 路由会变成：

```text
t0:
  Expert 1 在 EP0，本地算
  Expert 6 在 EP1，要发给 EP1

t1:
  Expert 2 在 EP0，本地算
  Expert 3 在 EP0，本地算

t2:
  Expert 5 在 EP1，要发给 EP1
  Expert 7 在 EP1，要发给 EP1

t3:
  Expert 0 在 EP0，本地算
  Expert 4 在 EP1，要发给 EP1
```

所以 EP 带来的典型流程是：

```text
1. Router 决定每个 token 去哪些 experts
2. All-to-All Dispatch 把 token 发到对应 expert 所在 GPU
3. Expert FFN 计算
4. All-to-All Combine 把结果发回原 token 所在位置
5. 按 gate 权重加权合并
```

比如 `t0`：

```text
t0 -> Expert 1 和 Expert 6

Expert 1 输出: y1
Expert 6 输出: y6

Gate 权重:
  Expert 1 权重 = 0.7
  Expert 6 权重 = 0.3

最终:
  y_t0 = 0.7 * y1 + 0.3 * y6
```

**一句话总结**
```text
DP=2: 两套逻辑模型副本，各自处理不同请求，提高吞吐。
TP=2: 同一个 Attention/Linear 被 2 张卡切开合作计算，需要 AllReduce。
EP=2: Experts 分成 2 组，token 根据 Router 被 All-to-All 发到对应 expert。
```

最重要的区别是：

```text
DP 是“不同请求分开跑”
TP 是“同一个算子切开跑”
EP 是“不同 expert 分开放，token 跑去找 expert”
```
````

### 005 User
````text
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- review_on_learning_outputs_20260529_185938.md: human_notes/review_on_learning_outputs_20260529_185938.md
- Q1.2_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.2_L1_answer.md
- Q1.1_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.1_L1_answer.md
- I.-INTRODUCTION.md: paper_secs/secs_moe/HAP Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference/I.-INTRODUCTION.md

## My request for Codex:
DP是数据和权重都复制#DP份(不同请求或batch并行,不需要通信),分别放到#DP个执行实例.TP是Attn和linear layer的权重切分到#TP份(layer中的不同张量并行).EP是MoE layer的所有Expert权重拆分到#EP份(不同Expert并行).TP和EP是分别作用于不同layer的张量并行方式(分拆权重,分拆数据,需要通信)?
````

### 006 Codex Final
````text
对，你理解的主干是对的，但有几个地方需要稍微精确化。

可以改成这版：

```text
DP: 复制“完整逻辑模型” #DP 份，不同 batch/request 分给不同 DP 实例。
    推理时 DP 实例之间通常不通信。

TP: 把 Attention / Linear / FFN 里的大矩阵按张量维度切成 #TP 份。
    同一个 token/request 需要多个 TP rank 合作计算，所以需要 AllReduce / AllGather / ReduceScatter。

EP: 把 MoE layer 里的 experts 按 expert 维度切成 #EP 份。
    token 根据 router 被发送到对应 expert 所在 GPU，所以需要 All-to-All Dispatch 和 Combine。
```

关键修正是：**DP 复制的是完整逻辑模型，不一定是每张 GPU 都复制完整权重**。

在 `DP=2, TP=2, EP=2` 时：

```text
总 GPU 数 = DP × TP × EP = 2 × 2 × 2 = 8
```

每个 DP 实例内部有 `TP × EP = 4` 张 GPU。也就是说：

```text
DP0: GPU0-GPU3 共同组成一份完整逻辑模型
DP1: GPU4-GPU7 共同组成另一份完整逻辑模型
```

不是：

```text
GPU0 有完整模型
GPU1 有完整模型
...
```

而是：

```text
GPU0-GPU3 的权重 shard 合起来 = 一份完整模型
GPU4-GPU7 的权重 shard 合起来 = 另一份完整模型
```

**TP 和 EP 的区别**
你的这句话：

> TP和EP是分别作用于不同layer的张量并行方式

要稍微改一下。更准确是：

```text
TP 是 tensor-level parallelism：切一个算子/矩阵内部的张量维度。
EP 是 expert-level parallelism：切 MoE 层里的 expert 集合。
```

所以 EP 不太叫“张量并行”，它更像是 **按专家粒度的模型并行 / 参数并行**。

对比一下：

| 并行 | 切什么 | 数据怎么动 | 通信 |
|---|---|---|---|
| DP | 切 batch/request | 不同请求给不同副本 | 推理通常无通信 |
| TP | 切矩阵/hidden/head/channel | 同一批 token 被多个 rank 共同计算 | AllReduce / AllGather / ReduceScatter |
| EP | 切 experts | token 被 router 动态送到目标 expert | All-to-All |

**“分拆数据”也要区分**
DP 的数据拆分是：

```text
Request A, B -> DP0
Request C, D -> DP1
```

这是按 batch/request 拆。

TP 的“数据拆分”不是按 request 拆，而是同一个 token 的 hidden 向量可能被切通道，或者中间结果分布在多个 TP rank 上。

EP 的数据拆分是动态的：

```text
token0 -> expert1, expert6
token1 -> expert2, expert3
token2 -> expert5, expert7
```

它不是固定把 batch 平均切开，而是 router 决定 token 去哪里。Top-2 routing 时，一个 token 还会被复制到两个 experts。

**最精确的一句话**
```text
DP 复制逻辑模型、切 batch/request；
TP 切 dense/attention/linear 的张量维度，同一请求协同计算；
EP 切 MoE experts，token 按 router 做动态 All-to-All 路由。
```

所以你最后括号里的说法：

> 分拆权重, 分拆数据, 需要通信

可以保留，但建议写成：

```text
TP/EP 都会分拆权重，也都会让 activation/data 在设备间重排；
TP 的通信是为了合并同一个算子的部分结果；
EP 的通信是为了把 token 送到对应 expert 并把结果送回来。
```
````

### 007 User
````text
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- review_on_learning_outputs_20260529_185938.md: human_notes/review_on_learning_outputs_20260529_185938.md
- Q1.2_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.2_L1_answer.md
- Q1.1_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.1_L1_answer.md

## My request for Codex:
权重复制多份, 计算随之并行, 隔离并行(重复存储开销). 权重分拆多份, 计算随之并行, 组成pipeline(额外通信开销). 
````

### 008 Codex Final
````text
对，这个抽象很接近本质。建议你改成这版，更精确：

```text
复制型并行：权重复制多份，请求/batch 分给不同副本并行执行。
优点是执行实例之间隔离，推理时几乎不通信；缺点是重复存储权重。

分片型并行：权重拆成多份，同一个请求需要多个设备协同计算。
优点是降低单设备显存和计算压力；缺点是中间 activation / token / partial result 需要通信。
```

但你这句里有一个词要小心：

> 组成 pipeline

TP 和 EP **不一定组成 pipeline**。更准确是 **组成协同计算图**。

区别是：

```text
TP: 同一层里的矩阵切开，同时算，然后 AllReduce / AllGather 合并。
EP: MoE experts 分开放，token 被 All-to-All 发到目标 expert。
PP: 不同层放到不同设备，micro-batch 像流水线一样流过设备。
```

所以：

| 类型 | 权重怎么放 | 计算关系 | 通信形态 |
|---|---|---|---|
| DP | 复制完整逻辑模型 | 不同请求隔离并行 | 推理基本无通信 |
| TP | 切 Attention/Linear 张量 | 同一层协同计算 | AllReduce / AllGather |
| EP | 切 MoE experts | token 动态找 expert | All-to-All Dispatch/Combine |
| PP | 切不同 layer | micro-batch 流水线 | 层间 activation 传递 |

你的句子可以最终写成：

```text
DP 是复制型并行：权重复制 #DP 份，不同 batch/request 隔离并行，换取吞吐，代价是重复存储。

TP/EP 是分片型并行：权重拆成 #TP/#EP 份，同一个请求跨设备协同计算，代价是 activation / partial result / token routing 的额外通信。

PP 才是严格意义上的 pipeline：按 layer 切分，micro-batch 在设备间流水执行。
```

再压缩成一句：

```text
DP 用“多份模型跑多份数据”换吞吐；TP/EP 用“多张卡拼一份模型”换显存和算力；PP 用“多张卡接力跑不同层”换模型规模。
```
````

### 009 User
````text
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- review_on_learning_outputs_20260529_185938.md: human_notes/review_on_learning_outputs_20260529_185938.md
- Q1.2_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.2_L1_answer.md
- Q1.1_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.1_L1_answer.md

## My request for Codex:
还有context parallel,是什么?
````

### 010 Codex Final
````text
Context Parallelism，简称 `CP`，可以理解为：

```text
CP = 沿 sequence/context length 维度切分 token。
```

如果 hidden state 是：

```text
h: [batch, seq_len, hidden_dim]
```

那么几种并行大概是：

```text
DP: 切 batch 维度
TP: 切 hidden/head/channel 维度
EP: 切 expert 维度
CP: 切 seq_len/context 维度
PP: 切 layer 维度
```

所以 CP 和你前面说的 TP/EP 不太一样。**CP 通常不是主要切权重，而是切 activation / KV cache / token 序列。**

例如长上下文：

```text
seq_len = 8192
CP = 4
```

可以把 context 切成 4 段：

```text
GPU0: token 0-2047
GPU1: token 2048-4095
GPU2: token 4096-6143
GPU3: token 6144-8191
```

每张 GPU 只保存一段 token 的 activation / KV cache。

**为什么需要 CP？**
因为长上下文 Attention 很贵：

```text
Attention 复杂度约 O(seq_len^2)
KV cache / activation 内存约 O(seq_len)
```

如果 `seq_len=128K`、`1M tokens`，单卡放不下，或者 attention prefill 太慢。CP 就把长序列拆到多张卡上。

**但是 CP 需要通信**
问题在于 attention 不是每个 token 只看自己这段。标准 causal attention 里：

```text
token i 需要看 token 0...i 的 K/V
```

所以如果 GPU3 负责最后一段 token，它的 query 需要看到 GPU0/GPU1/GPU2/GPU3 上的 KV。

因此 CP 的典型通信是：

```text
每个 GPU 持有自己的 Q chunk
但需要交换 / 流式读取其他 GPU 的 K/V chunk
最后计算自己负责 token 的 attention output
```

简化伪代码：

```text
# CP=4, 每个 rank 持有一段 context
Q_local = Q[:, my_seq_chunk, :]
K_local = K[:, my_seq_chunk, :]
V_local = V[:, my_seq_chunk, :]

output_local = 0

for kv_chunk in all_CP_ranks:
    K_remote, V_remote = get_kv_chunk(kv_chunk)
    partial = Attention(Q_local, K_remote, V_remote)
    output_local = online_softmax_merge(output_local, partial)

return output_local
```

这里 `get_kv_chunk` 可以用不同方式实现：

```text
AllGather: 先把所有 K/V 收齐，再算，简单但内存和通信压力大。
Ring Attention: K/V 块在 GPU 间环形传递，边收边算，省显存。
2D/Hierarchical CP: 同时沿 Q 和 KV tile，减少跨节点通信。
```

**和 DP/TP/EP 的对比**
| 并行 | 切什么 | 权重是否复制 | 主要通信 |
|---|---|---|---|
| DP | batch/request | 复制逻辑模型 | 推理通常无通信 |
| TP | hidden/head/linear tensor | 权重分片 | AllReduce / AllGather |
| EP | expert 集合 | expert 权重分片 | All-to-All token routing |
| CP | sequence/context | 权重通常复制，activation/KV 分片 | K/V exchange / ring / all-gather |
| PP | layer | 不同层权重分片 | 层间 activation 传递 |

所以你可以把它加进你的总结里：

```text
CP 是上下文维度的数据分片并行：
不主要切模型权重，而是把长序列 token / KV cache / activation 按 context length 拆到多设备上；
同一个请求跨设备协同完成 attention，代价是跨设备交换 K/V 或 attention partial results。
```

一句话记忆：

```text
DP 切 batch，TP 切 hidden，EP 切 expert，CP 切 sequence，PP 切 layer。
```
````

### 011 User
````text
# Context from my IDE setup:

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- L1_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L1_horizon_summary.md
- SKILL.md: .codex/skills/export-conversation-notes/SKILL.md
- Accelerating Sparse Transformer Inference on GPU (STOF).md: human_notes/Accelerating Sparse Transformer Inference on GPU (STOF).md
- Adaptive Draft Sequence Length_ Enhancing Speculative Decoding Throughput on PIM-Enabled Systems.md: human_notes/Adaptive Draft Sequence Length_ Enhancing Speculative Decoding Throughput on PIM-Enabled Systems.md

## My request for Codex:
[$export-conversation-notes](/data3/paper_analysis/.codex/skills/export-conversation-notes/SKILL.md)
````
