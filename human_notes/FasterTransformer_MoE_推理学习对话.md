# FasterTransformer MoE 推理学习对话

- 导出时间：2026-06-03 17:30 CST
- 来源：当前 Codex session 可用对话上下文
- 保存路径：human_notes/FasterTransformer_MoE_推理学习对话.md
- 范围：仅用户输入与 Codex 最终输出

## 对话记录

### 001 User
<environment_context>
  <cwd>/data3/paper_analysis</cwd>
  <shell>bash</shell>
  <current_date>2026-06-02</current_date>
  <timezone>Asia/Shanghai</timezone>
</environment_context>

### 002 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Active selection of the file:
### 方法6: FasterTransformer (已归档) — 生产级单 GPU MoE 推理

**笔记证据**: `knowledge_notes/编译知识笔记/FasterTransformer.md` (score: 171.9)

FasterTransformer 是 NVIDIA 的高性能 Transformer 推理库 (已归档至 TensorRT-LLM)。"Who Says Elephants Can't Run" 基于它构建首个生产级单 GPU MoE 推理系统，核心扩展：

```
FasterTransformer MoE 扩展 (已并入 TensorRT-LLM):
==================================================
1. DeepSpeed MoE 模型格式支持
2. GPU-efficient Token Routing:
   CUB radix sort (按 expert ID 排序 token)
   + CUTLASS GroupedGEMM (合并各 expert 的 GEMM)
3. INT4/INT8 Fused GEMM+Dequantize kernel
4. MoE Decoder Batch Pruning
5. Triton Inference Server 集成 (云规模弹性部署)
```

**注解**:
- **Pre-gated MoE (ISCA '24)**: 基于 FasterTransformer 构建，实现 pre-gate function 预测下一层激活 experts → 异步预取，消除 CPU→GPU 迁移延迟
- **Diff-MoE**: 基于 FasterTransformer v5.2，实现 per-expert 文件加载 + 三级缓存 (HPC/MPC/LPC) + GRU predictor 预取

---
## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- Q1.4_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.4_L1_answer.md
- SKILL.md: .codex/skills/export-conversation-notes/SKILL.md
- SKILL.md: .claude/skills/export-conversation-notes/SKILL.md
- AdaServe_ Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding.md: conv_notes/AdaServe_ Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding.md

## My request for Codex:
'### 方法6: FasterTransformer (已归档) — 生产级单 GPU MoE 推理

**笔记证据**: `knowledge_notes/编译知识笔记/FasterTransformer.md` (score: 171.9)

FasterTransformer 是 NVIDIA 的高性能 Transformer 推理库 (已归档至 TensorRT-LLM)。"Who Says Elephants Can't Run" 基于它构建首个生产级单 GPU MoE 推理系统，核心扩展：

```
FasterTransformer MoE 扩展 (已并入 TensorRT-LLM):
==================================================
1. DeepSpeed MoE 模型格式支持
2. GPU-efficient Token Routing:
   CUB radix sort (按 expert ID 排序 token)
   + CUTLASS GroupedGEMM (合并各 expert 的 GEMM)
3. INT4/INT8 Fused GEMM+Dequantize kernel
4. MoE Decoder Batch Pruning
5. Triton Inference Server 集成 (云规模弹性部署)
```

**注解**:
- **Pre-gated MoE (ISCA '24)**: 基于 FasterTransformer 构建，实现 pre-gate function 预测下一层激活 experts → 异步预取，消除 CPU→GPU 迁移延迟
- **Diff-MoE**: 基于 FasterTransformer v5.2，实现 per-expert 文件加载 + 三级缓存 (HPC/MPC/LPC) + GRU predictor 预取

---',教我

### 003 Codex Final
可以，把这段先抓成一句话：

**FasterTransformer 在这里不是一个算法名字，而是一个“单 GPU 上把 MoE 跑快、跑稳、跑进生产环境”的 CUDA/C++ 推理底座。**  
`Who Says Elephants Can't Run` 做的事，就是在 FasterTransformer 上补齐 MoE 需要的 routing、GroupedGEMM、量化、batch pruning、Triton serving 这些工程能力。

**核心痛点：MoE 推理为什么难？**

普通 dense FFN 是这样：

```text
所有 token → 同一个 FFN 权重 → 一个大 GEMM
```

MoE FFN 是这样：

```text
token0 → expert 3
token1 → expert 7
token2 → expert 3
token3 → expert 1
...
```

也就是说，每个 token 要先经过 router/gate 选择 expert。问题来了：

1. token 分散到不同 expert，内存布局乱。
2. 每个 expert 处理的 token 数不同，GEMM 形状不统一。
3. 如果每个 expert 单独 launch kernel，kernel launch 太多。
4. expert 权重很大，显存和带宽压力巨大。
5. decoder 里有些句子已经生成完了，还继续参与 batch 会浪费计算。

所以这段列出的 5 个扩展，本质是在逐个解决这些问题。

**1. DeepSpeed MoE 模型格式支持**

这只是入口问题：很多 MoE 模型训练时用 DeepSpeed MoE 保存权重。FasterTransformer 原本不一定认识这种格式，所以要支持它的 expert 权重布局、gate 参数、layer 结构。

你可以理解成：

```text
DeepSpeed 训练产物
        ↓ 转换/加载
FasterTransformer 可执行的 MoE 权重格式
```

没有这一步，后面的 CUDA 优化都用不上。

**2. GPU-efficient Token Routing：CUB radix sort + CUTLASS GroupedGEMM**

这是最关键的部分。

MoE routing 的目标是把 token 按 expert 分组。比如：

```text
原始 token 顺序:
t0 t1 t2 t3 t4 t5 t6 t7

router 选择:
t0→E2
t1→E0
t2→E2
t3→E1
t4→E0
t5→E1
t6→E2
t7→E0
```

先用 `CUB radix sort` 按 expert id 排序：

```text
排序后:
E0: t1 t4 t7
E1: t3 t5
E2: t0 t2 t6
```

这样每个 expert 的 token 就连续了。然后用 `CUTLASS GroupedGEMM` 一次性执行多个 expert 的 FFN：

```text
GroupedGEMM:
[E0 tokens] × W_E0
[E1 tokens] × W_E1
[E2 tokens] × W_E2
```

重点是：**不是每个 expert launch 一个 GEMM，而是把多个不同大小的 GEMM 打包进一次 grouped kernel 调用。**  
这就把 naive PyTorch 里大量 Python 循环、scatter/gather、kernel launch 开销干掉了。

**3. INT4/INT8 Fused GEMM+Dequantize kernel**

MoE 最大的问题之一是 expert 权重太多。虽然每个 token 只激活少数 expert，但所有 expert 权重都要存着。

所以它做 weight-only quantization：

```text
权重: FP16 → INT8 / INT4
激活: 仍然 FP16
```

推理时不能真的直接用 INT4 当 FP16 算，所以要反量化：

```text
W_int4 → dequantize → W_fp16 → GEMM
```

但如果单独做 dequantize，会多一次读写内存，反而慢。于是论文把它融合进 GEMM kernel：

```text
加载低比特权重 → 在寄存器里反量化 → 直接送 Tensor Core GEMM
```

这就是 `Fused GEMM+Dequantize`。  
直觉上：**不要先拆箱再搬一次，而是在计算现场边拆边算。**

**4. MoE Decoder Batch Pruning**

这是 decoder / beam search 场景下的优化。

假设 batch 里有 4 个句子：

```text
句子 A: 还没结束
句子 B: 已生成 EOS
句子 C: 还没结束
句子 D: 已生成 EOS
```

如果不剪枝，B 和 D 虽然已经完成，后面每一步还可能跟着 batch 进入 MoE routing，浪费 expert 计算。

Batch pruning 的做法很聪明：给完成的句子设置一个极大的 expert id，比如 `INT_MAX`：

```text
active token: expert_id = 0/1/2/...
finished token: expert_id = INT_MAX
```

经过 radix sort 后，完成的 token 自动排到最后：

```text
前面: active tokens
后面: finished tokens
```

后续 GroupedGEMM 只处理前面的 `active_tokens`。  
所以它不是复杂地删除数据结构，而是借 routing sort 顺手完成剪枝。

**5. Triton Inference Server 集成**

前面 1-4 都是“单次推理怎么快”。  
Triton 解决的是“生产服务怎么跑”。

它负责：

```text
客户端请求
  ↓
Triton dynamic batching
  ↓
FasterTransformer backend
  ↓
GPU MoE 推理
  ↓
返回结果
```

也就是请求聚合、模型实例管理、HTTP/gRPC 服务、动态 batching、部署扩缩容这些生产能力。

**Pre-gated MoE 和 Diff-MoE 的位置**

这两个注解是在说：后来的论文继续把 FasterTransformer 当底座改。

`Pre-gated MoE` 解决的是：

```text
expert 太多，GPU 放不下 → 放 CPU
但 CPU→GPU 传输慢 → 提前预测下一层要用哪些 expert → 异步预取
```

核心流程：

```text
当前层正在算 expert
同时另一个 CUDA stream 把下一层 expert 从 CPU 搬到 GPU
```

`Diff-MoE` 解决的是：

```text
不是所有 expert 都一样重要
常用 expert 留 GPU
短期热点 expert 临时缓存
偶尔用的 expert 用完就驱逐
```

它有三级缓存：

```text
HPC: 长期热门 expert，常驻 GPU
MPC: 最近热门 expert，动态替换
LPC: 临时加载/预取缓冲区
```

**你可以这样记**

```text
FasterTransformer = 高性能 Transformer/MoE CUDA 推理底座

Who Says Elephants Can't Run =
  让单 GPU MoE 进入生产:
  routing 快
  GEMM 快
  权重小
  decoder 少浪费
  Triton 可部署

Pre-gated MoE =
  expert 放 CPU，提前预测并搬到 GPU

Diff-MoE =
  expert 分冷热，做 GPU 缓存管理
```

最重要的技术链条是：

```text
gate 选 expert
→ CUB radix sort 按 expert 分组 token
→ CUTLASS GroupedGEMM 一次算多个 expert
→ INT4/INT8 fused dequant 降低权重带宽
→ batch pruning 跳过已完成序列
→ Triton 承接生产服务
```

你理解了这条链，就基本理解这段笔记的主干了。

### 004 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Active selection of the file:
DeepSpeed MoE 模型格式支持
## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- SKILL.md: .claude/skills/obsidian-keyword-explain/SKILL.md
- SKILL.md: .codex/skills/obsidian-keyword-explainer/SKILL.md
- Q1.4_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.4_L1_answer.md
- SKILL.md: .codex/skills/export-conversation-notes/SKILL.md

## My request for Codex:
细说:'DeepSpeed MoE 模型格式支持'是什么?怎么做?以及triton 作为serving scheduler?triton不是block level的计算描述语言?

### 005 Codex Final
你这里抓到了一个容易被轻描淡写的工程点。

**DeepSpeed MoE 模型格式支持**

这里的“格式支持”不是说 FasterTransformer 去调用 DeepSpeed 跑推理，而是说：**一个用 DeepSpeed MoE 训练/保存出来的 MoE checkpoint，FasterTransformer 能读懂、转换、加载，并按自己的 C++/CUDA kernel 跑起来。**

DeepSpeed MoE 模型里不只是普通 Transformer 权重，还多了这些信息：

- `num_experts`：每层多少个 expert
- `ep_size`：expert parallel group 多大
- `k`：top-1 还是 top-2 gating
- gate/router 权重
- 每个 expert 的 FFN 权重
- expert 在不同 GPU/rank 上的分片关系
- checkpoint shard / rank-local state_dict 命名

DeepSpeed 官方 MoE layer 本身就暴露这些核心参数，如 `num_experts`、`ep_size`、`k` 等；DeepSpeed 推理 API 也需要传 `mp_size`、`moe_experts`、`checkpoint` 等信息来恢复 MoE 并建立并行组。参见 DeepSpeed 文档：MoE layer 参数和 inference 初始化说明。([deepspeed.readthedocs.io](https://deepspeed.readthedocs.io/en/latest/moe.html)) ([deepspeed.ai](https://www.deepspeed.ai/tutorials/mixture-of-experts-inference/))

所以 FasterTransformer 要做的“格式支持”大概是这条链：

```text
DeepSpeed checkpoint
  ↓
解析模型结构: layer数 / hidden size / expert数 / top-k / MoE层位置
  ↓
合并或定位 checkpoint shards: DP / TP / EP / ZeRO 分片
  ↓
把 DeepSpeed state_dict key 映射到 FasterTransformer 内部权重名
  ↓
转置 / 重排 / pack 权重布局，适配 CUTLASS / 自定义 CUDA kernel
  ↓
可选: INT8 / INT4 weight-only quantization
  ↓
FasterTransformer runtime 加载并执行 MoE
```

一个简化伪代码是：

```python
for layer in moe_layers:
    gate_w = load_deepspeed_gate_weight(layer)

    write_ft_weight(f"layer.{layer}.moe.gate", convert(gate_w))

    for expert_id in range(num_experts):
        owner_rank = find_expert_owner(expert_id, ep_size)
        local_id = global_to_local_expert_id(expert_id, owner_rank)

        w1, w2 = load_expert_ffn_weights(layer, owner_rank, local_id)
        w1 = merge_tensor_parallel_shards(w1)
        w2 = merge_tensor_parallel_shards(w2)

        w1 = transpose_pack_for_grouped_gemm(w1)
        w2 = transpose_pack_for_grouped_gemm(w2)

        if use_int4:
            w1, scale1 = quantize_int4(w1)
            w2, scale2 = quantize_int4(w2)

        write_ft_expert(layer, expert_id, w1, w2, scales)
```

注解：

- `find_expert_owner`：解决 DeepSpeed expert parallel 里“哪个 rank 存哪个 expert”的问题。
- `merge_tensor_parallel_shards`：如果训练时还用了 tensor parallel，要把分片拼回推理 kernel 期望的形状。
- `transpose_pack_for_grouped_gemm`：PyTorch `Linear.weight` 常是 `[out, in]`，而推理 GEMM kernel 可能需要转置或特定内存布局。
- `quantize_int4`：论文里的 INT4/INT8 expert 权重压缩一般在转换阶段或加载阶段完成。
- 这一步本质是“checkpoint ABI 适配”，不是模型算法创新。

**Triton 作为 Serving Scheduler？**

这里的 Triton 是 **NVIDIA Triton Inference Server**，不是你说的那个 block-level GPU kernel DSL。

它们是两个东西：

| 名字 | 层级 | 做什么 |
|---|---|---|
| **NVIDIA Triton Inference Server** | serving / runtime 层 | 接 HTTP/gRPC 请求、排队、dynamic batching、模型实例管理、调用 backend |
| **Triton language / triton-lang / OpenAI Triton** | kernel / compiler 层 | 写 GPU kernel，描述 block-level/tile-level 计算，编译成 GPU 代码 |

所以论文里说的 Triton 是这个：

```text
Client requests
  ↓
NVIDIA Triton Inference Server
  ↓ dynamic batching / request scheduling
FasterTransformer backend
  ↓
CUB radix sort / CUTLASS GroupedGEMM / CUDA kernels
  ↓
GPU execution
```

NVIDIA Triton 官方文档明确说它包含 scheduling 和 batching algorithms；默认 scheduler 会把 inference requests 分发到配置好的 model instances，dynamic batching 会把多个请求合成一个 batch。([docs.nvidia.com](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/scheduler.html))

而你说的 block-level Triton 是这个：

```python
@triton.jit
def matmul_kernel(...):
    # 一个 program instance 处理一个 tile/block
    ...
```

Triton language 文档里讲的是 “Blocked Program, Scalar Threads” 这种编程模型，也就是 block/tile 级 GPU kernel 描述。([triton-lang.org](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html))

一句话区分：

```text
Triton Inference Server 调度请求和模型实例。
Triton language 描述 GPU kernel 的 block/tile 计算。
```

在 `Who Says Elephants Can't Run` 这段里，**Triton = NVIDIA Triton Inference Server**。真正的 kernel 计算描述/实现主要是 FasterTransformer + CUB + CUTLASS + 自定义 CUDA kernel，不是 Triton language。

### 006 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- SKILL.md: .claude/skills/obsidian-keyword-explain/SKILL.md
- SKILL.md: .codex/skills/obsidian-keyword-explainer/SKILL.md
- Q1.4_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.4_L1_answer.md
- SKILL.md: .codex/skills/export-conversation-notes/SKILL.md

## My request for Codex:
deepspeed有什么特别的?cub是什么?

### 007 Codex Final
DeepSpeed 特别在：**它不是一个普通模型库，而是“大模型分布式训练/推理系统”**。在 MoE 里尤其重要，因为 MoE 天生牵涉“专家怎么分布、token 怎么跨 GPU 发给专家、checkpoint 怎么按 expert/rank 保存”。

可以这样分层理解：

```text
HuggingFace / PyTorch:
  主要描述模型结构和前向计算，方便写模型。

DeepSpeed:
  负责把超大模型拆到很多 GPU / CPU / NVMe 上跑，
  管 ZeRO、并行组、通信、checkpoint 分片、MoE expert parallelism。

FasterTransformer:
  负责把已经训练好的模型用 C++/CUDA 高性能推理跑起来。
```

在 MoE 里，DeepSpeed 的特别之处主要是这几个：

1. **Expert Parallelism**
   
   MoE 里有很多 expert，不一定每张 GPU 都存所有 expert。DeepSpeed 会把 expert 分到不同 GPU：

```text
GPU0: expert 0, 1
GPU1: expert 2, 3
GPU2: expert 4, 5
GPU3: expert 6, 7
```

token 被 router 分配到 expert 后，如果 expert 在别的 GPU，就要 all-to-all 通信。

2. **Token Dispatch / All-to-All**

```text
每个 GPU 上的 token
  ↓ router 选 expert
跨 GPU all-to-all 发送 token
  ↓
各 GPU 计算本地 expert
  ↓
all-to-all 把结果送回原 GPU
```

这是 DeepSpeed MoE 的核心运行模式。

3. **ZeRO / Offload / Sharded Checkpoint**

DeepSpeed 很擅长把 optimizer state、gradient、parameter 拆开存。训练出的 checkpoint 往往不是一个简单 `.bin`，而是一堆按 data parallel / tensor parallel / expert parallel rank 切开的文件。

所以 “FasterTransformer 支持 DeepSpeed MoE 模型格式” 的意思是：

```text
DeepSpeed 训练出来的一堆分片 checkpoint
  ↓
解析 expert 分布、gate 权重、FFN 权重、rank shard
  ↓
转成 FasterTransformer 能加载的权重布局
  ↓
用 FasterTransformer 的 CUDA kernel 推理
```

一句话：**DeepSpeed 特别在训练/分布式系统侧；FasterTransformer 特别在单机/单卡高性能推理侧。**

CUB 是 **NVIDIA CUDA 里的高性能并行基础算法库**。你可以把它理解成 GPU 上的 STL/算法工具箱，提供：

```text
sort
scan / prefix sum
reduce
select
histogram
partition
```

在这篇 MoE 里用的是 `CUB radix sort`，也就是 GPU 上非常快的整数排序。

为什么 MoE 要 CUB？因为 token routing 需要把 token 按 expert id 分组。

例子：

```text
token:      t0 t1 t2 t3 t4
expert_id:  2  0  2  1  0
```

用 CUB 按 `expert_id` 排序：

```text
expert_id:  0  0  1  2  2
token:     t1 t4 t3 t0 t2
```

这样 token 就连续分组了：

```text
expert 0: t1, t4
expert 1: t3
expert 2: t0, t2
```

接下来 CUTLASS GroupedGEMM 就可以一次性算多个 expert 的 GEMM。

所以这几个东西的分工是：

```text
DeepSpeed: 训练 MoE / 保存 MoE checkpoint / 分布式 expert parallel
CUB: GPU 上做排序、scan、reduce 这类基础并行算法
CUTLASS: GPU 上做高性能 GEMM
FasterTransformer: 把 CUB + CUTLASS + 自定义 CUDA kernel 组织成推理框架
```

最短记忆版：

```text
DeepSpeed = 大模型分布式训练系统，MoE checkpoint 来源
CUB = CUDA 并行算法库，MoE 里用它按 expert_id 给 token 排序
```

### 008 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Active selection of the file:
CUTLASS
## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- SKILL.md: .claude/skills/obsidian-keyword-explain/SKILL.md
- SKILL.md: .codex/skills/obsidian-keyword-explainer/SKILL.md
- Q1.4_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.4_L1_answer.md
- SKILL.md: .codex/skills/export-conversation-notes/SKILL.md

## My request for Codex:
CUTLASS是什么?

### 009 Codex Final
CUTLASS 全名是 **CUDA Templates for Linear Algebra Subroutines**，是 NVIDIA 开源的 **C++ 模板级 GPU 线性代数库**。

最简单理解：

```text
cuBLAS  = NVIDIA 做好的黑盒高性能 GEMM
CUTLASS = 让你自己用模板拼一个高性能 GEMM kernel
CUB     = GPU 上的 sort / scan / reduce 等基础并行算法库
```

CUTLASS 主要用来写/生成高性能矩阵乘法：

```text
C = A × B
```

但它不是普通矩阵乘法库，而是把 GEMM 拆成 GPU 硬件层级：

```text
GEMM
  ↓
Threadblock tile   一个 CUDA block 负责一块矩阵
  ↓
Warp tile          一个 warp 负责 block 内的一小块
  ↓
MMA instruction    Tensor Core 执行最底层矩阵乘累加
```

比如一个 GEMM 可以被切成：

```text
Threadblock tile: 128 × 128 × 64
Warp tile:         64 × 64 × 64
Instruction tile:  16 × 8 × 32
```

CUTLASS 让你通过 C++ 模板参数指定：

```text
数据类型: FP16 / BF16 / INT8 / INT4 / FP8
矩阵布局: RowMajor / ColumnMajor
tile size
pipeline stages
是否使用 Tensor Core
epilogue 是否融合 scale / bias / dequant
```

所以它很适合做自定义高性能 kernel。

在你这段 MoE 里，CUTLASS 的作用是 **GroupedGEMM**。

MoE routing 后，token 会按 expert 分组：

```text
expert 0: 10 个 token
expert 1: 3 个 token
expert 2: 21 个 token
expert 3: 0 个 token
...
```

每个 expert 都要做一个 FFN GEMM：

```text
tokens_for_expert_i × W_expert_i
```

如果每个 expert 单独调用一次 GEMM，会有很多小 kernel，GPU 利用率差。CUTLASS GroupedGEMM 可以把它们打包：

```text
一次 kernel launch:
  GEMM(expert0)
  GEMM(expert1)
  GEMM(expert2)
  ...
```

这就是前面那条链：

```text
CUB radix sort:
  按 expert_id 给 token 排序分组

CUTLASS GroupedGEMM:
  一次性计算多个 expert 的 FFN GEMM
```

为什么不用 cuBLAS？因为 cuBLAS 对标准大 GEMM 很强，但 MoE 里是很多 **形状不同的小 GEMM**，还想融合低比特反量化、特殊 layout、epilogue 操作。CUTLASS 给你更细粒度控制。

一句话记：

```text
CUTLASS = NVIDIA 的 C++ 模板化高性能 GEMM/卷积 kernel 积木库。
在 MoE 里，它负责把多个 expert 的矩阵乘打包成高效 GroupedGEMM。
```

### 010 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- Q1.4_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.4_L1_answer.md
- SKILL.md: .claude/skills/obsidian-keyword-explain/SKILL.md
- SKILL.md: .codex/skills/obsidian-keyword-explainer/SKILL.md
- SKILL.md: .codex/skills/export-conversation-notes/SKILL.md

## My request for Codex:
你在调用obsidian get notes时遇到个是问题吗?请在skill中更新正确的使用规范(path)

### 011 Codex Final
对，问题就是 `obsidian_get_note.target` 不能传裸字符串，必须传带 `type` 的对象。

我已更新两处 skill：

- [.codex/skills/obsidian-keyword-explainer/SKILL.md](/data3/paper_analysis/.codex/skills/obsidian-keyword-explainer/SKILL.md:129)
- [.claude/skills/obsidian-keyword-explain/SKILL.md](/data3/paper_analysis/.claude/skills/obsidian-keyword-explain/SKILL.md:129)

现在规范里明确写了正确形式：

```json
{
  "format": "content",
  "target": {
    "type": "path",
    "path": "knowledge_notes/example.md"
  }
}
```

并标注了不要用 `"target": "..."` 或缺少 `type` 的 `{"path": "..."}`。

### 012 User
<environment_context>
  <current_date>2026-06-03</current_date>
  <timezone>Asia/Shanghai</timezone>
  <filesystem><workspace_roots><root>/data3/paper_analysis</root></workspace_roots><permission_profile type="managed"><file_system type="restricted"><entry access="read"><special>:root</special></entry><entry access="write"><path>/data3/paper_analysis</path></entry><entry access="write"><special>:slash_tmp</special></entry><entry access="write"><special>:tmpdir</special></entry><entry access="read"><path>/data3/paper_analysis/.git</path></entry><entry access="read"><path>/data3/paper_analysis/.agents</path></entry><entry access="read"><path>/data3/paper_analysis/.codex</path></entry></file_system></permission_profile></filesystem>
</environment_context>

### 013 User
# Context from my IDE setup:

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- L1_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L1_horizon_summary.md
- SKILL.md: .codex/skills/export-conversation-notes/SKILL.md
- Accelerating Sparse Transformer Inference on GPU (STOF).md: human_notes/Accelerating Sparse Transformer Inference on GPU (STOF).md
- Adaptive Draft Sequence Length_ Enhancing Speculative Decoding Throughput on PIM-Enabled Systems.md: human_notes/Adaptive Draft Sequence Length_ Enhancing Speculative Decoding Throughput on PIM-Enabled Systems.md

## My request for Codex:
[$export-conversation-notes](/data3/paper_analysis/.codex/skills/export-conversation-notes/SKILL.md) 

### 014 User
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
