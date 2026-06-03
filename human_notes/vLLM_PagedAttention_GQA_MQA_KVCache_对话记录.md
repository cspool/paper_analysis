# vLLM PagedAttention GQA MQA KVCache 对话记录

- 导出时间：2026-06-03 17:24 CST
- 来源：当前 Codex session 可用对话上下文
- 保存路径：/data3/paper_analysis/human_notes/vLLM_PagedAttention_GQA_MQA_KVCache_对话记录.md
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
### 方法1: vLLM — PagedAttention + Continuous Batching

**笔记证据**: `experiment_notes/系统实验笔记/A Survey of Resource-efficient LLM and Multimodal Foundation Models.md` (score: 1287.6); `experiment_notes/系统实验笔记/Shift Parallelism...md` (score: 2658.0)

**方法细节**（L1 算法 Pipeline 粒度 —— 带张量形状的伪代码）:

vLLM 的核心是 PagedAttention——将 KV cache 按 OS 虚拟内存方式以固定大小 block 分页管理，消除内存碎片。

```
算法: vLLM PagedAttention 推理 Pipeline
=========================================
输入: requests[], model_weights, max_seq_len, block_size=16
输出: generated_tokens[][]
初始化: Block_Manager 将 GPU HBM 划分为固定大小物理 block

1. 调度器 (Iteration-Level):
   while has_pending_requests(requests):
       // Orca-style: 选择本轮处理的请求, 无 padding
       scheduled = Scheduler.select_requests(requests)

2. KV Cache Block 管理:
   for req in scheduled:
       if req.current_block_full():
           new_block = Block_Manager.allocate()
             // 逻辑 block → 物理 block 映射 (类似 OS 页表)
           req.block_table.append(new_block)

3. Model Runner — 逐层 Transformer 执行:
   input_h: [B, hidden_dim]  // B=本轮 batch token 数

   for layer ℓ in 0..L-1:
       // 3a: Attention (PagedAttention kernel)
       // Q: [num_heads, head_dim], block_table: [max_blocks]
       // KV_cache: [num_blocks, block_size, num_kv_heads, head_dim]
       for each query token q_i:
           o_i = 0  // [head_dim] 累加器
           running_max = -inf; running_sum = 0
           for block_id in block_table[req_id]:
               K_block = KV_cache_k[block_id]  // [16, head_dim], HBM→SMEM
               V_block = KV_cache_v[block_id]  // [16, head_dim]
               scores = q_i @ K_block^T / sqrt(head_dim)  // [16]
               // Online Safe Softmax
               m_new = max(running_max, max(scores))
               running_sum = running_sum*exp(running_max-m_new) + sum(exp(scores-m_new))
               o_i = o_i*exp(running_max-m_new) + softmax(scores) @ V_block
               running_max = m_new
           o_i = o_i / running_sum

       // 3b: MoE Expert Routing (仅 MoE 层)
       if is_moe_layer:
           gate_logits = Router(attn_out)      // [1, num_experts]
           topk_e, topk_w = TopK(softmax(gate_logits), K=2)
           output = Σ_{e} topk_w[e] * ExpertFFN[e](attn_out)
       else:
           output = FFN(attn_out)

4. 采样: next_token = Sample(lm_head(output)) → 追加 KV cache
```

**注解**:
- **变量含义**: `block_size`=16/32 是 KV cache 管理最小粒度；`block_table` 类似 OS 页表维护逻辑→物理映射；`HBM→SMEM` 表示从全局显存加载到 shared memory
- **复杂度**: 标准 KV cache 预分配 `max_seq_len` 连续内存→利用率 ~20-30%；PagedAttention 按需分配→利用率 ~96%，消除碎片
- **数据依赖**: 每 token 的 attention 依赖其所有历史 KV blocks，但请求间无依赖→天然批处理友好
- **硬件适配**: PagedAttention kernel 利用 A100/H100 HBM 高带宽 (2.0/3.35 TB/s)，block 级访问通过指针跳转实现
- **显存规划**: H100 80GB HBM 可容纳 ~40K blocks @ block_size=16, Llama-70B→可同时服务 ~128 个并发请求

**实验环境**:
- **硬件**: NVIDIA H200 (141 GB HBM, 4.8 TB/s, FP8 1979 TFLOPS)，8×H200 NVSwitch 900 GB/s；H100/A100
- **模型**: Llama-3.3-70B-FP8, Mixtral-8x7B；**benchmark**: ShareGPT, production traces
- **关键指标**: TTFT 148ms, TPOT 51ms, peak throughput 69,147 tok/s (Llama-70B 4k/250); KV cache 利用率 ~96%
- **硬件适配**: vLLM 的 CUDA Graph capture 消除 decode 阶段 kernel launch overhead (~5μs→~0.5μs per kernel)，在 H200 上达到 near-roofline memory BW 利用率
## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- .mcp.json: .mcp.json
- settings.json: .claude/settings.json
- settings.local.json: .claude/settings.local.json
- review_on_learning_outputs_20260529_185938.md: human_notes/review_on_learning_outputs_20260529_185938.md

## My request for Codex:
教我,然后解释伪代码:'### 方法1: vLLM — PagedAttention + Continuous Batching

**笔记证据**: `experiment_notes/系统实验笔记/A Survey of Resource-efficient LLM and Multimodal Foundation Models.md` (score: 1287.6); `experiment_notes/系统实验笔记/Shift Parallelism...md` (score: 2658.0)

**方法细节**（L1 算法 Pipeline 粒度 —— 带张量形状的伪代码）:

vLLM 的核心是 PagedAttention——将 KV cache 按 OS 虚拟内存方式以固定大小 block 分页管理，消除内存碎片。

```
算法: vLLM PagedAttention 推理 Pipeline
=========================================
输入: requests[], model_weights, max_seq_len, block_size=16
输出: generated_tokens[][]
初始化: Block_Manager 将 GPU HBM 划分为固定大小物理 block

1. 调度器 (Iteration-Level):
   while has_pending_requests(requests):
       // Orca-style: 选择本轮处理的请求, 无 padding
       scheduled = Scheduler.select_requests(requests)

2. KV Cache Block 管理:
   for req in scheduled:
       if req.current_block_full():
           new_block = Block_Manager.allocate()
             // 逻辑 block → 物理 block 映射 (类似 OS 页表)
           req.block_table.append(new_block)

3. Model Runner — 逐层 Transformer 执行:
   input_h: [B, hidden_dim]  // B=本轮 batch token 数

   for layer ℓ in 0..L-1:
       // 3a: Attention (PagedAttention kernel)
       // Q: [num_heads, head_dim], block_table: [max_blocks]
       // KV_cache: [num_blocks, block_size, num_kv_heads, head_dim]
       for each query token q_i:
           o_i = 0  // [head_dim] 累加器
           running_max = -inf; running_sum = 0
           for block_id in block_table[req_id]:
               K_block = KV_cache_k[block_id]  // [16, head_dim], HBM→SMEM
               V_block = KV_cache_v[block_id]  // [16, head_dim]
               scores = q_i @ K_block^T / sqrt(head_dim)  // [16]
               // Online Safe Softmax
               m_new = max(running_max, max(scores))
               running_sum = running_sum*exp(running_max-m_new) + sum(exp(scores-m_new))
               o_i = o_i*exp(running_max-m_new) + softmax(scores) @ V_block
               running_max = m_new
           o_i = o_i / running_sum

       // 3b: MoE Expert Routing (仅 MoE 层)
       if is_moe_layer:
           gate_logits = Router(attn_out)      // [1, num_experts]
           topk_e, topk_w = TopK(softmax(gate_logits), K=2)
           output = Σ_{e} topk_w[e] * ExpertFFN[e](attn_out)
       else:
           output = FFN(attn_out)

4. 采样: next_token = Sample(lm_head(output)) → 追加 KV cache
```

**注解**:
- **变量含义**: `block_size`=16/32 是 KV cache 管理最小粒度；`block_table` 类似 OS 页表维护逻辑→物理映射；`HBM→SMEM` 表示从全局显存加载到 shared memory
- **复杂度**: 标准 KV cache 预分配 `max_seq_len` 连续内存→利用率 ~20-30%；PagedAttention 按需分配→利用率 ~96%，消除碎片
- **数据依赖**: 每 token 的 attention 依赖其所有历史 KV blocks，但请求间无依赖→天然批处理友好
- **硬件适配**: PagedAttention kernel 利用 A100/H100 HBM 高带宽 (2.0/3.35 TB/s)，block 级访问通过指针跳转实现
- **显存规划**: H100 80GB HBM 可容纳 ~40K blocks @ block_size=16, Llama-70B→可同时服务 ~128 个并发请求

**实验环境**:
- **硬件**: NVIDIA H200 (141 GB HBM, 4.8 TB/s, FP8 1979 TFLOPS)，8×H200 NVSwitch 900 GB/s；H100/A100
- **模型**: Llama-3.3-70B-FP8, Mixtral-8x7B；**benchmark**: ShareGPT, production traces
- **关键指标**: TTFT 148ms, TPOT 51ms, peak throughput 69,147 tok/s (Llama-70B 4k/250); KV cache 利用率 ~96%
- **硬件适配**: vLLM 的 CUDA Graph capture 消除 decode 阶段 kernel launch overhead (~5μs→~0.5μs per kernel)，在 H200 上达到 near-roofline memory BW 利用率'

### 003 Codex Final
可以把这段伪代码理解成一句话：

vLLM 在每一轮 decode 里，把很多请求的“下一个 token 计算”拼成一个动态 batch；同时把每个请求的历史 KV cache 像操作系统分页一样按 block 存在显存里，attention 时按页表去找历史 K/V。

**1. 先抓住问题**
普通 LLM 推理有两个大麻烦：

第一，KV cache 很占显存。  
每生成一个 token，每层 attention 都会保存这个 token 的 Key/Value。序列越长，KV 越大。

第二，不同请求长度不同。  
如果按传统 batch，把短请求 padding 到长请求长度，会浪费算力和显存。请求来了又走，KV cache 还会产生碎片。

vLLM 的核心就是：

```text
Continuous Batching: 每一轮动态挑选还能继续生成的请求
PagedAttention: KV cache 不要求连续内存，而是按 block 分页存
```

---

**2. 初始化：Block_Manager 是显存里的“内存管理员”**

```text
初始化: Block_Manager 将 GPU HBM 划分为固定大小物理 block
```

意思是：

GPU 显存 HBM 里专门划出一大片区域放 KV cache。  
这片区域被切成很多固定大小的物理块，例如每块存 16 个 token 的 KV。

类似操作系统：

```text
逻辑 token 序列:
req1: token 0~15, token 16~31, token 32~47 ...

物理显存 block:
block 7, block 103, block 12 ...
```

它们不需要连续，只要有一张 `block_table` 记录映射即可。

---

**3. 调度器：每轮只算“当前该算的 token”**

```text
while has_pending_requests(requests):
    scheduled = Scheduler.select_requests(requests)
```

这就是 continuous batching。

假设现在有 3 个请求：

```text
A: 已经生成到第 20 个 token
B: 刚进来，需要 prefill prompt
C: 已经生成到第 150 个 token
```

vLLM 不会等它们对齐，也不会固定一个 batch 到结束。每一轮它重新选一批请求：

```text
第 t 轮: A decode 1 token + C decode 1 token + B prefill 若干 token
第 t+1 轮: A 结束了，换 D 进来，C 继续，B 进入 decode
```

所以 batch 是动态流动的。

---

**4. KV Cache Block 管理：满一页就申请新页**

```text
for req in scheduled:
    if req.current_block_full():
        new_block = Block_Manager.allocate()
        req.block_table.append(new_block)
```

每个请求都有自己的 `block_table`。

比如 `block_size=16`，某个请求已经生成了 16 个 token，那么当前 block 满了。接下来生成第 17 个 token 时，需要申请一个新 block。

可以想象成：

```text
req.block_table = [5, 88, 21]
```

表示这个请求的历史 KV 分布在物理 block 5、88、21 里。虽然物理地址不连续，但逻辑上它们组成了这个请求的完整上下文。

这就是 PagedAttention 解决碎片的关键。

---

**5. Model Runner：真正跑 Transformer**

```text
input_h: [B, hidden_dim]
```

这里的 `B` 不是传统意义上的“请求数”，而是“本轮要处理的 token 数”。

例如这一轮有：

```text
请求 A decode 1 个 token
请求 B decode 1 个 token
请求 C prefill 32 个 token
```

那么 `B` 可能是 `1 + 1 + 32 = 34`。

`hidden_dim` 是模型隐藏维度，例如 Llama 70B 可能是 8192 这一类量级。

---

**6. 逐层执行 Transformer**

```text
for layer ℓ in 0..L-1:
```

LLM 有很多层 Transformer block。每一层都要做：

```text
Attention
FFN / MoE FFN
```

如果是 Mixtral 这类 MoE 模型，中间 FFN 会变成专家路由。

---

**7. PagedAttention：按 block 读历史 KV**

核心代码是这段：

```text
for each query token q_i:
    o_i = 0
    running_max = -inf
    running_sum = 0

    for block_id in block_table[req_id]:
        K_block = KV_cache_k[block_id]
        V_block = KV_cache_v[block_id]
        scores = q_i @ K_block^T / sqrt(head_dim)
        ...
```

意思是：对当前 token 的 query `q_i`，它要和自己请求里的所有历史 token 做 attention。

普通 attention 是：

```text
scores = q @ K_all^T
output = softmax(scores) @ V_all
```

PagedAttention 不要求 `K_all` 和 `V_all` 连续存放，而是分 block 读取：

```text
for 每个历史 KV block:
    读 K_block, V_block
    算当前 query 对这一页历史 token 的注意力
    累加结果
```

所以 `block_table[req_id]` 就是“这个请求的历史 KV 在哪里”。

---

**8. 张量形状怎么读**

```text
Q: [num_heads, head_dim]
KV_cache: [num_blocks, block_size, num_kv_heads, head_dim]
```

解释一下：

`num_blocks`：显存里总共有多少个 KV block。  
`block_size`：每个 block 存多少个 token，比如 16。  
`num_kv_heads`：K/V head 数量，GQA/MQA 下可能小于 `num_heads`。  
`head_dim`：每个 head 的维度，比如 128。

一块 KV 大概长这样：

```text
K_block: [16, head_dim]
V_block: [16, head_dim]
```

也就是这一页里 16 个历史 token 的 K 和 V。

---

**9. Online Safe Softmax 在干什么**

这段容易晕：

```text
m_new = max(running_max, max(scores))
running_sum = running_sum*exp(running_max-m_new) + sum(exp(scores-m_new))
o_i = o_i*exp(running_max-m_new) + softmax(scores) @ V_block
running_max = m_new
```

它想解决的问题是：attention 的 softmax 要对所有历史 token 一起归一化，但现在历史 KV 是一块一块读的，不能一次性把所有 scores 都摊开。

所以它用 online softmax：

```text
一边读 block
一边维护全局 max
一边维护全局 sum
一边累加 attention output
```

`running_max` 用来防止指数爆炸。  
`running_sum` 是 softmax 分母。  
`o_i` 是加权后的 Value 累加器。

更直观地说，它最终等价于：

```text
o_i = softmax(q_i @ K_all^T) @ V_all
```

只是实现上分 block 做。

注意：你这份伪代码里的这一行略简化：

```text
o_i = o_i*exp(running_max-m_new) + softmax(scores) @ V_block
```

严格实现里，当前 block 的贡献也要按照同一个全局归一化尺度累计，通常会用 `exp(scores - m_new)` 这样的未归一化权重来累加，最后再除以 `running_sum`。所以它表达的是思想，不是可直接运行的精确 kernel 代码。

---

**10. MoE Expert Routing：只在 MoE 层出现**

```text
if is_moe_layer:
    gate_logits = Router(attn_out)
    topk_e, topk_w = TopK(softmax(gate_logits), K=2)
    output = Σ topk_w[e] * ExpertFFN[e](attn_out)
else:
    output = FFN(attn_out)
```

如果模型是普通 dense Transformer，比如 Llama：

```text
output = FFN(attn_out)
```

如果模型是 MoE，比如 Mixtral：

```text
Router 先判断这个 token 应该走哪些专家
通常选 top-2 experts
每个专家各自算 FFN
最后按路由权重加权求和
```

关键点：PagedAttention 主要管 attention 阶段的 KV cache；MoE routing 是 FFN 阶段的计算方式。它们是两件事，只是在同一个推理 pipeline 里发生。

---

**11. 采样与追加 KV**

```text
next_token = Sample(lm_head(output)) → 追加 KV cache
```

最后一层输出后，通过 `lm_head` 得到词表 logits：

```text
logits: [vocab_size]
```

然后根据采样策略选下一个 token：

```text
greedy / top-k / top-p / temperature
```

选出来的新 token 会进入请求序列。下一轮 decode 时，这个 token 对应的 K/V 也会被追加进 KV cache。

---

**12. 一句话总结这段伪代码**

这段代码描述的是：

```text
vLLM 每一轮动态挑选请求组成 batch；
每个请求的 KV cache 按 block 存在显存中；
attention 时通过 block_table 找到该请求的历史 KV；
逐 block 计算 attention，并用 online softmax 合并；
最后经过 FFN/MoE、lm_head、采样，生成下一个 token。
```

最核心要记住两个词：

```text
Continuous Batching = 请求调度效率
PagedAttention = KV cache 显存效率
```

### 004 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- .mcp.json: .mcp.json
- settings.json: .claude/settings.json
- settings.local.json: .claude/settings.local.json
- review_on_learning_outputs_20260529_185938.md: human_notes/review_on_learning_outputs_20260529_185938.md

# Selected text:

## Selection 1
Q: [num_heads, head_dim]
KV_cache: [num_blocks, block_size, num_kv_heads, head_dim]
解释一下：

num_blocks：显存里总共有多少个 KV block。
block_size：每个 block 存多少个 token，比如 16。
num_kv_heads：K/V head 数量，GQA/MQA 下可能小于 num_heads。
head_dim：每个 head 的维度，比如 128。

一块 KV 大概长这样：

text


K_block: [16, head_dim]
V_block: [16, head_dim]
也就是这一页里 16 个历史 token 的 K 和 V。

## My request for Codex:
细说gqa,mqa,head-dim不等于h-dim?

### 005 Codex Final
对，`head_dim` **不等于** `hidden_dim`。通常关系是：

```text
hidden_dim = num_heads * head_dim
head_dim = hidden_dim / num_heads
```

比如 Llama hidden size 是 4096，attention heads 是 32：

```text
hidden_dim = 4096
num_heads = 32
head_dim = 4096 / 32 = 128
```

所以一个 token 的 hidden state 是：

```text
x: [hidden_dim] = [4096]
```

进入 attention 后被拆成多个 head：

```text
Q: [num_heads, head_dim] = [32, 128]
```

拼起来仍然是：

```text
32 * 128 = 4096
```

---

**MHA：Multi-Head Attention**

最标准的多头注意力：

```text
num_q_heads = num_kv_heads
```

例如：

```text
Q: [32, 128]
K: [32, 128]
V: [32, 128]
```

每个 Q head 都有自己独立的一组 K/V head。

第 0 个 Q head 看第 0 个 K/V head，  
第 1 个 Q head 看第 1 个 K/V head，  
一直到第 31 个。

优点是表达能力强。  
缺点是 KV cache 很大。

---

**MQA：Multi-Query Attention**

MQA 是极端压缩版：

```text
num_q_heads = 32
num_kv_heads = 1
```

形状变成：

```text
Q: [32, 128]
K: [1, 128]
V: [1, 128]
```

意思是：32 个 Q head 共享同一组 K/V。

```text
Q head 0  \
Q head 1   \
Q head 2    -> same K/V head
...        /
Q head 31 /
```

这样 KV cache 直接缩小约 32 倍，因为以前每个 token 每层要存 32 组 K/V，现在只存 1 组。

缺点是所有 query heads 共享 K/V，表达能力可能下降。

---

**GQA：Grouped-Query Attention**

GQA 是 MHA 和 MQA 的折中：

```text
num_q_heads = 32
num_kv_heads = 8
```

形状是：

```text
Q: [32, 128]
K: [8, 128]
V: [8, 128]
```

也就是每 4 个 Q heads 共享 1 个 K/V head：

```text
Q heads 0,1,2,3      -> KV head 0
Q heads 4,5,6,7      -> KV head 1
Q heads 8,9,10,11    -> KV head 2
...
Q heads 28,29,30,31  -> KV head 7
```

公式是：

```text
group_size = num_q_heads / num_kv_heads
kv_head_id = q_head_id // group_size
```

比如 `num_q_heads=32, num_kv_heads=8`：

```text
group_size = 4
q_head 13 -> kv_head 13 // 4 = 3
```

---

所以三者对比：

```text
MHA: num_q_heads = 32, num_kv_heads = 32
GQA: num_q_heads = 32, num_kv_heads = 8
MQA: num_q_heads = 32, num_kv_heads = 1
```

KV cache 大小大约按 `num_kv_heads` 成正比：

```text
MHA: 最大
GQA: 中等
MQA: 最小
```

---

放回你那段伪代码：

```text
KV_cache: [num_blocks, block_size, num_kv_heads, head_dim]
```

如果是 MHA：

```text
KV_cache: [num_blocks, 16, 32, 128]
```

如果是 GQA：

```text
KV_cache: [num_blocks, 16, 8, 128]
```

如果是 MQA：

```text
KV_cache: [num_blocks, 16, 1, 128]
```

`head_dim` 还是 128，不会因为 GQA/MQA 变成 hidden_dim。变的是 `num_kv_heads`，不是 `head_dim`。

最核心一句：

```text
hidden_dim 是整个 token 向量的宽度；
head_dim 是一个 attention head 的宽度；
GQA/MQA 减少的是 K/V head 的数量，不是每个 head 的维度。
```

### 006 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Active selection of the file:
        <!-- ! hidden_dim = num_heads * head_dim. -->
        <!-- ? MHA Decoder:token(hidden-dim)进入MHA(M个head)后,拆分后计算M个q(head-dim),M个k和M个v(head-dim) ,读取M组KV-Cache(head-dim),计算M组(i=1...M)qi*(Ki,ki)和si*(Vi,vi),将M个ki和vi加入KV-Cache. -->
        <!-- ? MHA Prefill:L个tokens计算M个Q(L,head-dim),M个K(L,head-dim),M个V(L,head-dim),计算M组Q*K和S*V得到token序列输出,将K和V加入KV-Cache.  -->
              
        <!-- ! MQA:M组query共享1组KV.  -->
        <!-- ? MQA:token拆分后计算M个q(head-dim),1个k和1个v(head-dim),读取1组KV-Cache(head-dim),计算M组(i=1...M)qi*(K,k)和si*(V,v),将新的k和v加入KV-Cache  -->
        <!-- ! GQA:M组query共享m组KV.  -->
        <!-- ? GQA:token拆分后计算M个q(head-dim),m个k和m个v(head-dim),读取m组KV-Cache(head-dim),计算M组(i=1...M,j=1...m)qi*(Kj,kj)和si*(Vj,vj),将m组k和v加入KV-Cache -->
## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- .mcp.json: .mcp.json
- settings.json: .claude/settings.json
- settings.local.json: .claude/settings.local.json
- review_on_learning_outputs_20260529_185938.md: human_notes/review_on_learning_outputs_20260529_185938.md

## My request for Codex:
        <!-- ! hidden_dim = num_heads * head_dim. -->
        <!-- ? MHA Decoder:token(hidden-dim)进入MHA(M个head)后,拆分后计算M个q(head-dim),M个k和M个v(head-dim) ,读取M组KV-Cache(head-dim),计算M组(i=1...M)qi*(Ki,ki)和si*(Vi,vi),将M个ki和vi加入KV-Cache. -->
        <!-- ? MHA Prefill:L个tokens计算M个Q(L,head-dim),M个K(L,head-dim),M个V(L,head-dim),计算M组Q*K和S*V得到token序列输出,将K和V加入KV-Cache.  -->
              
        <!-- ! MQA:M组query共享1组KV.  -->
        <!-- ? MQA:token拆分后计算M个q(head-dim),1个k和1个v(head-dim),读取1组KV-Cache(head-dim),计算M组(i=1...M)qi*(K,k)和si*(V,v),将新的k和v加入KV-Cache  -->
        <!-- ! GQA:M组query共享m组KV.  -->
        <!-- ? GQA:token拆分后计算M个q(head-dim),m个k和m个v(head-dim),读取m组KV-Cache(head-dim),计算M组(i=1...M,j=1...m)qi*(Kj,kj)和si*(Vj,vj),将m组k和v加入KV-Cache -->

### 007 Codex Final
你的笔记大体是对的，关键只需要改一个地方：**GQA 不是 M 个 query head 和 m 个 KV head 两两相乘**，而是 **每个 query head 被分配到某一个 KV head**。

也就是这里：

```text
计算M组(i=1...M,j=1...m) qi*(Kj,kj)
```

应改成：

```text
计算M组 attention：第 i 个 q_head 只使用 kv_head_id = i // group_size 对应的 K/V
```

其中：

```text
group_size = M / m
```

---

可以把你的笔记改成这样：

```html
<!-- ! hidden_dim = num_q_heads * head_dim. -->

<!-- ? MHA Decode:
token(hidden_dim) 进入 Attention 后，
计算 M 个 q(head_dim), M 个 k(head_dim), M 个 v(head_dim)。
第 i 个 q_i 读取第 i 组历史 KV-cache：
K_cache_i: [past_len, head_dim]
V_cache_i: [past_len, head_dim]
计算 scores_i = q_i @ K_cache_i^T -> [past_len]
计算 out_i = softmax(scores_i) @ V_cache_i -> [head_dim]
最后将当前 token 的 M 个 k_i/v_i 追加进 KV-cache。
-->

<!-- ? MHA Prefill:
L 个 tokens 计算：
Q: [M, L, head_dim]
K: [M, L, head_dim]
V: [M, L, head_dim]
每个 head 内计算：
scores_i = Q_i @ K_i^T -> [L, L]
out_i = softmax(mask(scores_i)) @ V_i -> [L, head_dim]
最后将 K/V: [M, L, head_dim] 写入 KV-cache。
-->

<!-- ! MQA: M 个 query heads 共享 1 组 KV head。 -->

<!-- ? MQA Decode:
token(hidden_dim) 计算 M 个 q(head_dim)，但只计算 1 个 k(head_dim) 和 1 个 v(head_dim)。
所有 q_i 都读取同一组历史 KV-cache：
K_cache: [past_len, head_dim]
V_cache: [past_len, head_dim]
每个 q_i 分别计算：
scores_i = q_i @ K_cache^T
out_i = softmax(scores_i) @ V_cache
最后只追加 1 个 k/v 到 KV-cache。
-->

<!-- ! GQA: M 个 query heads 分组共享 m 组 KV heads。MQA 是 m=1 的特例，MHA 是 m=M 的特例。 -->

<!-- ? GQA Decode:
token(hidden_dim) 计算 M 个 q(head_dim)，m 个 k(head_dim)，m 个 v(head_dim)。
每 group_size = M/m 个 query heads 共享 1 组 KV。
第 i 个 q_head 使用：
kv_head_id = i // group_size
即：
q_i 只和 K_cache[kv_head_id], V_cache[kv_head_id] 做 attention。
总共仍然计算 M 组 attention outputs，不是 M*m 组。
最后追加 m 个 k/v 到 KV-cache。
-->
```

最重要的关系是：

```text
MHA: num_q_heads = M, num_kv_heads = M
GQA: num_q_heads = M, num_kv_heads = m
MQA: num_q_heads = M, num_kv_heads = 1
```

所以 KV cache 大小约正比于：

```text
num_kv_heads * head_dim
```

GQA/MQA 省显存，是因为减少了 `num_kv_heads`，不是减少了 `head_dim`。

### 008 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- .mcp.json: .mcp.json
- settings.json: .claude/settings.json
- settings.local.json: .claude/settings.local.json
- review_on_learning_outputs_20260529_185938.md: human_notes/review_on_learning_outputs_20260529_185938.md

## My request for Codex:
每个token作decoding'时,新生成的k和v不参与当前token的计算吗?

### 009 Codex Final
要分清两个“当前 token”：

**情况 1：当前正在 forward 的输入 token**
例如现在已经有序列：

```text
x1, x2, ..., xt
```

这一轮 decode 输入的是 `xt`，目标是预测 `x_{t+1}`。

在这一轮里，每一层会先为 `xt` 计算：

```text
q_t, k_t, v_t
```

然后 attention 通常是：

```text
out_t = Attention(q_t, K_cache[1:t], V_cache[1:t])
```

这里的 `K_cache[1:t] / V_cache[1:t]` **包含当前 token xt 自己的 k_t/v_t**。

也就是说：

```text
当前输入 token 的新 k/v 会参与当前 token 的 attention 计算。
```

因为 causal self-attention 允许当前位置看自己和之前的位置：

```text
token t 可以看 token 1...t
不能看 token t+1...
```

---

**情况 2：刚刚被采样出来的新 token**
这一轮 forward 结束后，经过：

```text
logits_t = lm_head(out_t)
x_{t+1} = Sample(logits_t)
```

此时 `x_{t+1}` 刚刚被生成出来。

它的 `k_{t+1}, v_{t+1}` **还没有计算**，所以当然不参与这一轮计算。它要等下一轮 decode：

```text
下一轮输入 x_{t+1}
计算 q_{t+1}, k_{t+1}, v_{t+1}
attention 看 K/V[1:t+1]
预测 x_{t+2}
```

---

所以一句话：

```text
正在被处理的输入 token 的 k/v 会参与当前 attention；
刚采样出来的输出 token 的 k/v 不会参与当前计算，要到下一轮才计算。
```

时间线是：

```text
已知 x1...xt
输入 xt
计算 q_t, k_t, v_t
attention 使用 K/V[1:t]
输出 logits
采样得到 x_{t+1}

下一轮：
输入 x_{t+1}
计算 q_{t+1}, k_{t+1}, v_{t+1}
attention 使用 K/V[1:t+1]
输出 logits
采样得到 x_{t+2}
```

### 010 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- .mcp.json: .mcp.json
- settings.json: .claude/settings.json
- settings.local.json: .claude/settings.local.json
- review_on_learning_outputs_20260529_185938.md: human_notes/review_on_learning_outputs_20260529_185938.md

# Selected text:

## Selection 1
<!-- ? MHA Decode:
token(hidden_dim) 进入 Attention 后，
计算 M 个 q(head_dim), M 个 k(head_dim), M 个 v(head_dim)。
第 i 个 q_i 读取第 i 组历史 KV-cache：
K_cache_i: [past_len, head_dim]
V_cache_i: [past_len, head_dim]
计算 scores_i = q_i @ K_cache_i^T -> [past_len]
计算 out_i = softmax(scores_i) @ V_cache_i -> [head_dim]
最后将当前 token 的 M 个 k_i/v_i 追加进 KV-cache。

## My request for Codex:
你这里的描述正确吗?输入token产生k和没有参与计算??

### 011 Codex Final
你抓得对：我那段描述**不够准确**，容易让人误解成“当前输入 token 的 k/v 先不参与计算，只是最后追加”。更严谨的 decode 描述应该是：

```html
<!-- ? MHA Decode:
假设当前输入 token 是 x_t，已有历史 KV-cache 存着 x_1 ... x_{t-1} 的 K/V。

x_t(hidden_dim) 进入 Attention 后，
计算 M 个 q_t(head_dim), M 个 k_t(head_dim), M 个 v_t(head_dim)。

对第 i 个 head：
先把当前 token 的 k_{t,i}/v_{t,i} 与历史 cache 拼起来：
K_all_i = concat(K_cache_i[1:t-1], k_{t,i}) -> [t, head_dim]
V_all_i = concat(V_cache_i[1:t-1], v_{t,i}) -> [t, head_dim]

然后计算：
scores_i = q_{t,i} @ K_all_i^T -> [t]
out_i = softmax(scores_i) @ V_all_i -> [head_dim]

最后将当前 token 的 M 个 k_t/v_t 追加进 KV-cache，
供下一轮 token x_{t+1} 使用。
-->
```

所以结论是：

```text
当前输入 token x_t 产生的 k_t/v_t 会参与当前 token 的 attention。
```

之前那句：

```text
第 i 个 q_i 读取第 i 组历史 KV-cache
```

应该改成：

```text
第 i 个 q_i 读取历史 KV-cache + 当前 token 的 k_i/v_i
```

不过实现上有两种写法：

```text
写法 A：先把当前 k/v append 到 cache，再从 cache 读 K/V[1:t]
写法 B：读历史 cache K/V[1:t-1]，再临时拼上当前 k/v
```

数学上等价。你的质疑是对的，这里必须把“当前输入 token 的 k/v 参与当前 attention”写清楚。

### 012 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- .mcp.json: .mcp.json
- settings.json: .claude/settings.json
- settings.local.json: .claude/settings.local.json
- review_on_learning_outputs_20260529_185938.md: human_notes/review_on_learning_outputs_20260529_185938.md

# Selected text:

## Selection 1
当前输入 token 的 k/v 参与当前 attention”

## My request for Codex:
这是对的吗?请找出证据

### 013 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- .mcp.json: .mcp.json
- settings.json: .claude/settings.json
- settings.local.json: .claude/settings.local.json
- review_on_learning_outputs_20260529_185938.md: human_notes/review_on_learning_outputs_20260529_185938.md

## My request for Codex:
继续

### 014 Codex Final
是对的：**当前输入 token `x_t` 产生的 `k_t/v_t` 会参与当前这一步的 self-attention**。但你要区分它和“刚采样出来的下一个 token `x_{t+1}`”。

证据很直接：

Hugging Face 官方 caching 文档写的是，在推理时用最后一个 token 的 query 计算当前表示，attention 形式是：

```text
Attention(q_t, [cached k_1, ..., k_{t-1}, k_t],
             [cached v_1, ..., v_{t-1}, v_t])
```

也就是 `k_t/v_t` 明确在参与当前 attention。文档还给出 cache 接口顺序：

```text
new_K, new_V = cache.update(k_t, v_t, layer_idx)
attn_output = attn_layer_idx_fn(q_t, new_K, new_V)
```

来源：Hugging Face Caching 文档第 128-150 行  
https://huggingface.co/docs/transformers/cache_explanation

vLLM 代码证据也类似。它在 decoder attention 里先把当前 `key/value` 写入 `key_cache/value_cache`：

```text
ops.cpu_attn_reshape_and_cache(key, value, key_cache, value_cache, ...)
```

然后再调用：

```text
ops.cpu_attention_with_kv_cache(
    query=query,
    key_cache=key_cache,
    value_cache=value_cache,
    ...
)
```

也就是当前 token 的 `key/value` 被写进 KV cache 后，attention kernel 从 KV cache 读来计算。  
来源：vLLM CPU attention backend 第 3031-3128 行  
https://docs.vllm.ai/en/stable/api/vllm/v1/attention/backends/cpu_attn/

所以正确时间线是：

```text
已有 cache: k_1/v_1 ... k_{t-1}/v_{t-1}

当前输入 x_t
-> 计算 q_t, k_t, v_t
-> K_all = [k_1 ... k_{t-1}, k_t]
-> V_all = [v_1 ... v_{t-1}, v_t]
-> out_t = softmax(q_t @ K_all^T) @ V_all
-> logits_t 预测 x_{t+1}
```

但 **刚采样出来的 `x_{t+1}` 的 k/v 不参与这一轮**，因为它还没有 forward。它要到下一轮作为输入 token 时，才计算 `q_{t+1}, k_{t+1}, v_{t+1}`。

### 015 User
<environment_context>
  <current_date>2026-06-03</current_date>
  <timezone>Asia/Shanghai</timezone>
  <filesystem><workspace_roots><root>/data3/paper_analysis</root></workspace_roots><permission_profile type="managed"><file_system type="restricted"><entry access="read"><special>:root</special></entry><entry access="write"><path>/data3/paper_analysis</path></entry><entry access="write"><special>:slash_tmp</special></entry><entry access="write"><special>:tmpdir</special></entry><entry access="read"><path>/data3/paper_analysis/.git</path></entry><entry access="read"><path>/data3/paper_analysis/.agents</path></entry><entry access="read"><path>/data3/paper_analysis/.codex</path></entry></file_system></permission_profile></filesystem>
</environment_context>

### 016 User
# Context from my IDE setup:

## Open tabs:
- L1_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L1_horizon_summary.md
- SKILL.md: .codex/skills/export-conversation-notes/SKILL.md
- Accelerating Sparse Transformer Inference on GPU (STOF).md: human_notes/Accelerating Sparse Transformer Inference on GPU (STOF).md
- Adaptive Draft Sequence Length_ Enhancing Speculative Decoding Throughput on PIM-Enabled Systems.md: human_notes/Adaptive Draft Sequence Length_ Enhancing Speculative Decoding Throughput on PIM-Enabled Systems.md
- AdaServe_ Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding.md: human_notes/AdaServe_ Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding.md

## My request for Codex:
[$export-conversation-notes](/data3/paper_analysis/.codex/skills/export-conversation-notes/SKILL.md) 

### 017 User
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
