
## Speculative Decoding (投机解码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Speculative Decoding（投机解码）是一种 LLM 推理加速技术，由 Chen et al. (2023) 和 Leviathan et al. (2023) 独立提出。其核心思想是：用一个更小、更快的 draft model（草稿模型）来自回归地生成多个候选 token（draft tokens），然后由大而准确的目标模型（target model）在单次并行前向传播中验证所有这些 draft token。由于目标模型可以并行处理多个 token，验证的延迟接近于处理单个 token，而草稿模型生成每个 token 的速度远快于目标模型。投机解码已被数学证明生成的输出与标准自回归生成等价（lossless）。

标准流程（以 draft 长度 N=4 为例）：
1. Draft 阶段：draft model 自回归生成 N 个 draft token（N 次串行前向传播）
2. Verify 阶段：target model 输入 [last_real_token, draft_1, draft_2, draft_3, draft_4]，并行计算 logits
3. Accept/Reject：逐 token 比较 target logits 与 draft token——若 argmax(logits_j) == draft_{j-1} 则接受，否则在第一个不匹配处停止，丢弃后续所有 draft token
4. 将接受的 draft token + target 选择的下一个 token 作为下一轮输入

核心加速原理：target model 验证 N 个 draft token 的计算量 ≈ 处理 1 个 token 的计算量（因为 multi-token forward 的 batch 开销小），而 draft model 生成 N 个 token 的时间远小于 target model 生成 1 个 token 的时间。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 Agent-X 的 ExSpec 中，Speculative Decoding 被改造为使用 n-gram LUT 作为 draft model：

```
# ExSpec: n-gram LUT 驱动的 Speculative Decoding (trigram, n=3)
LUT = {}  # key: (token_{i-2}, token_{i-1}) → value: most_frequent_next_token

# LUT 构建 (每个 query 在线构建一次, 83ms)
extraction_stream = concat(few_shot_examples, user_query)
for i in range(2, len(extraction_stream)):
    key = (extraction_stream[i-2], extraction_stream[i-1])
    LUT[key][extraction_stream[i]] += 1  # 计数
for key in LUT:
    LUT[key] = argmax(LUT[key])  # 取最频繁后继 token

# Speculative Decoding Loop
output_tokens = []  # 已生成 token
while not done:
    ctx = (output_tokens[-2], output_tokens[-1])  # 最近 2-token 上下文
    
    # Step 1: Selective decision
    if ctx not in LUT:
        # 回退到标准自回归 (selective fallback)
        next_token = target_model.autoregressive(ctx)
        output_tokens.append(next_token)
        continue
    
    # Step 2: Draft generation (N=4)
    drafts = []
    cur_ctx = ctx
    for j in range(4):
        draft = LUT[cur_ctx]
        drafts.append(draft)
        cur_ctx = (cur_ctx[1], draft)
    
    # Step 3: Target model parallel verification
    input_seq = [output_tokens[-1]] + drafts  # 5 tokens
    target_logits = target_model.forward(input_seq)  # 一次并行 forward
    
    # Step 4: Token acceptance
    for j in range(4):
        if argmax(target_logits[j]) == drafts[j]:
            output_tokens.append(drafts[j])
        else:
            output_tokens.append(argmax(target_logits[j]))
            break  # 丢弃后续所有 draft
```

关键洞察：Agent-X 发现 agent 的 decode 输出与输入 prompt 高度相关（96% Planner 输出 token 和 87% Arbiter 输出 token 与 prompt 重叠），使得 n-gram LUT 能够以高准确率（selective: 0.25 draft accuracy）预测输出，远优于通用场景。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

**通用实现方式**（不限于 Agent-X）：
- HuggingFace Transformers: 使用 `assistant_model` 参数指定 draft model，或 `prompt_lookup_num_tokens` 使用 Prompt Lookup Decoding（基于 prompt 的 n-gram 匹配）
- vLLM: `--speculative-model [ngram]` 或指定 assistant model 路径
- TensorRT-LLM: 内置 n-gram speculative decoding，支持自动启用（AL > 1.37 on general chat）
- llama.cpp: 多种 n-gram 变体（`ngram-simple`, `ngram-map-k`, `ngram-cache`）
- EAGLE/Medusa: 使用训练的 draft head 而非独立 draft model

**Agent-X ExSpec 的特化实现**：
- Draft model: 仅从 few-shot examples + user query 构建 trigram LUT（KB 级内存，83ms 构建时间）
- Selective decoding: 首 token 生成前查 LUT——若不存在则立即回退自回归，避免 multi-token tax
- 在 MLX-LM 上的表现：131ms/token（单 token 自回归）vs 244ms（2 token 验证），multi-token tax = 1.86×
- Selective mode 使 Planner 平均 17 次/query、Arbiter 37 次/query 回退到自回归

涉及论文标题：
- Agent-X: Full Pipeline Acceleration of On-device AI Agents

## KV Cache (Key-Value Cache, 键值缓存)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

KV Cache 是 Transformer 自回归推理中的核心内存结构。在 causal attention 中，每个 token 的 Query 只关注自身及之前所有 token 的 Key 和 Value。生成新 token 时，之前所有 token 的 Key 和 Value 已经计算过，无需重复计算——将它们缓存在内存中即为 KV Cache。KV Cache 的大小随已处理的 token 数量线性增长：对于 L 层、H 个 attention head、d 维 head dim 的模型，每 token 的 KV Cache 大小为 $2 \times L \times H \times d \times \text{bytes_per_element}$。例如 TinyAgent-7B (32 层, 32 heads, d=128, FP16) 每 token 约 0.5 MB，1,739 token prompt 需约 0.87 GB KV Cache。

在 LLM 推理的两个阶段中：
- Prefill 阶段：处理所有输入 token，填充 KV Cache（compute-bound，计算密集）
- Decode 阶段：每次仅处理 1 个新 token，从 KV Cache 读取所有历史 K/V（memory bandwidth-bound，内存带宽受限）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Agent-X 中 KV Cache 作为 Prefix Caching 的核心载体：
```
# KV Cache 在 Attention 中的使用
def attention_with_kv_cache(Q_new, K_new, V_new, K_cache, V_cache):
    # 拼接历史 KV Cache 和新 token 的 K, V
    K_full = concat([K_cache, K_new])  # [seq_len, H, d]
    V_full = concat([V_cache, V_new])  # [seq_len, H, d]
    
    # 标准 scaled dot-product attention
    scores = Q_new @ K_full.T / sqrt(d)      # [1, seq_len]
    scores = causal_mask(scores)              # 仅关注当前位置及之前
    attn_weights = softmax(scores)
    output = attn_weights @ V_full            # [1, d]
    
    # 更新 KV Cache (追加新 token)
    K_cache = K_full   # 或写入预分配的 block
    V_cache = V_full
    return output

# Agent-X PromptWeaver 对 KV Cache 的创新使用
# 离线: 预计算并存储 KV Cache 到 SSD
kv_cache_all_tools = prefill_offline("[System] + [ALL 16 tools' descriptions]")
save_to_ssd("all_tools_kv", kv_cache_all_tools)    # 0.57 GB

# 在线: 按需加载
static_kv = load_from_ssd("all_tools_kv")          # SSD → 内存
cluster_kv = load_from_ssd(longest_prefix_match)    # 从预计算 cluster KV 中匹配
dynamic_kv = prefill_online(dynamic_tokens)         # 仅计算 ~519 不可缓存 token
full_kv = concat([static_kv, cluster_kv, dynamic_kv])
# 后续 decode 使用 full_kv
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

**通用实现方式**：
- 预分配策略：分配连续内存块，按最大序列长度预分配（如 max_seq_len=4096）
- PagedAttention (vLLM)：将 KV Cache 切分为固定大小 block（如 16 token），以页表管理，消除碎片
- RadixAttention (SGLang)：基于 radix tree 的 KV Cache 管理，支持前缀共享
- KV Cache 量化：KIVI、KVQuant 等方法对 KV Cache 进行 2-4 bit 量化以减少内存占用
- 卸载策略：将不活跃的 KV Cache block 卸载到 CPU 内存或 SSD（如 Agent-X）

**Agent-X 中的 KV Cache 管理**：
- 总存储开销：6.26 GB（0.95 GB 静态 + 5.31 GB cluster combinations, budget=15）
- Planner KV cache: 0.57 GB + Arbiter KV cache: 0.39 GB（静态部分）
- SSD 加载延迟占 prefill 延迟的 5.8%（Planner）和 11.7%（Arbiter）
- PromptWeaver 增加输入 token 从 1,739 到 3,790（+2,051 token 的 KV Cache，~256 MB 额外内存），使 decode 的 TPOT 增加 2.2%（122ms→125ms）

涉及论文标题：
- Agent-X: Full Pipeline Acceleration of On-device AI Agents

## n-gram Language Model for Speculative Decoding (n-gram 投机解码草稿模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

n-gram 语言模型是一种经典的统计语言模型，基于马尔可夫假设：当前 token 的概率仅依赖于前 (n-1) 个 token。在 Agent-X 的 ExSpec 中，n-gram 模型被用作 speculative decoding 的 draft model——从 prompt 中的 few-shot examples 和 user query 构建 trigram（n=3）lookup table（LUT），记录每个 (token_{i-2}, token_{i-1}) pair 出现最频繁的后继 token。该 LUT 在每次 query 时在线构建（83ms），内存占用仅数 KB，无需训练、无需独立 draft LLM。

与传统 LLM 作为 draft model 相比，n-gram LUT 的优势：(1) 零内存开销（KB vs. 数百 MB~数 GB）；(2) 零 token 生成延迟（LUT lookup 是 O(1) 常数时间 vs. LLM 的 131ms/token）；(3) 无需额外 tokenizer（避免 tokenizer 不匹配带来的开销）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Agent-X ExSpec: trigram (n=3) LUT 的构建和使用

# 1. 构建阶段（每 query 一次，输入 token 序列 T[0..M-1]）
LUT = defaultdict(Counter)  # key → {next_token: count}
for i in range(2, M):
    key = (T[i-2], T[i-1])           # 2-token 前缀
    LUT[key][T[i]] += 1              # 统计后继 token 频率

# 归一化为 most-frequent
for key in LUT:
    LUT[key] = max(LUT[key], key=LUT[key].get)  # 取最高频后继

# 2. 使用阶段（draft token 生成, ctx = 当前 2-token 上下文）
def generate_drafts(ctx, LUT, N=4):
    drafts = []
    cur = ctx
    for _ in range(N):
        if cur in LUT:
            drafts.append(LUT[cur])
            cur = (cur[1], LUT[cur])   # 滑动窗口
        else:
            break  # 未命中则停止（但 Agent-X 在第一步就 selective fallback）
    return drafts

# 3. n 的影响（Agent-X 实验）:
# - bigram (n=2): draft accuracy 0.10 (太低，1-token 上下文信息不足)
# - trigram (n=3): draft accuracy 0.25 (最佳 trade-off)
# - quadgram (n=4): draft accuracy 0.31 但 draft token 数量仅 trigram 的 72%
#   → 总 decode 延迟反而比 trigram 慢 5.1%（更长上下文使模型更保守，频繁回退）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

**通用实现方式**（不限于 Agent-X）：
- Prompt Lookup Decoding (PLD): 在 prompt 中搜索最近 n-gram 匹配，返回匹配位置后续的 token 作为 draft
- TensorRT-LLM n-gram: 维护全局 (key, value) n-gram pair 表，支持跨请求共享
- llama.cpp: `ngram-simple`（简单模式匹配）、`ngram-map-k`（哈希表，最小出现次数）、`ngram-cache`（统计短 n-gram 序列）
- HuggingFace: `prompt_lookup_num_tokens` 参数启用 PLD

**Agent-X ExSpec 的特化实现**：
- 提取区域：仅从 few-shot examples + user query 提取（非全 prompt），因 planner 全输入中有大量 tool descriptions 会"污染" LUT
- Selective decoding: 首 token 即检查 LUT——上下文不存在则直接回退自回归
- LUT 构建时间：83ms/query（可忽略）
- 在 TinyAgent-7B (M4 Pro) 上：draft accuracy 0.25 (selective)，decode 加速 1.73×

涉及论文标题：
- Agent-X: Full Pipeline Acceleration of On-device AI Agents

## Prefill and Decode Stages (预填充与解码阶段)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

LLM 推理分为两个阶段：
- **Prefill（预填充）阶段**：一次性处理所有输入 prompt token，计算并填充 KV Cache。计算特征是 compute-bound（算力密集型）——需要大量矩阵乘法。延迟与 prompt token 数成正比。
- **Decode（解码）阶段**：自回归逐 token 生成输出。每次仅处理 1 个新 token（加上读取 KV Cache 中所有历史 token）。计算特征是 memory bandwidth-bound（内存带宽密集型）——算力利用率低，大部分时间花在从内存读取 KV Cache。

在服务端 GPU 场景（NVIDIA H100/H200），decode 延迟压倒性占主导（>95%），因为服务端 GPU 有极高的计算吞吐和内存带宽。但在端侧设备（Apple M4 Pro、Snapdragon X Elite），端侧加速器的计算吞吐仅为服务端的 ~2%、内存带宽仅为 ~11%，这导致 prefill 也成为显著瓶颈。Agent-X 论文观察到端侧 agent 中 prefill 占 21.7%、decode 占 68.7%，因此需要全流水线加速。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Agent-X 中 Planner 和 Arbiter 的 prefill/decode 流程：
```
# Agent Workflow 中的 Prefill 和 Decode
# 以 "Schedule a meeting with John tomorrow at 5pm" 为例

# === Planner LLM (TinyAgent-7B) ===
# Prefill 阶段 (baseline: 1,739 tokens → 在线计算)
prompt = [System Prompt] + [Tool Descriptions] + [Guidelines] + [Examples] + [Query]
# Baseline: 100% 在线 prefill
# Agent-X: PromptWeaver 仅在线 prefill ~519 tokens (30%)

# Decode 阶段 (生成执行计划)
# Baseline: 标准自回归, 131ms/token × ~80 tokens = ~10.5s
# Agent-X: ExSpec trigram LUT speculative decoding, 1.73× speedup

# === Execution Unit ===
# 执行工具调用 (get_email_address, create_calendar_event)

# === Arbiter LLM (TinyAgent-7B) ===
# Prefill 阶段 (baseline: ~90% 静态, Agent-X 4.35× speedup)
# Decode 阶段 (决定是否重试, Agent-X 1.73× speedup)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

**通用优化方式**：
- Prefill 优化：Prefix Caching、Chunked Prefill (SARATHI)、Prefill-Decode Disaggregation (DistServe/Splitwise)
- Decode 优化：Speculative Decoding、KV Cache 量化、Continuous Batching (vLLM)
- 端侧特化：MLX-LM 利用 Apple Silicon 统一内存架构，prefill 和 decode 在 M4 Pro GPU 上执行

**Agent-X 中的特化实现**：
- Prefill: PromptWeaver 通过 prompt 重组实现 Prefix Caching
- Decode: ExSpec 通过 n-gram LUT 实现无 LLM 的 speculative decoding
- 端侧特征：M4 Pro 上 131ms/token（decode），2-token 并行推理 244ms（multi-token tax 1.86×）

涉及论文标题：
- Agent-X: Full Pipeline Acceleration of On-device AI Agents

## Selective Decoding (选择性解码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Selective Decoding 是 Agent-X ExSpec 中提出的 speculative decoding 增强机制。其核心思想是：在 speculative decoding 的每一步开始前，先查询 n-gram LUT 判断当前上下文是否能命中——若命中则进行 draft→verify 的 speculative 流程，若未命中则立即回退到标准自回归生成，避免无效的 draft token 生成和 multi-token tax 惩罚。

动机：在 MLX-LM 框架中，单 token 自回归推理 131ms，2-token 并行验证 244ms（multi-token tax = 1.86×）。如果 n-gram LUT 无法为当前上下文提供有意义 draft token，强制走 speculative 流程会导致验证阶段浪费计算——目标模型仍需进行 batch-2 的 forward（244ms），但 draft token 几乎必定被拒绝，最终输出与自回归相同，却多花了 113ms。

Selective Decoding 的关键优势：决策开销为零——LUT 查询本身是 O(1)，且 LUT 明确知道哪些 context 被覆盖（确定性），不像 LLM draft model 无法预知 draft 质量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Selective Decoding vs. Non-selective Decoding

# Non-selective (总是尝试 speculative)
while not done:
    drafts = generate_drafts(ctx, LUT, N=4)        # 可能生成随机 token
    target_logits = target.forward(token + drafts)  # 244ms (multi-token tax)
    # 若 drafts 全被拒绝 → 浪费 244ms - 131ms = 113ms
    accepted = verify_and_accept(target_logits, drafts)

# Selective (Agent-X ExSpec)
while not done:
    if ctx not in LUT:                              # O(1) 零开销决策
        token = target.autoregressive(ctx)          # 131ms
        output.append(token)
        continue
    drafts = generate_drafts(ctx, LUT, N=4)         # 仅当 LUT 有信息时
    target_logits = target.forward(token + drafts)  # 244ms
    accepted = verify_and_accept(target_logits, drafts)

# 实验效果 (Agent-X Table 3):
# Planner: Non-selective 364 draft tokens generated, 48 accepted (acc=0.13)
#          Selective   194 draft tokens generated, 48 accepted (acc=0.25)
# Arbiter: Non-selective 622 draft tokens generated, 56 accepted (acc=0.09)
#          Selective   218 draft tokens generated, 56 accepted (acc=0.26)
# → Selective 减少 47-65% 无效 draft token，draft accuracy 翻倍
# → Planner 平均 17 次/query 回退, Arbiter 平均 37 次/query 回退
# → Decode 总加速: 1.73× (vs. Non-selective 1.38×)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Selective Decoding 的实现依赖于 draft model 是否提供确定性的信心估计。LLM-based draft model 无法提供这种估计（只能通过 logits 概率近似），而 n-gram LUT 天然支持（key 存在/不存在即二元决策）。这使得 Selective Decoding 特别适合基于检索的 draft 方法（n-gram、suffix automaton、prompt lookup），但不适合 LLM draft model。

涉及论文标题：
- Agent-X: Full Pipeline Acceleration of On-device AI Agents
