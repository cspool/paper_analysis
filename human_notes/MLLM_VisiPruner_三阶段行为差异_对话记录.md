# MLLM VisiPruner 三阶段行为差异 对话记录

- 导出时间：2026-06-03 17:21 CST
- 来源：当前 Codex session 可用对话上下文
- 保存路径：/data3/paper_analysis/human_notes/MLLM_VisiPruner_三阶段行为差异_对话记录.md
- 范围：仅用户输入与 Codex 最终输出
- 备注：用户未指定目标文件名；文件名由当前可见主题推断。

## 对话记录

### 001 User
<environment_context>
  <cwd>/data3/paper_analysis</cwd>
  <shell>bash</shell>
  <current_date>2026-06-01</current_date>
  <timezone>Asia/Shanghai</timezone>
</environment_context>

### 002 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.1_L1_answer.md

## Active selection of the file:
=== 多模态 MLLM 单请求推理流程 ===
输入: Image I ∈ R^{H×W×3}, Text prompt P (如 "What is in this image?")
输出: Text answer A

# ═══ Phase 1: 模态编码 (异构计算单元映射) ═══
# --- Vision Encoding (在 GPU Tensor Core 上执行) ---
with torch.no_grad():
    # Vision Encoder (ViT-L/14, 冻结)
    patches = PatchEmbed(I)                          # [N_v, d_vis], N_v=576, d_vis=1024
    for vit_layer in ViT.blocks:                     # 24 ViT layers
        patches = vit_layer(patches)                 # Self-Attention + MLP
    V = patches[:, 1:]                               # [576, 1024], 去掉 CLS token
    # 笔记: 视觉编码器输出 576 tokens × 1024 dims

# --- Text Tokenization (在 CPU 上执行) ---
T_ids = Tokenizer(P)                                 # [T_text], 如 20 tokens
T_emb = Embedding(T_ids)                             # [T_text, d_llm], d_llm=4096

# ═══ Phase 2: 模态对齐 (Projector) ═══
# --- 视觉特征 → LLM Embedding 空间 ---
V_proj = Projector(V)                                # [576, 4096]
# Projector: Linear(GELU(Linear(V))) 或 更复杂的 Q-Former/Mamba-2 scan
#    参数: ~2·d_vis·d_llm ≈ 2·1024·4096 ≈ 8.4M (极小)
#    笔记: knowledge_notes/.../Multimodal Connector _ Projector in MLLM.md

# ═══ Phase 3: LLM Backbone (跨模态融合) ═══
# --- Token 拼接 ---
<!-- ! 文本和图像token拼接 -->
H = Concat([V_proj, T_emb])                          # [576+20, 4096] = [596, 4096]

# --- 逐层 Transformer (三阶段跨模态交互) ---
for l = 1..L:                                        # L=32 for LLaVA-v1.5
    H_norm = RMSNorm(H)                              # [596, 4096]
    Q, K, V = H_norm @ W_Q, H_norm @ W_K, H_norm @ W_V

    # Attention 矩阵包含 4 个区域:
    # [V→V][V→T]  视觉 self-attn + 视觉→文本 cross-attn
    # [T→V][T→T]  文本→视觉 cross-attn + 文本 self-attn
    attn_scores = Q @ K^T / sqrt(d_head)             # [596, 596]
    # → O((N_v+T_text)²) = O(596²) ≈ 355K 次乘法 (per head per layer)

    # --- 三阶段行为差异 (VisiPruner 笔记) ---
    <!-- ! -->
    if l <= 8:   # Stage 1: Shallow — Task Recognition
        # 跨模态 attention 几乎无效; 视觉 token 仅作为 attention sink
        # 文本 token 编码的是 task type (如 "number"→counting), 非视觉内容
        pass
    elif l <= 23: # Stage 2: Middle — Sparse Cross-Modal Grounding
        # 跨模态融合突然发生 (abrupt onset)
        # 仅 ~10/576 视觉 token 驱动跨模态信息传递
        # → 大部分视觉 token 的 cross-attention 计算是冗余的!
        pass
    else:         # Stage 3: Deep — Linguistic Alignment
        # 视觉信息已集成到文本表示, 视觉 token 不再需要
        # → 可安全裁剪 (Vision Exit)
        pass

    attn_out = Softmax(attn_scores) @ V              # [596, 4096]
    H = H + attn_out                                  # residual
    H = H + FFN(RMSNorm(H))                          # gate_proj+up_proj+down_proj

# ═══ Phase 4: 自回归解码 ═══
generated_ids = []
for step in range(max_new_tokens):
    logits = H[-1] @ W_lm_head                       # [vocab_size], 仅取最后位置
    next_token = Argmax(logits / temperature)         # 或采样
    generated_ids.append(next_token)
    if next_token == EOS: break
    # 新 token 追加到序列, KV-cache 复用 → 每步仅计算单个新 token
    # decode 阶段 FLOPs/token ≈ 2·L·d_model² + 3·L·d_model·d_ff
    # = 2·32·4096² + 3·32·4096·14336 ≈ 1.07G + 5.64G ≈ 6.7G FLOPs

answer = Detokenizer(generated_ids)
## Open tabs:
- Q1.1_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.1_L1_answer.md
- DDPM：NIPS20 2d8e12d10b6e8072acc2c4dcd7234cb7.md: human_notes/生成式模型原理/DDPM：NIPS20 2d8e12d10b6e8072acc2c4dcd7234cb7.md
- run_all_papers.py: scripts/run_all_papers.py

## My request for Codex:
=== 多模态 MLLM 单请求推理流程 ===
输入: Image I ∈ R^{H×W×3}, Text prompt P (如 "What is in this image?")
输出: Text answer A

# ═══ Phase 1: 模态编码 (异构计算单元映射) ═══
# --- Vision Encoding (在 GPU Tensor Core 上执行) ---
with torch.no_grad():
    # Vision Encoder (ViT-L/14, 冻结)
    patches = PatchEmbed(I)                          # [N_v, d_vis], N_v=576, d_vis=1024
    for vit_layer in ViT.blocks:                     # 24 ViT layers
        patches = vit_layer(patches)                 # Self-Attention + MLP
    V = patches[:, 1:]                               # [576, 1024], 去掉 CLS token
    # 笔记: 视觉编码器输出 576 tokens × 1024 dims

# --- Text Tokenization (在 CPU 上执行) ---
T_ids = Tokenizer(P)                                 # [T_text], 如 20 tokens
T_emb = Embedding(T_ids)                             # [T_text, d_llm], d_llm=4096

# ═══ Phase 2: 模态对齐 (Projector) ═══
# --- 视觉特征 → LLM Embedding 空间 ---
V_proj = Projector(V)                                # [576, 4096]
# Projector: Linear(GELU(Linear(V))) 或 更复杂的 Q-Former/Mamba-2 scan
#    参数: ~2·d_vis·d_llm ≈ 2·1024·4096 ≈ 8.4M (极小)
#    笔记: knowledge_notes/.../Multimodal Connector _ Projector in MLLM.md

# ═══ Phase 3: LLM Backbone (跨模态融合) ═══
# --- Token 拼接 ---
<!-- ! 文本和图像token拼接 -->
H = Concat([V_proj, T_emb])                          # [576+20, 4096] = [596, 4096]

# --- 逐层 Transformer (三阶段跨模态交互) ---
for l = 1..L:                                        # L=32 for LLaVA-v1.5
    H_norm = RMSNorm(H)                              # [596, 4096]
    Q, K, V = H_norm @ W_Q, H_norm @ W_K, H_norm @ W_V

    # Attention 矩阵包含 4 个区域:
    # [V→V][V→T]  视觉 self-attn + 视觉→文本 cross-attn
    # [T→V][T→T]  文本→视觉 cross-attn + 文本 self-attn
    attn_scores = Q @ K^T / sqrt(d_head)             # [596, 596]
    # → O((N_v+T_text)²) = O(596²) ≈ 355K 次乘法 (per head per layer)

    # --- 三阶段行为差异 (VisiPruner 笔记) ---
    <!-- ! -->
    if l <= 8:   # Stage 1: Shallow — Task Recognition
        # 跨模态 attention 几乎无效; 视觉 token 仅作为 attention sink
        # 文本 token 编码的是 task type (如 "number"→counting), 非视觉内容
        pass
    elif l <= 23: # Stage 2: Middle — Sparse Cross-Modal Grounding
        # 跨模态融合突然发生 (abrupt onset)
        # 仅 ~10/576 视觉 token 驱动跨模态信息传递
        # → 大部分视觉 token 的 cross-attention 计算是冗余的!
        pass
    else:         # Stage 3: Deep — Linguistic Alignment
        # 视觉信息已集成到文本表示, 视觉 token 不再需要
        # → 可安全裁剪 (Vision Exit)
        pass

    attn_out = Softmax(attn_scores) @ V              # [596, 4096]
    H = H + attn_out                                  # residual
    H = H + FFN(RMSNorm(H))                          # gate_proj+up_proj+down_proj

# ═══ Phase 4: 自回归解码 ═══
generated_ids = []
for step in range(max_new_tokens):
    logits = H[-1] @ W_lm_head                       # [vocab_size], 仅取最后位置
    next_token = Argmax(logits / temperature)         # 或采样
    generated_ids.append(next_token)
    if next_token == EOS: break
    # 新 token 追加到序列, KV-cache 复用 → 每步仅计算单个新 token
    # decode 阶段 FLOPs/token ≈ 2·L·d_model² + 3·L·d_model·d_ff
    # = 2·32·4096² + 3·32·4096·14336 ≈ 1.07G + 5.64G ≈ 6.7G FLOPs

answer = Detokenizer(generated_ids),教我

### 003 Codex Final
可以。你这段其实是在讲：**一张图 + 一段文字，如何变成 MLLM 的回答**。可以把它理解成四步流水线：

**1. 先把图像变成视觉 token**

图像 `I` 不能直接喂给 LLM，所以先经过 Vision Encoder，比如 ViT。

```python
patches = PatchEmbed(I)
for vit_layer in ViT.blocks:
    patches = vit_layer(patches)
V = patches[:, 1:]
```

意思是：

- 图片被切成很多 patch。
- 每个 patch 变成一个向量。
- ViT 处理这些向量，得到视觉特征。
- 这里 `576 tokens × 1024 dims` 表示：图像被表示成 576 个视觉 token，每个 token 是 1024 维。

你可以把 `V` 想成：

> 这张图的 576 个局部语义碎片。

比如某些 token 可能对应猫脸、桌子边缘、背景、文字区域等。

**2. 再把视觉 token 投影到 LLM 能懂的空间**

ViT 输出是 `1024` 维，但 LLM 的 embedding 维度是 `4096`，所以要接一个 projector：

```python
V_proj = Projector(V)  # [576, 4096]
```

这个 projector 的作用是：

> 把“视觉语言”翻译成“LLM embedding 语言”。

如果没有 projector，LLM 不知道这些视觉向量是什么意思。

所以此时：

```text
图像 token: [576, 4096]
文本 token: [20, 4096]
```

它们终于在同一个向量空间里了。

**3. 把图像 token 和文本 token 拼起来，交给 LLM**

```python
H = Concat([V_proj, T_emb])  # [596, 4096]
```

这一步非常关键。

MLLM 并不是每一层都单独开一个图文 cross-attention 模块，而是经常直接把图像 token 当成“前缀 token”，和文本 token 拼成一个长序列。

像这样：

```text
[图像1, 图像2, ..., 图像576, 文本1, 文本2, ..., 文本20]
```

然后 LLM 在 Transformer 层里做 self-attention。

这时 attention 矩阵大概可以看成四块：

```text
[V -> V]  图像 token 看图像 token
[V -> T]  图像 token 看文本 token
[T -> V]  文本 token 看图像 token
[T -> T]  文本 token 看文本 token
```

不过要注意：如果是 decoder-only LLM，通常有 causal mask。实际更重要的是：

> 文本 token 可以 attend 到前面的图像 token，因此文本表示会吸收视觉信息。

也就是说，当模型看到问题：

```text
What is in this image?
```

它的文本 token 会通过 attention 去“读取”前面的 576 个图像 token。

**4. 三阶段跨模态交互：不是所有层都同等重要**

你这里写的 VisiPruner 三阶段很有意思：

```python
if l <= 8:
    Stage 1: Task Recognition
elif l <= 23:
    Stage 2: Sparse Cross-Modal Grounding
else:
    Stage 3: Linguistic Alignment
```

可以这样理解：

浅层：先理解任务  
模型先弄清楚用户要干什么，比如：

- 是问数量？
- 是问颜色？
- 是问位置？
- 是问 OCR？
- 是问描述？

这时视觉 token 还没有真正高效参与推理。

中层：真正图文对齐  
这部分最关键。文本 token 开始从图像 token 里找答案。

比如问题是：

```text
How many dogs are in the image?
```

模型不需要所有 576 个视觉 token。它可能只需要和狗相关的少量 token。

所以你写的：

```text
仅 ~10/576 视觉 token 驱动跨模态信息传递
```

意思是：大量视觉 token 对当前问题是冗余的。

深层：语言组织  
视觉信息已经被整合进文本 hidden state 里了。后面更多是在组织语言、对齐答案格式。

比如从“检测到两个狗相关区域”变成：

```text
There are two dogs in the image.
```

**5. 最后进入自回归解码**

前面处理整段输入叫 prefill。之后模型开始一个 token 一个 token 地生成答案：

```python
for step in range(max_new_tokens):
    logits = H[-1] @ W_lm_head
    next_token = Argmax(logits / temperature)
```

每一步只预测下一个 token。

例如：

```text
There
There are
There are two
There are two dogs
...
```

KV-cache 的作用是：之前算过的 token 不重复算，只给新 token 算新的 Q/K/V，所以 decode 阶段比重新跑整段序列便宜很多。

一个核心总结：

```text
图像 -> ViT -> 视觉 token
视觉 token -> Projector -> LLM embedding token
图像 token + 文本 token -> LLM Transformer
文本 token 通过 attention 读取图像信息
最后自回归生成答案
```

可以把 MLLM 想成：

> ViT 是眼睛，Projector 是翻译器，LLM 是大脑和嘴巴。  
> 眼睛看到图像，翻译器把视觉信息变成语言模型能处理的向量，大脑结合问题理解图像，嘴巴逐字生成答案。

你这段笔记整体方向是对的。一个小提醒：`Projector` 参数量那行要看具体结构，如果是 LLaVA 常见的 `1024 -> 4096 -> 4096` 两层 MLP，参数量会比 `8.4M` 更大，大约二十多 M。

### 004 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.1_L1_answer.md

## Active selection of the file:
三阶段行为差异 (VisiPruner 笔记)
## Open tabs:
- Q1.1_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.1_L1_answer.md
- DDPM：NIPS20 2d8e12d10b6e8072acc2c4dcd7234cb7.md: human_notes/生成式模型原理/DDPM：NIPS20 2d8e12d10b6e8072acc2c4dcd7234cb7.md
- run_all_papers.py: scripts/run_all_papers.py

## My request for Codex:
三阶段行为差异 (VisiPruner 笔记),在详细点,Prefill一共L层

### 005 Codex Final
对，**Prefill 一共就是跑完整的 `L` 层 Transformer**。VisiPruner 的“三阶段”不是额外多跑三遍，而是把这 `L` 层按行为分成三段：

```text
Prefill:
H^0 = [visual tokens, text tokens]

for l = 1 ... L:
    H^l = TransformerLayer_l(H^{l-1})

# 这 L 层中：
# 浅层: 任务识别
# 中层: 稀疏图文融合
# 深层: 语言对齐
```

以 LLaVA-v1.5 7B 为例，`L=32`，大致是：

```text
Layer 1-8    : Shallow / Task Recognition
Layer 9-23   : Middle / Sparse Cross-Modal Grounding
Layer 24-32  : Deep / Linguistic Alignment
```

**核心直觉**

MLLM 不是每一层都“认真看图”。它在 Prefill 的 `L` 层里，大概经历这个过程：

```text
先理解问题类型 -> 再从图里找少量关键区域 -> 最后把视觉信息变成语言答案
```

也就是：

```text
Task Recognition -> Visual Grounding -> Language Refinement
```

**Stage 1: 浅层，Task Recognition**

这一段通常是 `l <= 8`。

此时模型主要在理解文本 prompt 的任务类型，而不是认真抽取图像内容。

比如 prompt 是：

```text
How many dogs are in the image?
```

浅层更像是在判断：

```text
这是 counting 问题
答案可能需要 number
需要关注 object 数量
```

而不是已经知道“图里有 2 只狗”。

所以虽然 attention 矩阵里存在 `Text -> Vision` 的注意力：

```text
A_TV^l = Softmax(Q_text K_vision^T)
```

但这个 attention 往往不代表真实视觉信息贡献。VisiPruner 的观察是：浅层很多 visual token 更像 **attention sink**，它们吸走 attention mass，让 softmax 稳定，但不真正传递任务相关视觉语义。

所以浅层的行为可以理解为：

```python
if l <= 8:
    # 文本 token: 主要识别任务类型
    # 视觉 token: 多数只是稳定 attention 分布
    # cross-modal attention 有形无实
```

这也是为什么 VisiPruner 敢在浅层做 aggressive pruning / attention merging：

```text
把很多视觉 attention 合并到一个 sink token
或者跳过浅层视觉相关 attention
```

因为浅层不是主要图文融合发生的位置。

**Stage 2: 中层，Sparse Cross-Modal Grounding**

这一段大概是 `9 <= l <= 23`。

这里才是真正“看图回答问题”的阶段。

还是这个问题：

```text
How many dogs are in the image?
```

中层开始发生真正的跨模态 grounding：

```text
文本 token “how many / dogs” 开始从图像 token 中找狗相关区域
```

但是关键点是：**不是 576 个视觉 token 都重要**。

假设图像有：

```text
576 个 visual tokens
```

真正对问题有贡献的可能只有：

```text
狗身体区域 token
狗头区域 token
另一个狗的区域 token
```

也许只有十几个 token。

所以 VisiPruner 的中层结论是：

```text
跨模态融合确实发生，但它是稀疏的。
```

可以写成：

```python
elif l <= 23:
    # 真正的 text-to-vision 信息读取发生
    # 但只有少数 visual tokens 对最后文本 hidden state 有影响
    # 大量 visual tokens 对当前问题是冗余背景
```

VisiPruner 不完全相信 attention score，而是看 **influence**：

```text
如果 mask 掉某个 visual token 后，最后 text token 的 attention output 明显变化，
说明这个 visual token 真有贡献。
```

形式上可以理解为：

```text
原始输出:
o = Attention(q_last, K_all, V_all)

mask 第 j 个视觉 token 后:
o_mask_j = Attention(q_last, K_all_without_j, V_all_without_j)

影响分数:
influence_j = ||o - o_mask_j||_2
```

如果 `influence_j` 很大，说明第 `j` 个视觉 token 重要；如果很小，就可以剪掉。

所以中层是：

```text
576 个视觉 token -> 只保留约 10 个关键 token
```

**Stage 3: 深层，Linguistic Alignment**

这一段大概是 `l >= 24`。

此时视觉信息已经被写入文本 hidden state 里了。

也就是说，中层之后，最后几个文本 token 的表示里已经包含了类似信息：

```text
图里有两只狗
狗在草地上
答案应该是 "two"
```

深层主要做的是语言组织：

```text
two -> There are two dogs in the image.
```

所以深层更像纯语言模型的 refinement：

```python
else:
    # 视觉信息已经进入文本表示
    # 后续层主要做语法、答案格式、语言对齐
    # visual tokens 可以退出
```

这就是 **Vision Exit**：

```text
从某一层开始，把所有 visual tokens 移除。
后面的层只保留 text tokens 继续前向传播。
```

以 LLaVA-v1.5 7B 为例，论文笔记里平均 exit 大概在 layer 24 左右。

**放回你的 Prefill 代码里，可以这样理解**

```python
for l in range(1, L + 1):  # Prefill 一共 L 层
    H_v, H_t = split(H)

    if l <= 8:
        # Stage 1: Task Recognition
        # 文本 token 学到“问题类型”
        # visual tokens 多数只是 attention sink
        # 跨模态 attention 有形无实
        pass

    elif l <= 23:
        # Stage 2: Sparse Cross-Modal Grounding
        # 真正图文融合发生
        # text token 从少数关键 visual tokens 读取信息
        # 适合做 influence-based token selection
        pass

    else:
        # Stage 3: Linguistic Alignment
        # 视觉信息已进入 text hidden states
        # 后面主要是语言组织
        # 可以 Vision Exit，删除 visual tokens
        pass
```

一句话总结：

```text
Prefill 跑 L 层。
浅层决定“我要看什么”，中层真正“从图里找答案”，深层把“找到的答案”说成人话。
```

所以 VisiPruner 的本质不是简单说“视觉 token 很多，可以剪”，而是更细：

```text
浅层：视觉 attention 大多是假忙，可以合并/跳过。
中层：视觉信息真的有用，但只需少量关键 token。
深层：视觉信息已经注入文本，可以让 visual tokens 退出。
```

### 006 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.1_L1_answer.md

## Active selection of the file:
    <!-- ! 多模态Attention 矩阵包含 4 个区域: -->
    # [V→V][V→T]  视觉 self-attn + 视觉→文本 cross-attn
    # [T→V][T→T]  文本→视觉 cross-attn + 文本 self-attn
    attn_scores = Q @ K^T / sqrt(d_head)             # [596, 596]
    # → O((N_v+T_text)²) = O(596²) ≈ 355K 次乘法 (per head per layer)

    # --- 三阶段行为差异 (VisiPruner 笔记) ---
    <!-- ! 重点在于text token的计算,image token可以跳过或只支持sink作用, text-tokens@text-tokens-->
    if l <= 8:   # Stage 1: Shallow — Task Recognition
        # 跨模态 attention 几乎无效; 视觉 token 仅作为 attention sink
        # 文本 token 编码的是 task type (如 "number"→counting), 非视觉内容
        pass
    <!-- ! text token从image token中找相关性强的, 因此稀疏, text-tokens@image-tokens -->
    elif l <= 23: # Stage 2: Middle — Sparse Cross-Modal Grounding
        # 跨模态融合突然发生 (abrupt onset)
        # 仅 ~10/576 视觉 token 驱动跨模态信息传递
        # → 大部分视觉 token 的 cross-attention 计算是冗余的!
        pass
    <!-- ! image token的信息融入text token中, text-tokens@text-tokens -->
    else:         # Stage 3: Deep — Linguistic Alignment
        # 视觉信息已集成到文本表示, 视觉 token 不再需要
        # → 可安全裁剪 (Vision Exit)
        pass

    <!-- ! CLIP方便多模态融合的训练 -->
    attn_out = Softmax(attn_scores) @ V              # [596, 4096]
    H = H + attn_out                                  # residual
    H = H + FFN(RMSNorm(H))                          # gate_proj+up_proj+down_proj
## Open tabs:
- Q1.1_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.1_L1_answer.md
- SKILL.md: .claude/skills/obsidian-keyword-explain/SKILL.md
- SKILL.md: .codex/skills/obsidian-keyword-explainer/SKILL.md
- DDPM：NIPS20 2d8e12d10b6e8072acc2c4dcd7234cb7.md: human_notes/生成式模型原理/DDPM：NIPS20 2d8e12d10b6e8072acc2c4dcd7234cb7.md
- run_all_papers.py: scripts/run_all_papers.py

## My request for Codex:
    <!-- ! 多模态Attention 矩阵包含 4 个区域: -->
    # [V→V][V→T]  视觉 self-attn + 视觉→文本 cross-attn
    # [T→V][T→T]  文本→视觉 cross-attn + 文本 self-attn
    attn_scores = Q @ K^T / sqrt(d_head)             # [596, 596]
    # → O((N_v+T_text)²) = O(596²) ≈ 355K 次乘法 (per head per layer)

    # --- 三阶段行为差异 (VisiPruner 笔记) ---
    <!-- ! 重点在于text token的计算,image token可以跳过或只支持sink作用, text-tokens@text-tokens-->
    if l <= 8:   # Stage 1: Shallow — Task Recognition
        # 跨模态 attention 几乎无效; 视觉 token 仅作为 attention sink
        # 文本 token 编码的是 task type (如 "number"→counting), 非视觉内容
        pass
    <!-- ! text token从image token中找相关性强的, 因此稀疏, text-tokens@image-tokens -->
    elif l <= 23: # Stage 2: Middle — Sparse Cross-Modal Grounding
        # 跨模态融合突然发生 (abrupt onset)
        # 仅 ~10/576 视觉 token 驱动跨模态信息传递
        # → 大部分视觉 token 的 cross-attention 计算是冗余的!
        pass
    <!-- ! image token的信息融入text token中, text-tokens@text-tokens -->
    else:         # Stage 3: Deep — Linguistic Alignment
        # 视觉信息已集成到文本表示, 视觉 token 不再需要
        # → 可安全裁剪 (Vision Exit)
        pass

    <!-- ! CLIP方便多模态融合的训练 -->
    attn_out = Softmax(attn_scores) @ V              # [596, 4096]
    H = H + attn_out                                  # residual
    H = H + FFN(RMSNorm(H))                          # gate_proj+up_proj+down_proj,看起来image token对text token的计算,image token对image token的计算不那么重要?

### 007 Codex Final
对，你抓到核心了：**在 LLaVA 这类 decoder-only MLLM 里，最终真正要算好的是 text token 的 hidden state**，因为答案是从最后一个 text token / generated token 继续生成出来的。

但要稍微精确分三块看。

假设序列是：

```text
H = [V tokens, T tokens]
```

attention 矩阵行是 Query，列是 Key/Value：

```text
          K/V
        V      T
Q V   [V,V]  [V,T]
  T   [T,V]  [T,T]
```

真正对回答最关键的是：

```text
[T,V]  text token 读取 image token
[T,T]  text token 读取前面的 text token
```

也就是你写的：

```text
text-tokens @ image-tokens
text-tokens @ text-tokens
```

**为什么 image token 的输出没那么重要？**

因为最后生成答案时通常用的是文本侧 hidden state：

```python
logits = H_text[-1] @ W_lm_head
```

不是用 image token 去生成词。

所以 image token 更像是“被读取的视觉记忆库”：

```text
image token 提供 K/V
text token 用 Q 去读它们
```

因此，对最终回答来说，最关键的数据流是：

```text
Image token 信息 -> 被 text token attention 读取 -> 融入 text hidden state -> 生成答案
```

**那 [V,V] image token 对 image token 重要吗？**

有用，但相对没那么关键，尤其是在 LLM backbone 里。

视觉 token 在进入 LLM 前，已经被 CLIP/ViT 编码过了：

```text
ViT 内部已经做过大量 image self-attention
```

所以到了 LLM 里，视觉 token 之间再做很多 `[V,V]` self-attention，边际收益可能比较小。VisiPruner 的观察就是：很多层里 `[V,V]` 和部分视觉相关 attention 可以跳过或弱化。

**那 [V,T] image token 对 text token 呢？**

如果是 LLaVA 常见的顺序：

```text
[image tokens, text tokens]
```

并且使用 causal mask，那么视觉 token 在前，文本 token 在后：

```text
image token 不能 attend 到后面的 text token
```

所以 `[V,T]` 这块通常会被 causal mask 掉，或者作用很弱。真正的跨模态读取主要是 `[T,V]`。

所以你这句可以改得更准确：

```text
重点不是 image token 如何更新自己，而是 text token 如何读取 image token。
image token 主要作为 K/V 视觉记忆存在；最终答案由 text token hidden state 生成。
```

三阶段对应就是：

```text
浅层: text 主要识别任务，[T,V] 表面存在但信息贡献弱，V 近似 sink
中层: text 真正从少数 V token 读取视觉信息，[T,V] 最重要
深层: 视觉信息已经进入 text hidden state，主要靠 [T,T] 做语言整理
```

所以你的判断基本对：**image token 对 image token 的计算、image token 作为 query 去算别人的计算，通常不是 MLLM 回答生成的主路径；主路径是 text token 作为 query，读取 image token 的 K/V。**

### 008 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.1_L1_answer.md

## Active selection of the file:
    <!-- ! 多模态Attention 矩阵包含 4 个区域: -->
    <!-- ! 核心逻辑是text-token对image token的读取 -->
    <!-- ! image-tokens@*是否用于image-tokens的稀疏计算?类似自蒸馏,但计算开销如何? -->
## Open tabs:
- Q1.1_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.1_L1_answer.md
- SKILL.md: .claude/skills/obsidian-keyword-explain/SKILL.md
- SKILL.md: .codex/skills/obsidian-keyword-explainer/SKILL.md
- DDPM：NIPS20 2d8e12d10b6e8072acc2c4dcd7234cb7.md: human_notes/生成式模型原理/DDPM：NIPS20 2d8e12d10b6e8072acc2c4dcd7234cb7.md
- run_all_papers.py: scripts/run_all_papers.py

## My request for Codex:
    <!-- ! 多模态Attention 矩阵包含 4 个区域: -->
    <!-- ! 核心逻辑是text-token对image token的读取 -->
    <!-- ! image-tokens@*是否用于image-tokens的稀疏计算?类似自蒸馏,但计算开销如何? -->

### 009 Codex Final
可以这样理解你这条注释：

```md
<!-- ! image-tokens@* 是否用于 image-tokens 的稀疏计算? 类似自蒸馏, 但计算开销如何? -->
```

答案是：**可以用于视觉 token 的冗余检测/稀疏化，但通常不适合作为 MLLM 回答生成的主路径；如果在线每层都算，开销可能抵消剪枝收益。**

Attention 四块里：

```text
          K/V
        V      T
Q V   [V,V]  [V,T]
  T   [T,V]  [T,T]
```

对最终回答最关键的是：

```text
[T,V]: text token 读取 image token
[T,T]: text token 做语言推理和答案组织
```

因为最后生成 logits 的通常是文本侧 hidden state：

```python
logits = H_text[-1] @ W_lm_head
```

而不是 image token 直接生成词。

所以 `image-tokens @ *` 更像是辅助路径：

```text
[V,V]: 视觉 token 之间互相更新，可用于发现视觉冗余
[V,T]: 视觉 token 读取文本，但在 LLaVA 这类 [image, text] + causal mask 里通常被 mask 掉
```

其中 `[V,V]` 可以用于视觉 token 稀疏计算，比如：

```text
1. 找相似 visual tokens，然后 merge/drop
2. 找视觉 token 的中心性/代表性
3. 估计哪些 patch 是冗余背景
4. 给后续层的 token pruning 提供先验
```

但它的问题是：**它不一定知道用户问什么。**

比如图里有狗、车、招牌。问题是：

```text
What is written on the sign?
```

`[V,V]` 可能觉得狗很显著，保留狗相关 token；但真正重要的是招牌 OCR 区域。  
所以只靠 image-token self-attention 做剪枝，容易保留“视觉显著”而不是“问题相关”。

VisiPruner 更看重 `[T,V]`，原因就在这里：

```text
text token 问什么，才决定哪些 image token 重要。
```

**计算开销上**

如果在线算完整 `[V,V]`：

```text
N_v = 576
[V,V] score 大小 = 576 × 576 = 331,776
[T,V] 如果 T=20，则 = 20 × 576 = 11,520
```

所以 `[V,V]` 比 `[T,V]` 大很多。  
如果每层都算 `[V,V]` 来辅助剪枝，很容易变成：

```text
为了省计算，先额外算一大块视觉 self-attention
```

收益就不稳了。

更合理的策略是：

```text
1. 不每层算 image-tokens@*
2. 只在早期某一层做一次视觉冗余估计
3. 或直接复用 CLIP/ViT 已经算过的视觉关系
4. 真正 question-aware 的剪枝用 [T,V] 或 influence score
```

你这段可以改成更准确的注释：

```md
<!-- ! 核心路径是 text-token 作为 query 读取 image-token 的 K/V，即 [T,V] -->
<!-- ! image-token 作为 query 的 [V,V]/[V,T] 主要更新视觉 token，本身不直接生成答案 -->
<!-- ! [V,V] 可用于视觉冗余检测/merge/pruning，但不一定 question-aware，且 O(N_v^2) 开销较大 -->
<!-- ! 若用于剪枝，最好只在少数层计算或复用 ViT/CLIP 的视觉关系；中层关键性仍应看 [T,V]/influence -->
```

一句话：**`image-tokens@*` 可以当“视觉 token 自我整理/压缩”的信号，但 MLLM 回答的主干是 `text-tokens@image-tokens`，也就是文本按问题去读图。**

### 010 User
<environment_context>
  <current_date>2026-06-03</current_date>
  <timezone>Asia/Shanghai</timezone>
  <filesystem><workspace_roots><root>/data3/paper_analysis</root></workspace_roots><permission_profile type="managed"><file_system type="restricted"><entry access="read"><special>:root</special></entry><entry access="write"><path>/data3/paper_analysis</path></entry><entry access="write"><special>:slash_tmp</special></entry><entry access="write"><special>:tmpdir</special></entry><entry access="read"><path>/data3/paper_analysis/.git</path></entry><entry access="read"><path>/data3/paper_analysis/.agents</path></entry><entry access="read"><path>/data3/paper_analysis/.codex</path></entry></file_system></permission_profile></filesystem>
</environment_context>

### 011 User
# Context from my IDE setup:

## Open tabs:
- L1_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L1_horizon_summary.md
- SKILL.md: .codex/skills/export-conversation-notes/SKILL.md
- Accelerating Sparse Transformer Inference on GPU (STOF).md: human_notes/Accelerating Sparse Transformer Inference on GPU (STOF).md
- Adaptive Draft Sequence Length_ Enhancing Speculative Decoding Throughput on PIM-Enabled Systems.md: human_notes/Adaptive Draft Sequence Length_ Enhancing Speculative Decoding Throughput on PIM-Enabled Systems.md
- AdaServe_ Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding.md: human_notes/AdaServe_ Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding.md

## My request for Codex:
[$export-conversation-notes](/data3/paper_analysis/.codex/skills/export-conversation-notes/SKILL.md) 

### 012 User
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
