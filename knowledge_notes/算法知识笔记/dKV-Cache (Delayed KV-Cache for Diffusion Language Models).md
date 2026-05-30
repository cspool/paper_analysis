## dKV-Cache (Delayed KV-Cache for Diffusion Language Models)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

dKV-Cache（Delayed KV-Cache）是首个为 Diffusion Language Models (DLMs) 设计的 KV-Cache 机制，由 NUS xML Lab 提出（NeurIPS'25）。传统 KV-Cache 在 DLM 中不可用，因为 DLM 的双向注意力（每步 K/V 变化）和非顺序解码（无法预知下一步位置）破坏了缓存复用的前提。dKV-Cache 的核心洞见来自对 DLM 去噪过程中 token 表征动态的实证分析：已解码 token 的 K/V 在后续步趋于稳定（可缓存），而 [MASK] token 持续波动（需重新计算）；最大的表征变化发生在 token 从 [MASK] 变为解码状态的那一步。

基于此提出三个核心机制：(1) **延迟缓存**（delayed caching）：仅缓存已解码 token 的 K/V，未解码（掩码）token 每步重新计算，解决双向注意力下 K/V 时变问题；(2) **一步延迟**（one-step delay）：使用上一步的掩码集合 M_{t-1} 而非当前步 M_t 决定缓存对象，避免刚解码 token 在表征剧变时被过早缓存——实验显示无延迟时性能崩溃至接近零；(3) **缓存刷新**（cache refreshing）：每 N 步清空并重新计算全序列 KV，防止长时间累积的近似误差导致质量退化。

设计两种互补变体：(a) **dKV-Cache-Decode**：近乎无损，刷新间隔大（N=4-8），复杂度 O(L³)；(b) **dKV-Cache-Greedy**：激进缓存，仅计算当前 token D_t、上一步 token D_{t-1} 和局部窗口 W(t)（≤6 个 token）的 QKV，将复杂度降至 O(L²)，但质量略有下降。另有 dKV-Cache-Prefill（预填充 token 永久缓存不刷新）和 dKV-Cache-PD（prefill 永久+decode 间歇刷新）处理长 prefill 场景。方法为 training-free，直接应用于现有预训练 DLM。加速比 2-10×，GPU batch size 越大加速比越高。

从算法pipeline角度拆解术语。

**dKV-Cache-Decode 伪代码**（步 t）：

```
Require: x^{1:L}_{c(t)}, M_t (掩码 token 位置集合), 
         K_{t-1}^{I\M_{t-1}} (缓存 K), V_{t-1}^{I\M_{t-1}} (缓存 V)

// 一步延迟：使用上一步的掩码集 M_{t-1}
1: x' ← x[M_{t-1}]                          // 仅取掩码 token 子序列
2: PE' ← [PE[I\M_{t-1}]; PE[M_{t-1}]]       // 重排位置编码：缓存侧在左

3: Transformer(x') → Q_t^{M_t}, K_t^{M_t}, V_t^{M_t}  // 仅计算掩码 token
4: K_t^I ← Concat(K_{t-1}^{I\M_{t-1}}, K_t^{M_{t-1}}) // 拼接缓存与新 K
5: V_t^I ← Concat(V_{t-1}^{I\M_{t-1}}, V_t^{M_{t-1}}) // 拼接缓存与新 V
6: Reorder(K_t^I, V_t^I, mapping_to_I\M_t)   // 提取下一步的缓存集
7: p' ← Attention(Q_t^{M_t}, K_t^I, V_t^I)   // 双向注意力（全 K/V）
8: p ← Scatter(p', M_{t-1})                  // logits 散播回原位置

// 每 N 步刷新：设 M_{t-1}=∅，重新计算全序列 KV
```

**dKV-Cache-Greedy 的对齐计算集**：
```
// M_t = 上一步掩码集（所有未解码 + [MASK] token）
// Greedy 变体：M_t = {D_t, D_{t-1}} ∪ W(D_{t-1})
// 其中 W 是以 D_{t-1} 为中心、半径 floor(w/2) 的局部窗口
// w ≤ 6， |M_t| = O(1)，从而 O(L³) → O(L²)
```

**cache ratio 度量**：
```
cache_ratio = (1/T) Σ_{i=1}^T |T_i^{cache}| / L
其中 T_i^{cache} = 步 i 从缓存复用的 token 数 = |I \ M_{i-1}|
```

术语一般如何实现？如何使用？

开源实现：https://github.com/horseee/dKV-Cache（Python/PyTorch）。修改 HuggingFace 模型的 forward 函数，插入 concat_reorder 逻辑：重排 token 位置→仅计算掩码 token→concat 缓存 KV→注意力→scatter→更新缓存。使用方式：

```python
# Dream 模型
from models.dream import DreamModel
model = DreamModel.from_pretrained(
    "Dream-7B", use_cache=True, 
    cache_type="decode",       # 或 "greedy", "prefill", "pd"
    cache_steps=4               # 刷新间隔
)

# LLaDA 模型
from models.llada import LLaDAModelLM  
model = LLaDAModelLM.from_pretrained(
    "LLaDA-8B-Instruct", use_cache=True,
    cache_type="decode", cache_steps=8
)
```

关键实现细节：(1) concat_reorder 通过重排序列使缓存 token 连续，将索引开销从 K/V 矩阵 [B,L,D] 层级转移到 token [B,L] 层级；(2) 位置编码随序列重排而同步调整，每次仅需一次 PE reorder、跨层共享；(3) batch size 对加速比影响显著——batch=1 时 memory-bound 导致缓存可能反而减速，batch>1 时加速效果显著。支持 Dream 的三种缓存策略：Un-Shift（标准）、Right-Shift（右移一位）、Un&Right-Shift（两者条件组合）。实测 A6000/H20 GPU 上 1.75-10.19× 加速。

涉及论文标题：
- dKV-Cache: The Cache for Diffusion Language Models
