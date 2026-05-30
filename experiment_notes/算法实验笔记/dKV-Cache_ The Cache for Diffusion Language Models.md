## dKV-Cache: The Cache for Diffusion Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 dKV-Cache —— 首个为 Diffusion Language Models (DLMs) 设计的 KV-Cache 机制。核心实现基于对扩散去噪过程中 token 表征动态的观察：(1) 一旦 token 被解码（从 [MASK] 变为具体 token），其 K/V 表征趋于稳定；(2) token 表征最大的变化发生在解码步和去噪早期。基于此提出 **延迟缓存策略**（delayed caching）：仅缓存已解码 token 的 K/V，掩码 token 每步重新计算；并引入 **一步延迟缓存**（one-step delayed caching）：使用上一步的掩码集合 M_{t-1} 而非当前步 M_t 来决定缓存哪些 token，避免缓存刚解码 token 的不稳定 K/V。设计两种变体：**(1) dKV-Cache-Decode**：近乎无损加速，每 N 步刷新缓存，保持 O(L³) 复杂度；**(2) dKV-Cache-Greedy**：激进缓存，仅对当前 token D_t、上一步 token D_{t-1} 和局部窗口 W(t) 内的 token 计算 QKV，将复杂度降至 O(L²)，代价是轻微性能下降。还提出 dKV-Cache-Prefill 和 dKV-Cache-PD 变体处理长 prefill 场景。底层实现引入 concat_reorder 算子：将缓存 token 重排到序列一端（左侧），避免非连续位置的 gather/scatter 操作，将索引开销从 [B,L,D] 矩阵层级转移到 [B,L] token 层级。

  实验比较：(1) 与 Few-Steps/Half-Steps baseline（减少去噪步数加速）对比（Table 1）：在 LLaDA-8B-Instruct 上，dKV-Cache-Greedy 在 random remasking 下超越 Few-Steps baseline（MMLU 45.77 vs 43.19，GSM8K 67.93 vs 65.58），dKV-Cache-Decode 在 confidence remasking 下接近无损（MMLU 51.00 vs Base 51.11，GSM8K 78.85 vs Base 77.56，HumanEval 46.34 vs Base 39.63 甚至超越）；(2) Dream-7B 实验（Table 2-3）：dKV-Cache-Decode 在 GSM8K 上加速 2.09-4.13×，dKV-Cache-Prefill 加速高达 10.19×（GPQA, L=128），dKV-Cache-PD 在 MBPP 上加速 5.35×；(3) 长 prefill 设置（Table 3）：MLMU T=4 时 dKV-Cache-Prefill 加速 7.40×，GPQA 上 18.91×；(4) 与 cache ratio 和 One-step delay 消融（Figure 3-4）；(5) Memory/Speed tradeoff 分析（Figure 5）：加速 1.75-3.3×，内存开销适中；(6) Batch size 对加速的影响（Figure 7）：batch size=1 时加速有限甚至退步，batch size 增大后加速比显著提升。

- 硬件平台是什么，配置是什么。
  NVIDIA A6000（LLaDA 测速）；NVIDIA H20（Dream 测速）。训练硬件论文未明确说明，方法为 training-free。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaDA-8B-Instruct（masked diffusion LLM，8B 参数）、Dream-Base-7B（从 AR 模型适配的 diffusion LLM，7B 参数）。
  数据集/Benchmark：MMLU（通用语言理解）、GSM8K（数学推理）、Math500（数学推理）、GPQA（研究生级 QA）、HumanEval（代码生成）、MBPP（代码生成）。LLaDA 使用 zero-shot evaluation，Dream 使用 few-shot in-context learning（3-shot/5-shot/8-shot）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/horseee/dKV-Cache（NeurIPS'25 accepted），包含 Dream 和 LLaDA 的修改版模型和生成脚本。

  使用方法：
  ```python
  # Dream 模型加载
  from models.dream import DreamModel
  model = DreamModel.from_pretrained("Dream-7B", use_cache=True, cache_type="decode", cache_steps=4)
  # LLaDA 模型加载
  from models.llada import LLaDAModelLM
  model = LLaDAModelLM.from_pretrained("LLaDA-8B-Instruct", use_cache=True, cache_type="decode", cache_steps=8)
  ```

  算法 Pipeline 伪代码（dKV-Cache-Decode，步 t）：
  ```
  Require: x^{1:L}_{c(t)}, M_t (masked token set), K^{I\M_{t-1}}_{t-1}, V^{I\M_{t-1}}_{t-1}
  1: x' ← x[M_{t-1}]             # 只取未缓存 token（上一步的掩码集），一步延迟
  2: PE' ← [PE[I \ M_{t-1}]; PE[M_{t-1}]]  # 重排位置编码：缓存 token 在左，未缓存 token 在右
  3: Q_t^{M_t}, K_t^{M_t}, V_t^{M_t} ← Transformer(x')  # 仅计算未缓存 token 的 Q/K/V
  4: K_t^I ← Concat(K_{t-1}^{I\M_{t-1}}, K_t^{M_{t-1}})  # 拼接缓存 K 与新计算 K
  5: V_t^I ← Concat(V_{t-1}^{I\M_{t-1}}, V_t^{M_{t-1}})  # 拼接缓存 V 与新计算 V
  6: K_t^{I\M_t} ← Reorder(K_t^I, I'), V_t^{I\M_t} ← Reorder(V_t^I, I')  # 重排序提取下一步缓存
  7: p' ← Attention(Q_t^{M_t}, K_t^I, V_t^I)  # 完整注意力计算
  8: p ← Scatter(p', M_{t-1})  # 将 logits 散播回原始位置
  9: Return p, K_t^{I\M_t}, V_t^{I\M_t}
  ```

  核心张量计算（Eq. 4，dKV-Cache-Decode 的数学形式）：
  ```
  z_t = softmax(Q_t^{M_{t-1}} (K_t^I)^T / sqrt(d_k)) V_t^I
  where:
    K_t^I = concat_reorder(K_{t-1}^{I \ M_{t-1}}, K_t^{M_{t-1}})
    V_t^I = concat_reorder(V_{t-1}^{I \ M_{t-1}}, V_t^{M_{t-1}})
  ```

  dKV-Cache-Greedy 将 M_{t-1} 替换为三个组件的并集：
  ```
  M_t = {D_t} ∪ {D_{t-1}} ∪ W(t)
  where W(t) = {x_i : i ∈ [D_t - ceil(w/2), D_t + floor(w/2)]}  # 以 D_{t-1} 为中心的局部窗口
  ```

  cache ratio 度量：1/T Σ_{i=1}^T |T_i^{cache}| / |T_i|，其中 T_i^{cache} 为步 i 从缓存复用的 token 子集。
  concat_reorder 的关键优化：将 token 序列重排使得缓存 token 在左侧连续排列，从而用 concat（而非 gather/scatter）操作实现 KV 合并，索引开销从 O(BLD) 降至 O(BL)。
