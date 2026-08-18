## Tree-based Speculation（树形投机解码：draft budget / tree width / masked attention）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
投机解码的一类扩展：draft 模型每步提出多个候选 token 形成树（候选树），target 用带掩码的注意力（masked attention）一次前向并行验证整棵树的候选，提高接受概率与单轮产出（EAGLE、Medusa、Lookahead 等 [3][38][48][64]）。两个控制参数：draft budget（每轮候选树的总 token 预算）与 tree width（每层分支数/每步候选数）；chain 式是 width=1 的特例。HybridSpec 用它做运行时调制的杠杆：(1) budget 增大接受长度先增后饱和（图 10(a)，超出阈值收益递减，冗余计算在拒绝候选上浪费）→ 设上限 B；(2) 固定预算下接受长度随 width 先增后减（图 10(b)——宽树覆盖更多候选但单支变浅，探索 vs 深度权衡）→ 用 SVR 拟合 (budget→最优 width) 查找表。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 每轮（width=p、预算 B_t 的候选树）
HB 栈（draft）: 按 tree width 并行/自回归扩展候选树（逐 token，masked 位置）
    current_size += tree_width; 若 current_size <= draft_budget 继续扩展
XPU（target）: VerifyTask(draft_budget) —— masked attention 一次前向验证整树
接受: 沿树保留被接受的最长前缀；拒绝分支的 KV cache 清除
```
参数调制（Algorithm 1）：HB 栈算术强度低于 roofline 时 tree_width+1（≤p，多探索吃满带宽）、否则 -1；XPU 未满时 draft_budget ×2（≤B）、否则 ÷2。实测预算/宽度随请求率从 (30.74,3.72) 降到 (9.25,1.58)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：候选树以树/序列集合表示，验证用自定义 attention mask 指示各候选的位置前缀；开源实现见 EAGLE-3/Medusa/Lookahead（vLLM/SGLang 支持）；HybridSpec 的 Fig.10 数据来自 [64] 的开源实现测量。使用要点：width 与 budget 存在"探索-深度"权衡，需按目标模型接受特性离线拟合、运行时按利用率在线调制。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
