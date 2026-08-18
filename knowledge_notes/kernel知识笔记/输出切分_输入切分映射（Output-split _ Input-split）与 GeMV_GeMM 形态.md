## 输出切分/输入切分映射（Output-split / Input-split）与 GeMV/GeMM 形态

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把一层矩阵乘法 W·X 切分到多个 PIM bank 的两种策略：output-split = 按输出维把 W 的行切给各 bank，各 bank 算自己的输出段，输入 X 需广播到所有 bank（无归约、有广播）；input-split = 按输入维切 W 的列与 X 的段，各 bank 算部分和，需跨 bank 归约（有归约、无广播）。GeMV（广义矩阵向量乘，batch=1 形态）与 GeMM（batch>1）是同一算子在不同 batch 下的两种形态：GeMV 内存受限（每字节权重只做一次乘加）、GeMM 计算受限。DRAM-PIM 传统回避 input-split——global buffer 的归约带宽有限且需串行访问 bank——被迫用 output-split，但 output-split 造成形状失衡：CompAir 中 Llama2-13B Q/K/V 每 bank 权重为 5120×10（输入输出比 >17:1），输入广播代价大。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CompAir 的分析流程：① 对每个 FC 算子判断形态——batch 增大时 Q/K/V 投影从 GeMV 转 GeMM，SRAM-PIM 收益出现（batch=32 6.3×、batch=1 无收益）；② SRAM-PIM 偏好平衡映射（均值不等式：输入输出维相近时带宽需求最小），(512,8) output-split 形状失衡、(256,16) input-split 降带宽压力；③ CompAir-NoC 归约树消除 global buffer 串行归约后，input-split 可行甚至更优（2560×20 一致优于纯 output-split）。伪代码（bank b 视角）：
```
# output-split：X 广播给所有 bank，无归约
Y[b] = W[b] @ X          # 每 bank 输出段
# input-split：X 分段，部分和经 Reduce 树归约
Y = Reduce('+', [W[:, b] @ X[b] for b in banks])
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPU 上对应概念是权重/输入切分 + all-reduce；PIM 上归约经 global buffer（串行、慢）或 NoC 树（并行、快）完成。选择规则（CompAir）：按算子形态选硬件（GeMM→SRAM-PIM、GeMV→DRAM-PIM）；切分维按归约成本与广播成本权衡；TP 沿 seqlen 切 K^T/V 时 seqlen 映射为 SRAM-PIM 的 batch 维、输出维对齐 GQA group size。Qwen 8K 采用 input-split 使本地指令 +27%、但经紧凑的 NoC_Reduce 稀释为系统级 +2%。

涉及论文标题：
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
