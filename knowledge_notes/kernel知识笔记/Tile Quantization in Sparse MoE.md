## Tile Quantization in Sparse MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tile Quantization 指 GPU GEMM 必须 pad 矩阵维度到 tile size 整数倍时产生的计算浪费。对于 MoE：expert e 收到 T_e 个 token，若 T_e mod M_tile ≠ 0，需 pad 到 ceil(T_e/M_tile)×M_tile 个 token，padding 位置的 GEMM 计算全部浪费。稀疏 MoE 下 T_e 很小（如 E=256, T=16K, K=4 时 T_e≈250, M_tile=128 需 2 tiles=256，waste 6/256≈2.3%），但绝对浪费随稀疏度增加。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MoE forward+backward FLOPs = (6+12)T_e·n·d per expert。当 E 从 32 增至 256（保持 T 和 K 不变），T_e 从 2000 降至 250，tile quantization waste 从 ~2.3% 升至 ~2.3%（比例相近但绝对 tile 数翻倍）。实际影响不仅 FLOPs——小 M tile 降低 SM occupancy 和 TMA efficiency。SonicMoE Figure 8 显示 T=16k, d=4k, n=1k, K=4 下 waste 随 E 增长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
解决：(1) Token Rounding（SonicMoE）：routing 阶段将 f_e 舍入到 M_tile 倍数；(2) Token Dropping：丢弃超 capacity token；(3) Dynamic tile shape。SonicMoE TR 在 K/E ≤ 1/64 时带来 16% kernel TFLOPS 提升。

涉及论文标题：
- SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations
