## FFT-based Token Mixing（FNet 式傅里叶 token 混合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FFT-based token mixing 是用傅里叶变换替换自注意力的 token 交互机制：FNet（Lee-Thorp et al., NAACL 2021）用 2D-FFT 同时沿 token 维与隐藏维做全局混合（固定傅里叶基），移除注意力基于内容（content-dependent）的 pairwise 权重。复杂度 sub-quadratic（O(ND log N) 量级），大幅降低 quadratic attention 的 FLOP 与数据流量。它是 MLX 论文的 baseline 之一（Fig.1(c) 方向："用稀疏注意力或傅里叶变换替换 token mixing"），也是 FABNet（FPGA 蝴蝶加速器）的注意力实现方式。
- 缺陷（MLX 论文指出）：(1) 完全去除内容相关的 token-to-token 交互会伤害精度——2D-FFT 无法适配输入特定的局部或语义依赖；(2) 与 prefill/decode 流水不兼容——cache 增量更新（KV-cache）困难；(3) 无法在标准 LLM pipeline 直接部署。论文实验佐证：FNet 式 2D-FFT（"fnet.fft"）在 ViT 上同 FLOP 削减下比稠密 baseline 损失 2-3% 精度，而 MLX 的 FFT-CMP（保留低频、按层自适应）65% FLOP 削减仅 1.6% 精度下降。MLX 保留傅里叶思想但改成"语义感知 chunked FFT + 低频截断"以保留 informative 分量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FNet 2D-FFT 注意力替代 pipeline（N=序列长、D=隐藏维）：
```
# 对每个 transformer block 的输入 X ∈ R^(N×D)：
F = FFT2D(X)                    # 先沿 token 维再沿隐藏维的 2D FFT（或反之）
X' = real(F)                    # 取实部（FNet 取实部丢弃虚部）
# 下游：X' → 前馈网络；无 Q/K/V 投影、无注意力矩阵、无 KV cache
# 复杂度 O(ND log N log D) ≈ O(ND log N) 量级（对比注意力 O(N²D)）
```
对照 MLX 的 FFT-CMP：按层 chunk（L=N/f_H）内做 1D FFT + 截断 sL + iFFT 得到缩短序列，保留内容信息（低频语义分量）且 decode 可用 append-only 压缩 KV cache——是"傅里叶混合 + 内容保持 + cache 兼容"的折中。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用：FNet 直接把 BERT 的 self-attention 替换为 2D-FFT + 取实部（无需训练改动、收敛更快）；FABNet 硬件上用 2D-FFT 做注意力加速（专用复数蝴蝶单元）。MLX 用它作算法验证与硬件对比基线：算法上对比精度-计算折中（fnet.fft vs bd.* vs FFT-CMP），硬件上 FABNet 重实现对比（MLX 2D-FFT attention 部分 1.11-1.23× 加速、BSMM-FFN 1.21-1.31×）。局限：无内容自适应交互、精度损失（2-3%）、KV-cache 不兼容、GPU 上 FFT kernel 带宽受限（OI 低且低于 roofline）。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures
