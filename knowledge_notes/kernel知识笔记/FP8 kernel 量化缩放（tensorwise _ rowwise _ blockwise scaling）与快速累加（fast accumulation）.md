## FP8 kernel 量化缩放（tensorwise / rowwise / blockwise scaling）与快速累加（fast accumulation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FP8 GEMM 执行前需把高精度输入/权重量化到 8-bit 浮点，量化精度取决于缩放（scale）的粒度——这是低精度 kernel 的"recipe"核心选择维度：tensorwise（整张量共享一个 scale，开销最小但精度最差）、rowwise（每行一个 scale，精度与开销折中）、blockwise（如 DeepGEMM 按 128×128 块每块独立 scale，精度最好、需额外缩放计算）。fast accumulation（快速累加）指 GEMM 累加器用 FP32 而非低精度，避免累加误差累积，是精度-性能权衡的另一个开关。论文对比的三库 recipe：TorchAO（tensorwise TW / rowwise RW / 混合 RW GW HP——前向 rowwise、输入梯度高精度、权重 backward tensorwise）、DeepGEMM（blockwise BW）、FBGEMM（rowwise RW）。
- LoKA 的量化开销实证（H100，27 个生产 LRM shape，从 (2048,256)@(256,768) 到 (2048,123200)@(123200,1024)）：端到端 FP8 相对 BF16 平均仅 1.6×；最大有效 TFLOPS < 硬件峰值 20%；量化开销占端到端 GEMM 延迟 >30%；计入 layout 操纵等内存分配开销后 FP8 可能差于 BF16——量化/反量化与布局转换是低精度收益的主要吞噬者。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次 FP8 GEMM kernel 执行流程（以 rowwise 为例）：输入 x∈R^{M×K}、W∈R^{K×N} → 量化 kernel：对 x 每行求 max→scale_x[i]=max|x[i,:]|/127，x8[i,:]=round(x[i,:]/scale_x[i])（W 按列或块同法）→ FP8 张量核 GEMM：C_fp32[m,n]=Σ_k x8[m,k]·W8[k,n]（FP32 快速累加）→ 反量化：C[m,n]=C_fp32[m,n]·scale_x[m]·scale_w[n]（tensorwise 为单标量相乘；blockwise 需按块拼装）。
- 纯计算 vs 端到端拆解：纯 GEMM 吞吐（只计时张量核部分）掩盖量化/反量化/布局成本；端到端含 quantize→layout→GEMM→dequantize→写回。论文 Fig.4 显示二者差距即量化开销占比（>30%），且 shape 越小开销占比越高（小 GEMM 是 LRM 主导形态）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现载体：DeepGEMM（https://github.com/deepseek-ai/DeepGEMM，块级 scaling 的 FP8 GEMM，CUDA 手写 kernel + 数值测试）、FBGEMM（https://github.com/pytorch/FBGEMM，INT8/FP8 高性能力矩阵 kernel）、TorchAO（https://github.com/pytorch/ao，PyTorch 原生低精度训练到服务优化库，drop-in 替换层）。使用：库提供不同 recipe 选项，用户/调度器按 shape 与精度需求选择；LoKA Dispatch 即自动做此逐算子选择（见 LoKA Dispatch 条目）。局限：scaling 粒度越细精度越好但量化/缩放开销越大，这正是"库优化解决不了、需系统层编排"的原因。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale
