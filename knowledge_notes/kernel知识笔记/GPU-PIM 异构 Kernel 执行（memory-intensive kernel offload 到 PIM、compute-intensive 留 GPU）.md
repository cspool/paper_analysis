## GPU-PIM 异构 Kernel 执行（memory-intensive kernel offload 到 PIM、compute-intensive 留 GPU）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
异构 xPU-PIM 系统中按 kernel 访存特征划分执行位置：compute-intensive kernel（GEMM、卷积，如 LLM 的 FFN 主体）跑在 Host GPU；memory-intensive kernel（GEMV、element-wise、attention 的 QK^T/SV）offload 到 PIM core，利用 bank 级聚合带宽。PIM 设备有两种工作模式：作为标准 DRAM 供 Host load/store（Host 执行 kernel），或由 Host 把 kernel offload 给 PIM core 独占访问本地 bank 执行。选择依据：GEMV 的算术强度低（每元素约 2 FLOP、受外部带宽主导），PIM 内部带宽远高于外部 I/O，offload 收益大；GEMM 权重复用高、GPU 张量核算力强，留在 GPU。DCC 的适用边界即"适合 PIM 的 memory-intensive kernel"（GEMV/RED/ATTN/VA/RELU）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LLM 解码一层的执行划分（DCC + AttAcc_Full）：
```
# GPU 侧：QKV 生成与 projection 在 AttAcc 默认实现中留 GPU（固定 tiling 落后 GPU 1.25x）
# DCC 侧：QKV/attention/projection 的 GEMV 形态全部移到 PIM
for b in batch, i in seqlen:            # GEMV：weight 常驻 PIM bank
    PIM_group_broadcast_MAC(weight[i], x[b], acc[b])   # 每 bank GEMV 单元并行
PIM_softmax_unit(score)                 # AttAcc 每 channel softmax 单元
for b in batch, i in seqlen:            # SV
    PIM_group_broadcast_MAC(v[b], p[b], out[b])
# GPU 侧：FFN 的大部分 GEMM 仍在 GPU（compute-intensive）
```
关键调度决策（DCC 自动化）：每 kernel 选多少 PIM 组/核、张量维如何切（data-tile）、循环如何映射（compute-tile）、重排如何做——同一 GEMV 在不同 batch/tensor 尺寸下最优分区不同（batch 1→8 时 DCC 对 HBM-PIM 增益 1.33×→1.58×、AttAcc 1.14×→1.31×）。AttAcc 默认实现用固定分布（batch/head→16 pCH、第一维→16 bank group、第二维→4 bank）应对所有尺寸，是 QKV/projection 层被迫留 GPU 的原因；DCC 联合搜索后把这两层移上 PIM，分别 2.58×/2.91× 对 GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译器生成"重排指令 + group/bank 级 DRAM 计算命令 + 结果回收"三段调度（见 PIM 数据重排、Group/Bank 控制命令条目）；DCC API：@DCC_kernel 定义 PIM kernel → DCC.Layer 替换模型层 → DCC.Kernel.preLoad 预载权重到 PIM → forward() 执行并返回 torch.Tensor。使用：memory-bound 层（attention/GEMV 形态）放 PIM、compute-bound 层（GEMM/FFN）放 GPU；DCC 与 GPU 侧 ML 编译器（TVM 等）经 PyTorch 协同（PIM 层标为 non-fusible）。效果：kernel 级对 GPU 至多 13.17×（AttAcc）/7.68×（HBM-PIM），LLM 端到端平均 4.52×。

P3-LLM 补充视角（ISCA'26，NPU-PIM 异构算子映射）：P3-LLM 把异构执行扩展到"NPU（主机）+ HBM-PIM"且量化感知的算子映射——decode 阶段 NPU 只保留高精度元素级算子（RoPE、Softmax），线性层 GEMM/GEMV 与注意力矩阵乘（Q·K^T、P·V）offload 到低精度 PCU；Q·K^T 的映射取决于 key cache 的量化位置：pre-RoPE 量化（Llama-1/2 短序列）的 key 缺位置信息，需每轮在线 RoPE（元素级、开销可忽略）并留在 NPU 高精度计算；post-RoPE 量化（Llama-3/Mistral 长序列）的 key 可直接与 query 相乘、Q·K^T 上 PCU。算子映射伴随量化算子融合（见去量化隐藏条目）：把 per-channel 平滑因子 SSF 融合进 query 的 FP8 量化缩放、把 per-value-head 缩放 S^V/S^V_max 融合进 attention-score、线性层 dequant 后置到 GEMM 之后，使 PIM 侧全程消费量化操作数、NPU 侧不出现逐操作数的在线 dequant。batch≥8 时线性层在 PIM 变 compute-bound，P3-LLM 将线性层 offload 回 NPU、attention 层（GQA 复用低）继续留 PIM——异构映射是动态的（Fig.16 大规模解码实验）。

涉及论文标题：
- DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
