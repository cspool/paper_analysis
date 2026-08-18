## GEMM epilogue 融合（归一化 / 激活 / 量化融合进 GEMM 尾处理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GEMM epilogue 是 kernel 在张量核算完矩阵乘后、把累加结果写回 HBM 前的尾处理阶段：通常含反量化/缩放、加 bias、激活、写回。epilogue 融合（epilogue fusion）指把归一化、激活、量化/反量化等后续算子合并进该阶段，在输出 tile 尚在片上（L1/L2/寄存器）时完成，避免中间张量往返 HBM——这是降低低精度开销的关键手段（Triton 等框架原生支持 epilogue 融合）。
- LoKA 的用法：BlockNorm（块级 RMS）设计目标就是把归一化融合进 GEMM epilogue——GEMM 输出 tile 在片上时立即按 256 元素块算 RMS 并归一化，再融合 Hard Swish 与反量化后写回；相比标准归一化（先写 HBM、读回、算全局统计、再写），省掉两轮全局内存流量。约束：融合要求归一化统计在单 thread block 内可算（Case 1 大 batch 小 N 成立；Case 2 小 batch 大 N 需跨 block 同步，收益被抵消，BlockNorm 用固定块规避）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 融合 kernel 尾处理流程（一次 Wukong 线性层）：张量核算完 C_tile∈R^{BM×BN}（BM×BN 为 tile 尺寸，驻留 SMEM/寄存器）→ 反量化 C=(C_fp32·scale_x·scale_w) → 块归一化：C.view(BM, BN/256, 256)，每块 rms=sqrt(mean(block²)+ε)，C_b/=rms → Hard Swish：C_b·clamp(C_b+3,0,6)/6 → 写 HBM。全过程零全局内存往返。
- 对比未融合路径：GEMM→写 HBM→（下一 kernel）读回→全局统计 LayerNorm→激活→写 HBM→（下一 kernel）读回量化……每多一次往返就多一次 HBM 流量与 kernel 启动开销，小 GEMM 下开销占比更高。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Triton（kernel 内 tl.dot 后直接算归一化/激活）、CUDA 手写 kernel 的 epilogue 段、torch.compile 的算子融合 pass（fuse attention/norm）。LoKA 通过把归一化改成 BlockNorm 使统计本地化，从而让融合可行（标准 LayerNorm 沿特征维全局统计无法在 tile 内完成）。使用：低精度/小 GEMM 场景最大化片上复用；论文 Fig.13 消融显示 BlockNorm 因使融合可行而带来显著延迟降低。参考：epilogue fusion 通用概念见 Triton/FlashAttention 相关工作。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale
