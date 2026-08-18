## Shape-aware Dispatch（跨平台 kernel 部署：离线生成+验证每平台专属变体，在线按 shape 选最优并回退 vendor library）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Shape-aware Dispatch 是 KernelEvolve 的异构部署策略：因为针对一个平台优化的 kernel 不必然迁移到其他平台（异构平台内存层次、执行模型、编程抽象不同），对每个硬件平台离线（offline optimization phase）生成并验证平台专属 kernel 变体，部署时对每个输入 shape 选择该 shape 下性能最高的变体，生成 kernel 表现不佳时回退到 vendor library（conv1d/conv2d 等）或 PyTorch 基线，确保"自动化合成带来性能收益而不引入回归"（safe production deployment）。论文中的 fallback 触发条件：conv1d 在 out-of-distribution shape（64×768×768×1024）上生成 kernel 仅 0.49-0.63×（相对 PyTorch），此时部署走回退；Optimized FM 对 N≤64 生产 shape 用 fused kernel（2-4×），更大 N（tiling overhead 占优）回退 PyTorch 非融合 baseline；MapId 在 MTIA v2i/v3 的 edge case（batch 2000 时 0.78×）按输入维度 runtime dispatch 回退 PyTorch。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 调度逻辑（伪代码）：
```
# 离线（每平台 p，每算子 op）：生成+验证专属变体
for p in {NVIDIA_H100, AMD_MI350, MTIA_v3, ...}:
    K_p[op] = KernelEvolve(op, hardware=p)   # 图搜索 + 知识库检索生成
    validate(K_p[op])                        # TritonBench 正确性 + speedup
# 在线：shape-aware 选择
def dispatch(op, shape, platform):
    if shape in PROFILED_SHAPES[op] and speedup(K[platform][op], shape) > 1:
        return K[platform][op]               # 生成 kernel 最快
    else:
        return vendor_library[platform][op]  # cuDNN/cuBLAS 或 PyTorch 回退
```
- 具体例子（conv1d 跨 5 平台生产 shape (2048,96,96,200) FP16）：离线为 A100/H100/MI300/MI350/MTIA v3 各生成专属变体（NVIDIA 走 Tensor Core tile 化+3D grid+double-buffer，AMD 走 Infinity Cache 感知 tiling，MTIA 走 SFU/跨 PE/dual-core）→ 在线对生产 shape 选生成 kernel（1.77×/2.30×/1.75×/2.54×/6.54× vs conv1d），对 out-of-distribution shape（64×768×768×1024）自动回退 PyTorch（生成 kernel 在此 shape 仅 0.49×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：离线阶段由 KernelEvolve 的图搜索流水线完成（每个平台用对应解释器 meta_kernel_gpu/amd/mtia_interpreter 编译执行 + TritonBench 测速），部署时 wrapper 内含 shape-keyed 变体表 + fallback 路径；评估 harness 由 evaluation code generator 确定性生成保证一致性。收益：跨平台一致加速 + 无回归风险（论文强调所有 PFFN 配置 speedup ≥1.0、MapId/MBDT edge case 走回退），使自动生成能安全进入生产。局限：离线 profile 只覆盖目标 shape 分布，out-of-distribution shape 表现下降（论文明示"optimization targets production distributions rather than arbitrary inputs"）。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta
