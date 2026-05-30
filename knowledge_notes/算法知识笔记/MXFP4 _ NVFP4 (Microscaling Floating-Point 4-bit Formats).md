## MXFP4 / NVFP4 (Microscaling Floating-Point 4-bit Formats)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MXFP4 和 NVFP4 是新型微缩放（Microscaling）浮点格式，专为硬件加速的 4-bit LLM 推理设计。两者采用分层量化架构——元素分块共享 scale factor——但关键设计选择不同：

**MXFP4（Microscaling FP4，OCP 标准）**：G=32（每组 32 个元素共享一个 scale）、E=FP4 E2M1（1 sign + 2 exponent + 1 mantissa bits，7 个正候选值 + 0）、S=E8M0（scale 量化为 power-of-two，8 bits 全给 exponent、无 mantissa），总计 4.25 bits/element。E8M0 设计简化了硬件乘法操作（power-of-two 缩放等价于指数加法），但 scale 的粗粒度近似（仅 power-of-two 步长）引入额外量化误差。NVIDIA Blackwell B200 和 AMD CDNA4 GPU 支持 MXFP4。

**NVFP4（NVIDIA FP4，Blackwell 架构）**：G=16（每组 16 个元素，更小 group → 更精确的 per-group scaling）、E=FP4 E2M1（同 MXFP4 的 E2M1 基类型）、S=E4M3（完整 FP8 scale，4 exponent + 3 mantissa bits），总计 4.5 bits/element。仅 NVIDIA Blackwell GPU 支持。

两者的核心权衡：MXFP4 以粗 scale（E8M0 power-of-two）换取更低存储（4.25 vs 4.5 bits/elem）和简化硬件；NVFP4 以额外 0.25 bits/elem 换取更精确的 scale 表达（E4M3 full FP8）和更小 group（G=16 更细粒度）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# FP4 E2M1 候选值（对称，7个正+0+7个负）:
# {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}

# MXFP4 (G=32, E=E2M1, S=E8M0) — 4.25 bits/elem
for group in split(tensor, 32):
    s_G = absmax(group)
    s_G_q = E8M0_quantize(s_G)     # power-of-two only
    for x in group: x_q = RTN(x / s_G_q, grid=E2M1)

# NVFP4 (G=16, E=E2M1, S=E4M3) — 4.5 bits/elem
for group in split(tensor, 16):
    s_G = absmax(group)
    s_G_q = E4M3_quantize(s_G)     # full FP8 precision
    for x in group: x_q = RTN(x / s_G_q, grid=E2M1)
```

论文通过 MSE 理论分析揭示的关键发现：Laplace（原生）vs Normal（旋转后）分布下，MSE 收敛率存在 crossover——小 G 时 Laplace MSE 更低（NVFP4 G=16 不应旋转），大 G 时 Normal MSE 更低（MXFP4 G=32 应旋转）。这直接指导了 MR-GPTQ 的设计策略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MXFP4 由 OCP MX Alliance 标准化（ocp-microscaling-formats-mx-v1-0-spec）。NVFP4 为 NVIDIA Blackwell 专有格式。硬件支持：NVIDIA Blackwell GPU 通过 tcgen05.mma 指令支持 NVFP4/MXFP4 矩阵乘法；AMD CDNA4 支持 MXFP4。PyTorch 中通过 fake_quantize 进行模拟量化；真实推理通过 CUTLASS/QuTLASS kernel 库调用硬件指令。

QeRL (Huang et al., NVIDIA, 2025) 将 NVFP4 用于 RL 训练 pipeline：用 AWQ calibration 对预训练 LLM 做 NVFP4 量化（calibration: OpenThoughts-114k），结合 LoRA adapter 进行 GRPO/DAPO 强化学习。QeRL 发现 NVFP4 量化噪声可增加策略熵（H(π(|q))），增强 RL 探索能力。NVFP4 结合 Marlin kernel 实现 1.2-2× rollout 加速，7B 模型仅 5.9GB vs BF16 15.2GB。

涉及论文标题：
- Bridging the Gap Between Promise and Performance for FP4 Quantization
- QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

---
