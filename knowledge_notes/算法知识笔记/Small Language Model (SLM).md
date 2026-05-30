## Small Language Model (SLM)

术语解释
参数量在 0.1B-2B 范围内的小型语言模型，专为资源受限设备设计，强调在有限参数预算下实现尽可能高的精度和效率。与 LLM（通常 >7B）相比，SLM 在预训练成本、推理内存、推理延迟和能耗方面有数量级优势。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SLM 的核心设计权衡是在模型容量（深度 × 宽度）和参数预算之间取得最优平衡。三种典型设计策略：
1. **宽度缩减型**（如 baseline1: 22 layers / hidden 1024 / 0.54B）：保持层数不变，缩减 hidden dimension → bottleneck effect（表征能力受限）。
2. **深度缩减型**（如 baseline2: 8 layers / hidden 2048 / 0.52B）：保持宽度不变，缩减层数 → 层次化语言理解能力下降。
3. **参数共享型**（如 MobiLlama: 22 layers / hidden 2048 / 0.52B）：保持高容量架构，通过参数共享机制削减参数 → 当前最优设计策略。

从算法pipeline角度拆解术语，给出具体例子。
MobiLlama 0.5B 完整架构配置：
```
hidden_size = 2048
num_layers = 22
num_heads = 32
intermediate_size = 5632
vocab_size = 32000
max_seq_len = 2048
norm_eps = 1e-6  # RMSNorm epsilon

# 嵌入: 32000 × 2048
# Attention (每层独立): Q/K/V/O projection，GQA kv_heads=4
# FFN (所有层共享): gate/up/down projection，仅 1 份参数
# 总参数 ≈ 0.52B
```

SLM 边界部署性能（MobiLlama 0.5B）：RTX2080Ti bf16: 63.38 tok/s, 3046 MB；i7 CPU 4bit GGUF: 36.32 tok/s, 799 MB；Snapdragon-685 4bit: 7.02 tok/s, 770 MB。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SLM 开发流程：架构设计（选共享策略）→ 预训练（Amber/RedPajama/C4 等公开数据，Flash-Attention 加速）→ 评估（Open LLM Leaderboard benchmarks）→ 量化部署（GGUF 4-bit → CPU/边缘设备）。全透明开源（数据 pipeline + 训练代码 + 模型权重 + 300+ checkpoints + 评估代码）是 MobiLlama 的重要贡献。

涉及论文标题：
- MobiLlama Small Language Model tailored for edge devices
