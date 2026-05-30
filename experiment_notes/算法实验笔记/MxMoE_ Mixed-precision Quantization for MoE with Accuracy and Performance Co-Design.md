## MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MxMoE 是一个面向 MoE 模型的混合精度量化框架，核心包含两部分：(1) **硬件感知的 bitwidth 分配**：对 MoE block 中每个 expert 的每个 linear block（gate_proj, up_proj, down_proj），在由参数量化敏感度 Δ_{i,j,k}、expert 激活频率和硬件资源构成的多维设计空间中，通过 ILP（整数线性规划）求解最优混合精度量化方案。目标函数为 min L^r · T^{1-r}，其中 L 为量化输出扰动（Euclidean distance）、T 为基于 tile 级 profiling 的执行时间，r 为精度-性能权衡超参数（weight-only 极低比特 r=1，weight-activation r=0.75）。求解后使用 randomized Hadamard 变换 + GPTQ-based 量化完成权重 quantization；(2) **自动混合精度 Group-GEMM kernel 生成**：micro-kernel specialization + resource configuration + tile scheduling。量化方案以 linear-block 为粒度（而非 expert 级），校准使用 WikiText2 的 128 sequences × 4096 tokens。
  - 实验比较：
    - Weight-only quantization（2.25-bit / 3.25-bit 平均位宽）：MxMoE vs GPTQ（含相同 random Hadamard 变换预处理），在 DeepSeek-V2-Lite、Qwen1.5-MoE、Qwen2-MoE、Mixtral-8×7B 上对比 WikiText2 perplexity 和 7 个零样本任务（AC/AE/HS/LO/LS/PQ/WG）的 Avg Acc
    - Weight-activation quantization（5-bit 平均位宽）：MxMoE vs QuaRot（4-bit uniform），r=0.75
    - Ablation：linear-block vs expert-level bitwidth allocation granularity；超参数 r 对 accuracy-performance tradeoff 的影响
    - 性能分析：MoE block 计算吞吐量（memory-bound 512 tokens / compute-bound 8192 tokens），混合精度 vs uniform precision vs full-precision

- 硬件平台是什么，配置是什么。
  - 单张 NVIDIA RTX 4090 GPU（24GB 显存）

- 模型是什么。数据集和bench分别是什么。
  - 模型：DeepSeek-V2-Lite（16B 总参，2.4B 激活，64+2 experts, TopK=6）、Qwen1.5-MoE（14.3B 总参，2.7B 激活，60+4 experts, TopK=4）、Qwen2-MoE-Instruct（54.2B 总参，7B 激活，64+8 experts, TopK=8）、Mixtral-8×7B-Instruct-v0.1（46.7B 总参，12.9B 激活，8 experts, TopK=2）
  - 校准数据：WikiText2 训练集，128 条 sequence，每条长度 4096
  - Perplexity 评测：WikiText2
  - 零样本评测：Arc-Challenge, Arc-Easy, HellaSwag, LAMBADA-openai, LAMBADA-standard, PIQA, WinoGrande
  - Expert 激活频率统计：HumanEval-X 数据集

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/cat538/MxMoE
  - 算法 pipeline 核心流程（伪代码）：
    ```
    === 阶段 1: 离线校准与统计收集 ===
    输入: MoE 模型 (M 层), 校准数据 X_cal (128 seqs × 4096 tokens)

    1. 对每个 MoE block 的每个 linear-block (expert i ∈ [1,E], block j ∈ [1,N]):
       for each supported quantization scheme k ∈ S:
           计算量化扰动 Δ_{i,j,k}:
             W_q = GPTQ_quant(W_{i,j}, scheme k)
             Ô = MoE_block_forward(X_cal, with W_q at (i,j), rest FP16)
             O = MoE_block_forward(X_cal, all FP16)
             Δ_{i,j,k} = ||Ô - O||₂

    2. 统计 expert 激活频率 f_i = P(expert i | token)

    === 阶段 2: ILP 求解混合精度方案 ===
    3. 构建 ILP:
       minimize L^r · T^{1-r}
       where:
         L = Σ_{i,j,k} Δ_{i,j,k} · x_{i,j,k}
         T = (1/P) · Σ_{i,j,k,t} c_{i,j,k,t} · y_{i,j,k,t} · x_{i,j,k}
       s.t.:
         x_{i,j,k} ∈ {0,1}, Σ_k x_{i,j,k} = 1  (每 linear block 一方案)
         y_{i,j,k,t} ∈ {0,1}, Σ_t y_{i,j,k,t} = 1  (每 linear block 一 tile 配置)
         Σ_{i,j,k} W_{i,j,k} · x_{i,j,k} ≤ M  (内存预算)

    === 阶段 3: 量化执行 ===
    4. 按 ILP 输出 {x_{i,j,k}} 逐 linear block 量化:
       - randomized Hadamard 变换 (incoherence processing, 来自 QuaRot)
       - GPTQ per-channel/per-group symmetric/asymmetric min-max quantization
       - 权重离线完成; 激活运行时动态量化
    ```
  - 关键参数：S = {W2A16, W4A16, W4A4, W4A4-g128, W8A8, ...}；平均位宽 2.25/3.25（weight-only）或 5（weight-activation）；r=1 或 0.75
  - 核心结果：2.25-bit 下 WikiText2 PPL 比 GPTQ 低 2.4（DeepSeekV2-Lite）；W5A5 速度比 full-precision 快 3.4×，比 uniform W8A8 快 29.4%
