## MicroMix Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - MicroMix 提出基于 Microscaling (MX) 数据格式的混合精度量化算法，支持 MXFP4 (E2M1)、MXFP6 (E3M2/E2M3)、MXFP8 (E4M3/E5M2) 三种精度，对每个线性层自适应分配三种精度通道比例。核心算法机制：(1) 通过排列（permutation）将激活通道按绝对值均值升序重排；(2) 定义量化阈值 T(4) 和 T(6)（基于 INT8 量化误差上界推导，确保 MXFP 量化误差不超 INT8 上界），将超过阈值的元素分配到更高精度；(3) 离线计算每层的 p4/p6/p8 比例和排列 σ。推理时激活 online 执行 fused reorder-and-quantize，权重 offline 预量化。GEMM 使用 CUTLASS MXFP kernel，各精度分组独立计算后拼接。
  - 实验比较了 QuaRot (W4A4)、QUIK (mixed 4/8)、Atom (mixed 4/8 with 128 INT8 channels)、FlatQuant (W4A4)、AMXFP4 (MXFP4 only)、INT6 baseline，以及 FP16 参考。比较维度：(a) zero-shot accuracy (ARC_C, BoolQ, Lambada, PIQA, Winogrande); (b) 5-shot MMLU; (c) WikiText2 PPL; (d) 代码生成 (HumanEval, MBPP); (e) 数学推理 (GSM8K, MATH, MMLU-STEM, CMATH); (f) 单 kernel 延迟; (g) prefill/decode 端到端性能; (h) 峰值内存占用。
- 硬件平台是什么，配置是什么。
  - NVIDIA RTX 5070Ti Laptop GPU（Blackwell）、RTX 5090（Blackwell）、RTX PRO 6000（Blackwell），均支持 FP4 Tensor Cores。Blackwell FP4 Tensor Core 吞吐为 FP16 的 4×、FP8/INT8 的 2×。
- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama3.1-8B、Qwen2.5-32B、Qwen2.5-Coder-14B/32B-Instruct、Qwen2.5-Math-7B-Instruct、Mixtral-8x7B-v0.1-Instruct。
  - 校准数据集：WikiText2、Pile、C4（校准采样 32 条，覆盖 batch size 8/16/32/64 和 sequence length 512/1024/2048/4096）。
  - Zero-shot benchmarks (lm-eval)：ARC_C、BoolQ、Lambada、PIQA、Winogrande。
  - 5-shot：MMLU。
  - PPL：WikiText2。
  - 代码 benchmarks：HumanEval、HumanEval+、MBPP、MBPP+。
  - 数学 benchmarks：GSM8K、MMLU-STEM、CMATH、MATH。
- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/lwy2020/MicroMix
  - 算法流程（伪代码）：
    ```
    # === 离线校准阶段 ===
    for each linear layer k in model:
      # 用校准数据前向传播，收集该层输入激活 X^k ∈ R^{L×I}
      M^k[j] = (1/L) * Σ_i |X^k[i,j]|  for j = 1..I  # 通道绝对值均值
      σ^k = argsort(M^k)  # 升序排列索引

      # 计算阈值：T(n) = 2^b * 2^(n-1)/q_max * max(|X|)/254
      # 对 MXFP4 (E2M1): n=4, b=1, q_max=6
      T(4) = 2^1 * 2^3/6 * max(|X|)/254 = max(|X|)/95.25
      # 对 MXFP6 (E3M2): n=6, b=3, q_max=28
      T(6) = 2^3 * 2^5/28 * max(|X|)/254 = max(|X|)/27.8

      # 分组：按重排后的值划分
      X_sorted = X^k[:, σ^k]
      G4 = X_sorted[:, 0:p4*I]  where |x| ≤ T(4)
      G6 = X_sorted[:, p4*I:(p4+p6)*I]  where T(4) < |x| ≤ T(6)
      G8 = X_sorted[:, (p4+p6)*I:I]  where |x| > T(6)

      # 存储配置 (p4^k, p6^k, p8^k, σ^k)

    # === 量化阶段 ===
    # 权重量化（离线，一次性）：
    W^k_reordered = W^k[σ^k, :]  # 按激活排列重排
    for block in W^k_reordered.reshape(-1, 32):  # block_size=32
      s = 2^{floor(log2(max(|block|))) - b}  # E8M0 scale
      Q(block) = round(clip(block/s, -q_max, q_max))

    # 激活量化（在线，fused reorder-and-quantize kernel）：
    X_quant = fused_reorder_and_quantize(X^k, σ^k, p4^k, p6^k, p8^k)
    # 输出三组 MX 格式张量 [G4_mxfp4, G6_mxfp6, G8_mxfp8]

    # === GEMM 推理 ===
    # 各分组独立执行 CUTLASS MXFP GEMM
    Y4 = MXFP4_GEMM(G4_mxfp4, W4_mxfp4)  # FP4 Tensor Core, MMA fused dequant
    Y6 = MXFP6_GEMM(G6_mxfp6, W6_mxfp6)  # FP6 MMA on Tensor Cores
    Y8 = MXFP8_GEMM(G8_mxfp8, W8_mxfp8)  # FP8 MMA on Tensor Cores
    Y = concat_and_reorder_back(Y4, Y6, Y8, σ^k)  # 恢复原通道序，输出 BF16
    ```
  - 关键结果：Llama3.1-8B 平均 5.51 bits，zero-shot avg 71.56 (FP16: 73.03)，MMLU 62.65 (FP16: 65.24)。Qwen2.5-32B 平均 5.22 bits，zero-shot avg 75.20 (FP16: 75.55)，MMLU 81.79 (FP16: 83.32)——近乎无损。Mixtral-8x7B 精度 drop <0.4 分，执行时间从 5m18s 降至 2m03s。
