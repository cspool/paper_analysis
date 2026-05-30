## DartQuant Efficient Rotational Distribution Calibration for LLM Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  DartQuant 提出基于分布的旋转矩阵校准方法用于 LLM Post-Training Quantization（PTQ），由三部分组成：(1) **Rotational Distribution Calibration**：将旋转矩阵优化重新定义为将激活变换到最适合量化的分布；(2) **Whip loss**：`Whip = Σ exp(-|x_i|)`，驱动旋转后激活趋向均匀分布，减少量化误差；(3) **QR-Orth**：通过 QR 分解保证正交性（`R = QR(Z)`，优化隐参数 Z 替代直接在 Grassmannian 流形上优化 R），避免 Cayley SGD 等复杂黎曼优化器。
  实验比较 RTN、SmoothQuant、GPTQ、OmniQuant、QuaRot、SpinQuant、OSTQuant 在 4-8-16、4-4-16、4-4-4 比特设置下的 WikiText2/C4/PTB PPL 和 9 项零样本任务准确率，以及旋转矩阵优化的 GPU 时间和内存开销。

- 硬件平台是什么，配置是什么。
  NVIDIA A800 GPU 服务器（主实验和表 3 时间/内存对比）。单卡 NVIDIA RTX 3090（展示 70B 校准可行性，约 3 小时完成，23.47 GiB 内存）。

- 模型是什么。数据集和bench分别是什么。
  模型: Llama-2 (7B/13B/70B)、Llama-3 (8B/70B)、Mixtral-8×7B (MoE)、DeepSeek-MoE-16B。
  数据集和 benchmark: PPL 用 WikiText2、C4、PTB；零样本用 LAMBADA、HellaSwag、PIQA、WinoGrande、OpenBookQA、SIQA、MMLU、ARC-Easy、ARC-Challenge。
  校准集: 128 samples from WikiText2，sequence length 2048。GPTQ weight reconstruction 使用相同校准集，per-token asymmetric 激活量化。Token sampling ratio 为 10%。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源: https://github.com/CAS-CLab/DartQuant.git

  **DartQuant 核心算法（Algorithm 1）：**
  ```
  输入: LLM model, 校准序列 S, 隐参数 Z_0 ∈ R^{n×n}, 最大迭代 T, 学习率 η
  输出: 旋转矩阵 R ∈ R^{n×n}

  1. X ← LLM(S)                     // 前向传播收集所有指定层的激活
  2. X ← token_sampling(X)          // 随机采样 10% token 以减少计算量
  3. Z ← Z_0                        // 初始化隐参数为随机 Hadamard 矩阵
  4. for k = 0 to T do
  5.     R ← qr_decomposition(Z)    // QR-Orth: 通过 QR 分解获得正交旋转矩阵
  6.     O ← X @ R                  // 激活旋转: O = XR
  7.     L ← Whip(O)                // Whip loss = Σ_{i=1}^{C_in} exp(-|o_i|)
  8.     Z ← Z - η ∂L/∂Z            // 标准 SGD/Adam 更新隐参数（无需约束优化器）
  9. end for
  ```

  **旋转矩阵在 Transformer 中的融合（Computational Invariance）：**
  - R_1（可学习，DartQuant 优化）：右乘 W_q, W_k, W_v, W_up, W_gate；R_1^T 左乘 W_out, W_down, W_embedding；R_1 右乘 W_lm_head
  - R_2（可学习，DartQuant 优化）：插入 W_v 和 W_o 之间，R_2 融入 W_v，R_2^T 融入 W_o
  - R_3（在线 Hadamard）：在 attention score 计算中在线执行，抵消 KV cache 量化损失（因 RoPE 存在无法融合入权重）
  - R_4（在线 Hadamard）：在 FFN down-projection 前在线执行（因 gating 机制无法融入 W_up/W_gate）

  **Whip Loss 机制：** 激活向量 x ∈ R^{C_in}，Whip = Σ exp(-|x_i|)。该函数在零附近有较大梯度，将接近零的小值推开；在 norm-invariance 约束（||Rx|| = ||x||）下，小值被迫增大 → outliers 被迫减小以保持 L2 范数不变 → 整体分布趋向均匀。灵感来自 Laplace→Uniform 的 CDF 变换：U_X(x) = τ[exp(x/b)-1] for x≤0。

  **QR-Orth vs Cayley SGD 计算复杂度：**
  Cayley SGD 额外计算量约 6n³（矩阵乘法+投影），QR-Orth 仅需 QR 分解约 4/3 n³。100 步 SGD 耗时：QR-Orth 5.7h vs Cayley 8.2h（1.44× 加速）；QR-Orth SGD 仅 6 步即达到 Cayley SGD 100 步同等效果（41× effective 加速）。

  **推理流程：** 所有权重 INT4 存储，激活在矩阵乘法前量化为 INT4 → TensorCore INT4×INT4 矩阵乘产生 INT32 → 立即转换为 FP16（含 scale）。R1, R2 预融合无推理开销，R3, R4 使用快速 Hadamard kernel 在线计算。

  **关键结果：** 70B 模型校准时间从 SpinQuant 42.9 GPU-hours → DartQuant 0.91 GPU-hours（47× 加速），内存从 238.89 GiB → 23.47 GiB（10× 节省）。首次在单卡 RTX 3090 上完成 70B 旋转校准（~3h）。Llama-3-70B w4a4kv16 零样本 avg loss 仅 3.31%（vs SpinQuant 6.64%, OSTQuant 4.76%）。
