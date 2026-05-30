## QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

- baseline方法是什么？
  - Baseline 方法：(1) **QuIP** (Chee et al., 2023)：使用 Kronecker 积构造的 2-factor 正交矩阵（U = U_1 ⊗ U_2, V = V_1 ⊗ V_2）做非相干处理，复杂度 O(n√n)；标量 LDLQ 按列自适应舍入（一次一列）；舍入目标为半整数格（1D）；无微调。(2) **OmniQuant** (Shao et al., 2024)：通过学习可微的模型保持变换（model-preserving transformations）按 Transformer Block 减少离群值，启发式方法在低比特下失效。(3) **AWQ** (Lin et al., 2023)：在量化前按激活幅度缩放权重，2.15-bit 即崩溃。(4) **AQLM** (Egiazarian et al., 2024)：使用可学习非结构化 8D 向量量化码书（每层一个 2^16×8 码书占 1MiB），码书太大无法放入 L1 cache 导致推理慢于 FP16。
  - 全栈执行例子（Baseline: QuIP 2-bit on Llama 2 7B）：
    - **算法pipeline**：Llama 2 7B FP16 权重加载 → 计算代理 Hessian H（RedPajama 6144 seqs × 4096 ctx）→ Kronecker 非相干处理：随机生成正交矩阵 U_1,U_2（≈√n 维）和 V_1,V_2（≈√m 维），构造 U=U_1⊗U_2, V=V_1⊗V_2 → Ŵ ← UWV^T, Ĥ ← VHV^T → LDLQ 按列标量舍入：对 H 做 LDL 分解 H=L^TDL，设置 U=L^T-I，逐列 Ŵ_k = round(Ŵ_k + (Ŵ_{:(k-1)} - Ŵ̂_{:(k-1)})a_k) → 推理：激活 x → Vx → 量化权重矩阵乘法 → U^T(quantized(Ŵ)(Vx))。主要缺陷：(a) Kronecker μ_W 依赖 log²(mn/δ)，不如 RHT 的 log 依赖；(b) 标量舍入产生的可表示权重向量形成超立方体，与 RHT 变换后球状高斯分布不匹配；(c) 无微调导致量化误差仅局部最小化。
    - **系统框架**：无特定 Serving 框架修改，量化后模型以 PyTorch 推理。离线 PTQ 流程。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：标准 FP16 矩阵乘法 + Hadamard 变换乘法。无定制压缩解码 kernel。QuIP 标量量化权重可直接用标准 INT 运算。
    - **硬件架构**：GPU（A100 量化 / A6000 推理），标准 CUDA 路径。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QuIP# = **RHT 非相干处理** + **BlockLDLQ + E8P 格基码书向量量化** + **层间微调**。三大组件分别解决 Baseline 核心缺陷：
    (1) **RHT 取代 Kronecker**（解决非相干效率与理论界问题）：用 Randomized Hadamard Transform（Had(S·x)）替代 QuIP 的 2-factor Kronecker 积。优势：(a) μ_W = 2log(4mn/δ) vs Kronecker 的 A²log²(4Cmn/δ)²——对数依赖替代对数平方依赖；(b) 时间复杂度 O(n log n) vs O(n√n)；(c) Hadamard 矩阵元素为 {±1}，乘法无需浮点运算，常数因子更低；(d) 消融实验（Table 4 "no E8" 行 vs QuIP 行）验证 RHT 独立于其他组件即带来显著困惑度改善（2-70B 2-bit: 4.58 vs 5.90 Wikitext2）。
    (2) **BlockLDLQ + E8P 格基码书取代标量舍入**（解决分布匹配问题）：(a) BlockLDLQ 将 QuIP 的标量 LDLQ 推广到 g 列块级向量量化——基于 g-block LDL 分解 H=L^TDL，设置 U=L^T-I，按 8 列块迭代 Ŵ_k = Q(Ŵ_k + (Ŵ_{:(k-1)} - Ŵ̂_{:(k-1)})A_k)——Theorem 4.1 给出误差界 ∝ gmμ²σ² tr(H^{1/2})²/n；(b) E8P codebook 基于 E8 格（8 维最优球填充密度，kissing number 最优），通过符号翻转对称性将 2^16 条目压缩为 2^8 源码书（1KiB = L1 cache fit），球状码书形状匹配 RHT 变换后的亚高斯权重分布；(c) RVQ 扩展高比特：4-bit = 2×2-bit E8P 残差量化；(d) E8P 在相同的 2-bit 下显著优于标量半整数舍入（2-70B: 4.16 vs 4.58 Wikitext2）。
    (3) **层间微调取代纯 PTQ**（解决层间交互缺失问题）：(a) Transformer Block 内微调：冻结已量化层权重，Adam 优化未量化层和 sign vectors（FP16）以最小化 Block 输出 MSE——减少激活误差累积；(b) 端到端微调：所有层量化完毕后，优化 layernorms、S_U、S_V、LM head 以最小化 CrossEntropy——捕获全局层间交互。2-bit 模型受益最大（2-7B: 8.22 → 6.19 Wikitext2 含 FT）。约 50 GPU-hours 量化 70B 模型，显著少于 QAT（LLM-QAT 需 960 GPU-hours 仅生成训练数据）。
  - 全栈执行例子（QuIP# 2-bit on Llama 2 70B, A100 + RTX 4090）：
    - **算法pipeline**：FP16 权重加载 → RedPajama 生成 Hessian H（6144 seqs × 4096 ctx）→ IP-RHT（Algorithm 3）：采样 S_U∼{±1}^m, S_V∼{±1}^n → Ŵ←Had(S_U·Had(S_V·W^T)^T), Ĥ←Had(S_V·Had(S_V·H)^T) → BlockLDLQ（g=8）：Ĥ=L^TDL → U=L^T-I → 逐 8 列块 Ŵ̂_k = Q_E8P(Ŵ_k+(Ŵ_{:k-1}-Ŵ̂_{:k-1})A_k) → RVQ 残差：2× E8P 2-bit → 层间微调（Algorithm 5）：per-block Adam MSE → 端到端 Adam CrossEntropy → 推理：x → Had(S_V⊙x) (FWHT O(n log n)) → E8P_decode_matvec kernel (MMA Tensor Core) → Had(S_U⊙y) → 下一层。Wikitext2 PPL 3.91（2-bit 70B）。
    - **系统框架**：论文未修改特定 Serving 框架（如 vLLM），但提供可直接加载的 PyTorch 量化模型（HuggingFace relaxml），推理过程集成 FWHT + E8P GEMV CUDA kernel。未来可集成到 vLLM 等框架的量化后端。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：自研 CUDA kernel `decode_matvec_e8p_kernel`：(a) 从压缩 uint2 码字中解码 4 个 E8P 码字 → 查 256 条目 codebook_abs → XOR 符号翻转 + ±1/4 偏移 → 生成 FP16 权重；(b) `mma.sync.aligned.m16n8k16` Tensor Core MMA 指令累加；(c) 1KiB codebook 放 L1 cache → 无 DRAM 往返 → 2-70B 达 56.84% peak mem BW（32.74 tok/s）。AQLM 1MiB codebook 导致 cache thrashing（8.27 tok/s, < FP16）。
    - **硬件架构**：NVIDIA A100（量化计算）、RTX 4090（推理性能测试），标准 Tensor Core + CUDA 路径。无定制硬件。
