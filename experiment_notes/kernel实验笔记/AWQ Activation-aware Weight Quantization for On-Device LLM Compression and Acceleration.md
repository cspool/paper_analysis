## AWQ Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现 TinyChat 推理系统，将 AWQ 的 4-bit weight-only 理论内存节省转化为实际端到端加速。核心包括：(1) **On-the-fly weight dequantization**：将 4-bit→FP16 反量化与矩阵乘法 kernel 融合，避免将反量化后的 FP16 权重写回 DRAM；(2) **SIMD-aware weight packing**：在 ARM NEON（128-bit SIMD）和 GPU（CUDA cores）上重新排列 4-bit 权重以高效利用 SIMD 指令解包；(3) **Kernel fusion**：融合 LayerNorm、QKV 投影、positional embedding 计算和 KV cache 更新，减少 GPU kernel launch 开销。
  - 实验比较：在 RTX 4090（桌面）、RTX 4070（笔记本）、Jetson Orin（移动 GPU）、A100（数据中心）上与 HuggingFace FP16 实现对比 token/s 加速比。在 Jetson Orin 上与 llama.cpp、exllama、AutoGPTQ 等第三方推理系统对比 Llama 模型的 token/s。在 Raspberry Pi 4B 上展示 7B 模型部署能力。

- 后端平台是什么，配置是什么。
  - GPU 后端：CUDA/PTX（NVIDIA RTX 4090, RTX 4070, Jetson Orin, A100）。实现时注重跨 GPU 架构的通用性，使用原生 PyTorch API 编写 forward pass。
  - CPU 后端：ARM NEON（128-bit SIMD，Raspberry Pi 4B）、AVX（x86 CPU）。CPU 端将整个计算图下沉到 C++ 以最小化框架开销。
  - 推理框架：TinyChat（自研），PyTorch 前端 + 设备特定指令集后端（CUDA/PTX、NEON、AVX）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估方法：batch_size=1，固定 prompt 长度 4 tokens，生成 200 tokens，取中位延迟（tokens/s）。协议参照 exllama。
  - TinyChat 系统的 kernel 修改：
    1. **On-the-fly dequantization kernel**：线性层中，INT4 权重从 DRAM 读取后立即在寄存器中反量化为 FP16，融合进 GEMM/GEMV 主循环。避免写回 FP16 权重到 DRAM。对于矩阵-矩阵乘（prefill）和矩阵-向量乘（decode）分别实现融合 kernel。
    2. **SIMD-aware weight packing**（ARM NEON 128-bit 为例，Figure 4）：
       - 传统排布：`w_0, w_1, w_2, ..., w_31` 顺序存储，每 4-bit 权重解包需 3 条标量指令（shift + AND + FMA scaling），32 个权重共 96 条指令。
       - AWQ 排布：`w_0, w_16, w_1, w_17, ..., w_15, w_31`。一个 128-bit 寄存器可同时解包 32 个 4-bit 权重，仅需 3 条 SIMD 指令（AND、shift、mask）。
       - 通用规则：对于 2^n-bit SIMD 寄存器，相邻权重的索引差为 `1/8 × 2^n`。
       - GPU 端排布：每 8 个权重打包为 `w_{0,2,4,6,1,3,5,7}` 顺序（参照 Kim et al., 2022）。
    3. **Kernel fusion**：
       - LayerNorm kernel：融合乘法、除法、平方根等所有操作。
       - Attention kernel：QKV 投影融合为单个 kernel + on-the-fly positional embedding 计算 + KV cache 预分配及更新。
       - 对比 Falcon-7B 官方实现中 KV cache 处理有 bug，TinyChat 的 FP16 kernel fusion 即使不量化也带来 1.6× 加速。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/mit-han-lab/llm-awq（含 TinyChat 完整推理代码）
  - TinyChat 推理全过程（以 Llama-2-7B W4A16 在 RTX 4090 上生成一个 token 为例）：

    **阶段 1: 预处理与权重加载**
    - 输入：INT4 packed 权重（每 2 个 4-bit 权重占 1 byte）+ per-group 量化参数（Δ，FP16，每组 128 个权重存 1 个 Δ）+ per-channel scale s（FP16，C_in 维，用于 AWQ 的等效变换）。
    - 输入 prompt tokens 已处理完毕，KV cache 已填充。

    **阶段 2: 自回归生成循环（decode 阶段，memory-bound）**
    - Step 2a: **LayerNorm kernel**（fused）。输入：上一层的 output hidden state h ∈ R^4096。执行：μ = mean(h), σ = sqrt(var(h)+ε)，output = γ·(h-μ)/σ + β。所有运算在单个 CUDA kernel 中完成，无中间 DRAM 写入。
    - Step 2b: **QKV projection kernel**（fused）。输入：LayerNorm 输出 h。执行：W_Q、W_K、W_V 三个 INT4 packed 权重依次读取，on-the-fly dequantization 后执行 GEMV（h 为向量，W 为矩阵，batch_size=1 时 decode 为 matrix-vector 乘积），输出 Q、K、V 三个向量。三个投影在单个 kernel 内完成。
    - Step 2c: **Attention kernel**（fused）。输入：Q, K, V。执行：Q·K_cache^T 计算 attention scores → softmax → score·V_cache → 更新 K_cache 和 V_cache（直接写入预分配 cache 地址，无额外 kernel launch）。Positional embedding（RoPE）在计算 Q·K^T 前 on-the-fly 应用。
    - Step 2d: **Output projection kernel**。输入：attention output。执行：W_O（INT4 packed）dequantize → GEMV → 输出 h_attn。
    - Step 2e: **MLP kernel**。输入：h + h_attn（残差连接，在 LayerNorm 之后）。执行：W_gate（INT4）dequantize → GEMV → SiLU activation；W_up（INT4）dequantize → GEMV → element-wise multiply → W_down（INT4）dequantize → GEMV → 输出 h_mlp。
    - Step 2f: **残差相加** → 完成一层 Transformer Block → 输入到下一层。

    **阶段 3: 性能输出**
    - 每个 kernel 的耗时在 0.01ms 量级（RTX 4090），kernel launch overhead 与计算时间可比，因此 kernel fusion 效果显著。
    - FP16 baseline：52 tokens/s（HuggingFace）。TinyChat FP16（含 kernel fusion）：62 tokens/s。TinyChat W4A16（fusion + dequantization）：~194 tokens/s（≈3.1× over FP16 fusion baseline, 3.7× over HF FP16）。
    - On-the-fly dequantization 避免了将 4× 的 FP16 反量化权重写回 DRAM，使 decode 阶段的 arithmetic intensity 从 ≈1 提升到 ≈4 FLOPs/Byte，将 4090 上的峰值性能上限从 ~1 TFLOPS 提升至 ~4 TFLOPS。

    **阶段 4: 跨平台部署**
    - TinyChat 使用同一套 PyTorch forward pass 代码，通过不同的设备后端（CUDA/PTX for NVIDIA GPU, NEON for ARM CPU, AVX for x86 CPU）部署。VILA-7B 在 Jetson Orin 上从 FP16 的 11.5 tokens/s 加速至 W4A16 35.6 tokens/s（3.1×）。
