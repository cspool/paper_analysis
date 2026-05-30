## QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：QuaRot 实现了三类核心 CUDA kernel：(1) **4-bit 线性层 kernel**：输入 FP16 激活 → 可选在线 Hadamard 变换 → 量化 kernel 将激活转为 sub-byte INT4 → CUTLASS 4-bit GEMM kernel（TensorCore, INT32 accumulator）执行 INT4×INT4 GEMM → dequant 输出 FP16；(2) **量化 KV cache attention kernel（基于 FlashInfer）**：三阶段——Init（prefill 时初始化量化 KV cache，直接用 Flash Attention 计算 attention output）、Append（解码时对当前 K/V 做 asymmetric group-wise 量化 → pack sub-byte → 追加到 cache）、Decode（从 HBM 加载量化 KV cache → 反量化 → 与 FP16 query 做 online softmax attention → FP16 output）；(3) **在线 Walsh-Hadamard 变换 kernel**：对激活值执行 O(d log d) 快速 Walsh-Hadamard 变换，支持 FP16 和 FP32，在 down-projection 和 out-projection 前调用。
  - 实验比较：(a) 4-bit linear layer vs FP16 linear layer 延迟对比（不同矩阵规模 4096×4096 ~ 28672×8192，batch=1-32，Table 14）；(b) 有无在线 Hadamard 变换的开销（INT4 vs INT4+FP32 Had vs INT4+FP16 Had）；(c) 单 transformer block 的 prefill 加速比（TTFT, batch=1/4/16/32/64, seq=2048，Figure 4 左，Table 16）；(d) 解码阶段峰值内存节省（batch=1/16, seq=256-4096，Figure 4 右，Table 17）；(e) KV cache decode kernel 延迟 vs FP16（不同 head_num×head_dim, batch=1-32, Table 15）；(f) 2D linear layer 速度随 batch scaling 行为（Figure 7）。

- 后端平台是什么，配置是什么。
  - NVIDIA RTX 3090 GPU（Ampere 架构）。CUDA 12.1。PyTorch + Hugging Face Transformers。CUTLASS 库（github.com/NVIDIA/cutlass）提供 INT4 TensorCore GEMM template。FlashInfer 库（github.com/flashinfer-ai/flashinfer）提供量化 KV cache attention 的 append/decode 路径。

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研 benchmark 脚本：对单 transformer block 中每个线性层和 attention 层分别测量延迟（100 次运行取平均, CUDA events 计时）。修改 CUTLASS：适配 QuaRot 的 sub-byte packed INT4 数据布局（激活值 per-token 量化为 INT4 → 2×INT4 pack 为 1 byte，权重 per-column INT4 → pack 格式）。修改 FlashInfer：在 attention decode 路径中加入量化 KV cache 的加载和反量化逻辑，支持 asymmetric group-wise dequant（K_fp16 = (K_q - z_k) × s_k, group=128）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/spcl/QuaRot
  - Kernel 输入到性能输出全过程（以 4-bit 线性层 W_down 11008×4096 为例）：
    1. **输入**：FP16 激活 X ∈ R^{B×T×4096}（prefill 阶段）+ INT4 packed 权重 W_q（per-column symmetric quant, weight scales s_w ∈ R^{11008}）+ per-column weight scales
    2. **在线 Hadamard**：X_h = FastWalshHadamard(X, dim=-1)（O(B×T×4096×log₂(4096)) = O(B×T×4096×12) 操作, ~7% 额外开销, FP16 和 FP32 精度几乎等效）
    3. **激活量化**（per-token symmetric INT4）：s_x[t] = max(|X_h[t,:]|) × 0.9 / 7.0（clipping ratio=0.9）→ X_q[t,i] = round(clip(X_h[t,i] / s_x[t], -7, 7)) → 2×INT4 pack 为 1 byte
    4. **CUTLASS INT4 GEMM kernel launch**：
       - Grid/Block: 按 M (B×T) 和 N (11008) 维度 tile 分配
       - Shared memory: 加载 packed X_q tile + packed W_q tile
       - Dequant on-the-fly: ŵ = unpack_4bit(w̃) × s_w[col]（per-column scale）
       - TensorCore: INT4×INT4 → INT32 accumulate（m16n8k32 tile）
       - 输出: Y_int32 ∈ R^{B×T×11008}
    5. **Dequant 输出**：Y_fp16 = (float(Y_int32) ⊙ s_x[:, None] ⊙ s_w[None, :]) → cast to FP16
    6. **评估原理**：CUDA events (cudaEventRecord) 记录 kernel launch 到 completion 的 wall clock time。预热 10 次后测量 100 次取平均。对比 FP16 cuBLAS GEMM baseline 的 wall clock time。加速比 = FP16_time / INT4_time。LLAMA2-7B 4-bit linear layer 达到 3.2× speedup (W_down, batch=1)，LLAMA2-70B 达到 4.3×。
    7. **Attention Decode Kernel 流程**（32 heads × 128 dim, KV cache 2047 tokens）：
       - 输入: FP16 query q (B×32×128) + INT4 packed KV cache
       - 加载: 按 group=128 从 HBM 加载 INT4 cache → dequant K/V → FP16
       - Online softmax: qK^T/√128 → softmax → ×V（逐 tile 累加, 避免完整 attention matrix 物化）
       - 输出: attention output (B×32×128)
       - 性能: batch≥16 时 4-bit 比 FP16 快 1.72×；小 batch (≤8) 时 4-bit 因量化/反量化开销略慢于 FP16
