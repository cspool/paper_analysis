## SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：基于OpenAI Triton实现的自定义attention kernel，融合INT8量化与FlashAttention-2风格tiling。核心技术：(1) 利用NVIDIA Tensor Core的INT8 mma(u8.u8.s32)指令加速$QK^\top$ Matmul（相比FP16有2-4×吞吐提升）；(2) 利用FP16-with-FP16-accumulator mma(f16.f16.f16)指令加速$PV$ Matmul（RTX4090/3090上相比FP32 accumulator有2×加速）；(3) Kernel Fusion — 将ROPE（Rotary Position Embedding）与Q/K量化融合，在ROPE结果写入global memory前完成量化，消除量化的IO开销；(4) 将系数$1/\sqrt{d}$在芯片上乘入Q后再量化（on-chip fuse），避免在attention kernel内额外操作；(5) 四个kernel变体（SAGEAttn-T/B/vT/vB）实现不同speed-accuracy tradeoff：T=per-token INT8 QK + FP16 PV, B=per-block INT8 QK + FP16 PV, vT=per-token INT8 QK + INT8 PV, vB=per-block INT8 QK + INT8 PV。
  - 实验比较：kernel speed vs FlashAttention2、xformers、Torch Attention（TOPS和GFLOPS，head_dim=64/128，sequence length 512~32768，w/wo causal mask）；real model speedup（Llama2, CogvideoX, UltraPixel, Unidiffuser, TIMM on RTX4090/3090）；quantization overhead（smooth K <0.2%）；adaptive quantization benefit（+11.7% OPS）。

- 后端平台是什么，配置是什么。
  - GPU: NVIDIA RTX 4090（Tensor Core INT8 throughput 660 TOPS理论值）；NVIDIA RTX 3090。RTX 4090服务器：PCIE 5.0, 16-core Xeon 6430 CPU, 120GB DDR4 RAM。RTX 3090服务器：16-core Xeon 8358P CPU, 80GB DDR4 RAM。
  - 软件栈: OpenAI Triton（triton-nightly 20240816版）→ PTX → NVIDIA Tensor Core指令。torch 2.4.0+cu121, python 3.11, gcc/g++ 9, Ubuntu 22.04。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估框架：自建Triton kernel benchmark + end-to-end模型推理脚本（基于HuggingFace Diffusers/Transformers加载模型，替换attention实现）。
  - 修改内容：
    - 实现Triton kernel：`sageattention.py`中实现forward kernel，调用`tl.dot()`利用Tensor Core mma指令
    - Fused ROPE + quantization：在ROPE kernel的epilogue中插入`ψ_Q`和`ψ_K`量化操作（`(δ, x̂) = quantize_int8(x)`）
    - kernel配置：block sizes b_q=128, b_kv=64；Num Warps=4/8 (headdim=64/128)；Num Stages=3/4/5
    - Adaptive quantization selector：对每层计算SAGEAttn-vB的cosine similarity，若>99.8%则选vB
  - 评估原理：(1) Kernel micro-benchmark: 对不同sequence length和headdim测量单个attention kernel的TOPS（Tera Operations Per Second），warmup后取多次平均；(2) 真实模型speedup：将模型中所有attention调用替换为SageAttention，测量attention部分latency和end-to-end latency speedup。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/thu-ml/SageAttention
  - Kernel输入到性能输出全过程（以SAGEAttn-B, headdim=64, RTX4090为例）：
    1. **Kernel Launch准备**：CPU侧将FP16 Q, K, V tensor（each ∈ R^{N×64}）从PyTorch传入Triton kernel。Kernel配置：grid=(T_m=N/128,), block size由Num Warps=4和Num Stages=3决定。Q block size b_q=128, KV block size b_kv=64。
    2. **Kernel输入**：FP16 Q ∈ R^{N×64}, K ∈ R^{N×64}, V ∈ R^{N×64}（HBM）。ROPE已在上一kernel完成并fused量化写入HBM：Q̂_INT8, K̂_INT8连同per-block scales δ_Q, δ_K 已就绪。
    3. **Kernel内执行**（单次Triton kernel launch，outer loop parallel on SMs）：
       (a) 从HBM加载Q̂_i[128×64] INT8 tile + δ_Q[i]到SRAM；
       (b) Inner loop j=1..T_n: 加载K̂_j[64×64] INT8 + V_j[64×64] FP16 + δ_K[j]到SRAM；
       (c) INT8 Tensor Core MMA: S_temp = tl.dot(Q̂_i, K̂_j^T) → INT32 accumulator → S_i^j = S_temp.to(FP16) × δ_Q[i] × δ_K[j]（dequant via scale broadcast）；
       (d) Online Softmax (FP16): m_i^j = max(m_i^{j-1}, rowmax(S_i^j)), P̃_i^j = exp(S_i^j - m_i^j)；
       (e) FP16 Tensor Core MMA with FP16 accumulator: ΔO = tl.dot(P̃_i^j.to(FP16), V_j.to(FP16), accum=FP16) → O_i^j = diag(e^{m_i^{j-1}-m_i^j})O_i^{j-1} + ΔO；
       (f) 循环结束：O_i = diag(l_i^{T_n})^{-1}O_i^{T_n}；
    4. **Kernel输出**：FP16 O_i[128×64] → 写回HBM。
    5. **性能输出**：TOPS = (2×N×d + 2×N²×d) / latency_μs × 10^{-6}。实测340 TOPS (headdim=64, N=8192, non-causal)，达到RTX4090 INT8理论峰值660 TOPS的52%。FlashAttention2对比：165 TOPS（FP16理论峰值330 TOPS的50%）。SageAttention 2.1× faster than FlashAttention2。
