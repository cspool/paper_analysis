## BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

- 属于算法pipeline的实现是什么？实验比较什么？
  提出利用Tensor Cores加速低比特KV Cache解码的量化推理pipeline。核心算法设计：(1) 低比特KV Cache量化（4-bit/2-bit Key，8-bit Value），支持channel-wise和tensor-wise两种scaling粒度，适配多种量化算法（KIVI、Gear、KVQuant等）。量化参数（scale + zero-point）压缩为half2格式降低metadata访存。(2) Residual block-based KV cache分区：将KV cache按Tensor Cores对齐的residual block size Nr = Pn × Wn × R分割（Pn=warp tile elements, Wn=N维warp数, R=packing ratio ω/β），前Np entries做packed low-bit存储，后Nr entries保持FP16 residual cache。decode时新tokens先追加到residual cache→达Nr后由Residual Kernel批量量化写入。(3) Query transformation：将Q从[1, (gq, hkv)] reshape为[gq, hkv]，使GQA/MQA的grouped query heads在Tensor Cores上形成更大GEMM block以提高occupancy。(4) 4-bit量化仅0.2% LongBench accuracy degradation（LLaMA-3.1-8B-Instruct, seq_len=32K），2-bit量化2.7% degradation。实验比较：kernel-level speedup vs FP16 FlashDecoding-v2（Blackwell 8.6×, Hopper 8.0×, Ada 7.5×, Ampere 4.8×），end-to-end throughput vs Kivi (non-fused) 和 QServe (fused CUDA-only)，page-setting下over 2× higher throughput than QServe。

- 硬件平台是什么，配置是什么。
  NVIDIA Blackwell (RTX 5090, RTX PRO 6000)、Hopper (H100)、Ada (RTX 4090)、Ampere (A100 80GB)。多GPU: 8×A100 for LLaMA-3.1-70B。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-2-7B (MHA)、LLaMA-3.1-8B (GQA)、LLaMA-3.1-70B (GQA, 8×A100)、Qwen3-8B (GQA)、Qwen3-14B (GQA)。benchmark：LongBench（长上下文理解accuracy评估，seq_len=32K）。kernel benchmark：不同seq_len (1K-128K)、batch_size (1-128)、attention head配置 (h_q=32-128, h_k=8-32, d=128)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源: https://github.com/OpenBitSys/BitDecoding。算法pipeline：
  1. Prefill阶段（prompt length L）：Residual Kernel执行——QK^T和PV mma (FP16 Tensor Cores)→每个thread in-register计算scale/zero→INT4/INT2量化+pack到INT16→前L-(L mod Nr)个KV entries写入packed low-bit KV cache，剩余res_len entries保持FP16在residual cache。
  2. Decode阶段（逐token自回归生成）：
     a. 新生成K/V tokens（FP16）追加到FP16 residual cache
     b. Packing Kernel执行Attention：Q reshape [1, (gq, hkv)]→[gq, hkv] (Query Transformation)→加载low-bit packed K/V到register→lop3-based dequant (INT4→FP16)→QK^T mma (Tensor Cores)→cooperative softmax (cross-warp reduction via shared memory)→PV mma (Tensor Cores)→output
     c. 当residual cache累计达Nr tokens→Residual Kernel触发：将FP16 residual批量量化写入packed KV cache→清空residual cache
  3. Channel-wise quantization example (group_size=128, d=4096, β=4): K tensor (seq_len, d) → 沿seq_len维度分组，每组128 tokens计算channel-wise scale/zero → KEY quantized to INT4 → 4个INT4 pack到1个INT16。Tensor-wise: 沿hidden维度分组。
  4. Blackwell原生mxfp4：跳过dequant→Q (FP16) × K_packed (mxfp4) 直接用mxfp4 mma→softmax→P需dynamic re-quantize to mxfp4→P_packed (mxfp4) × V_packed (mxfp4) mma。