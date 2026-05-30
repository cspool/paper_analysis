## LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

- 属于Serving调度的实现是什么？实验比较什么？
  实现是LiquidServe——基于LiquidGEMM W4A8 GEMM kernel构建的端到端LLM serving系统。LiquidServe集成了：(1) FlashAttention-2用于runtime attention计算；(2) PagedAttention用于KV cache管理；(3) SmoothQuant per-token动态激活量化（FP16→INT8）；(4) KV cache per-channel静态INT8量化（scale factor离线计算）；(5) 权重离线两级量化（FP16→INT8→UINT4, group size=64）。实验比较baseline：QServe（W4A8 KV4, group size=128）、TensorRT-LLM（FP16/W4A16/W8A8/FP8）。评估指标：peak token generation throughput（input=1024, output=512, batch size 1-256）、固定batch size下的throughput、per-layer time breakdown（GEMM/Attention/Others）。消融实验：LiquidServe/wo（替换LiquidGEMM为QServe的W4A8 kernel）对比LiquidGEMM的系统级加速贡献。

- 硬件平台是什么，配置是什么。
  NVIDIA H800 GPU（80GB HBM），Intel Xeon Platinum 8457C CPU，2.9TB RAM。PyTorch 2.4.0，CUDA 12.4。

- 开源Serving框架是什么。修改了什么。
  未使用单一现有开源Serving框架作为主代码基。LiquidServe自建serving系统，集成多个开源组件：(1) FlashAttention-2 [6]——替换为FlashAttention-2而非FlashAttention-3（后者专为FP8优化，但LiquidServe使用INT8 activation）；(2) PagedAttention [12]——用于高效KV cache管理，支持内存分页；(3) SmoothQuant [29]——per-token动态激活量化，通过smooth scale除以激活后在线量化FP16→INT8。LiquidGEMM GEMM kernel使用CUTLASS和Cute编程原语构建，集成WGMMA指令、TMA异步数据搬运和barrier同步。修改/自建内容：(1) 自建LiquidGEMM kernel——fused dequantization+MMA mainloop，Dual-MMA packed layout，ImFP pipeline；(2) 离线量化pipeline——FP16→INT8→UINT4两级量化+per-token激活量化；(3) KV cache INT8量化——per-channel静态量化替代QServe的4-bit KV cache。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源情况：LiquidGEMM/LiquidServe未提供开源代码。论文说明LiquidGEMM已部署为ByteDance Seed生产LLM serving基础设施的primary GEMM kernel。

  作用：LiquidServe是一个W4A8量化的端到端LLM serving系统，实现最高达4.94x系统级加速（vs QServe）。核心优化在于LiquidGEMM kernel通过LiquidQuant硬件友好dequantization和ImFP pipeline解决W4A8 GEMM的dequantization瓶颈，使W4A8在实践中超越W8A8和FP8，而不仅仅是理论roofline分析上的优势。

  全过程（以LLaMA2-7B单层decoding为例）：
  ```
  Serving系统接收请求（input_len=1024, output_len=512, batch_size=128）
    → Prefill阶段：FlashAttention-2处理prompt tokens
      - KV cache: PagedAttention管理，per-channel INT8量化存储
    → Decoding阶段（逐token生成）：
      Layer处理循环（自回归）：
        Step 1 - Attention:
          FlashAttention-2: QKV projection (LiquidGEMM W4A8 GEMM + output projection)
          KV cache更新: PagedAttention append新token的KV
        Step 2 - FFN:
          gate_proj + up_proj: LiquidGEMM W4A8 GEMM（INT8 activation × UINT4 weight）
          SiLU activation: elementwise on FP16
          down_proj: LiquidGEMM W4A8 GEMM
        Step 3 - Activation Quantization:
          动态per-token: FP16 activation → smooth_scale除法 → clamp → INT8
          （fused到前一个kernel的epilogue中，overhead微小）
    → 输出token → 循环至output_len完成后返回

  LiquidGEMM kernel内部（每个GEMM调用）：
    Load WG: TMA从GMEM加载UINT4 weight tile到SMEM（Dual-MMA packed layout, LDG.128）
    → ImFP: Load WG将weight切分为fine-grained tasks写入SMEM
    → Compute WG_0: 从SMEM竞争获取task → LDS.128加载到RF → unpack 4-bit
      → IMAD + XOR dequantization (4 elements/2 instructions, CUDA Cores)
      → WGMMA.m64nNk32 MMA (Tensor Cores, INT8)
    → Compute WG_1: 同时处理另一task（dequantization与MMA跨WG自然重叠）
    → Epilogue: 第一级dequantization（INT8→FP16, per-channel scale）→ 写回GMEM
  ```
