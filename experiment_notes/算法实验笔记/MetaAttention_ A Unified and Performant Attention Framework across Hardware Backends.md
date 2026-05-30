## MetaAttention: A Unified and Performant Attention Framework across Hardware Backends

- 属于算法pipeline的实现是什么？实验比较什么？
  实现将多样化的 attention 机制统一抽象为两个核心操作——**relevance scoring**（相关性评分）和 **aggregation**（聚合），并通过两种计算模式实例化：(i) **Parallel Pattern**——需要全局上下文信息，relevance scoring = `matmul(Q, K)`，aggregation = `matmul(scores, V)`，适合 Softmax Attention、RetNet Parallel、MLA、Sigmoid Attention 等；(ii) **Recurrent Pattern**——迭代遍历序列，将上下文压缩为固定大小的 hidden state h，relevance scoring = `matmul(Q, h)`，aggregation = `h = h + matmul(K[i]^T, V[i])` 或类似递推公式，适合 Mamba2 SSM、RetNet Recurrent、Gated Retention 等。在此基础上引入两类 **customizable functions**：Mod（元素级变换，如 Q/K/V scaling、score masking、sparse mask 应用）和 RowNorm（行级归一化，如 softmax、sigmoid、L2 norm、ReLU），以及 **RowNorm online interface**（online_prologue/online_forward/online_epilogue 三阶段）支持通用 online 行归一化——在 on-chip memory 内逐 block 完成行归一化，避免中间结果写回 global memory。

  实验比较：(1) 10 种 attention 变体在 NVIDIA H100 上 kernel 延迟 vs FlashAttention-2/3、FlashSigmoid、FlashMLA、Mamba2 chunk kernel、Flash-Linear-Attention (FLA) Triton library、FlexAttention、FlashInfer、PyTorch native；(2) Customized Parallel Attention（Sigmoid/ReLU/Retention Parallel）——无 handcrafted library 的变体，MetaAttention 平均 3.6× speedup（1.1×~10.4×）；(3) Recurrent Pattern Attention（Mamba2/RetNet Recurrent/YOCO/RFA-Big）——forward 1.66×/backward 1.78× vs FLA；(4) MLA vs FlashMLA——性能相当 + 4.6× vs Triton；(5) Sparse GQA vs SeerAttention——1.71× speedup；(6) 端到端推理（DeepSeek-V2-Lite/Diff-Transformer-3B/Mamba2-2.7B/YOCO-160M, 16K input）——1.4× speedup；(7) 端到端训练（Diff-Transformer-3B/YOCO-160M/ViT-S/16 ReLU Attn, 8K seqlen）——1.4× speedup；(8) AMD MI250 GPU 上跨平台验证——3.3× forward / 2.0× backward；(9) 编译时间（分钟级，比传统 DL compiler 如 Ansor 更快）；(10) 开发工作量（22-90 LoC vs 手写 library 400-3000 LoC）。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 SXM5 (CUDA 12.4, Triton 2.3.1, 132 SMs, 80GB HBM, 989 TFLOPS FP16)，AMD Instinct MI250 (ROCm 6.2.4, Triton 3.1.0)。Benchmark 配置：batch sizes=1/8，sequence lengths=2K/4K/8K，数据类型 FP16。

- 模型是什么。数据集和bench分别是什么。
  模型：LLAMA-3.1-8B、DeepSeek-V2-lite、Diff-Transformer 3B、RetNet-6.7B、Mamba2-2.7B、YOCO-13B、RFA-Big、DeepSeek-V3 (MLA)、ViT-s/16 with ReLU Attention。Attention 配置见表3（head=6-128, dimqk=64-576, dimv=64-512）。端到端推理使用 Transformers 框架替换 attention operator；端到端训练使用 TRL 框架。Microbenchmark 为合成 tensor 直接测量 attention kernel 延迟，不含端到端 benchmark 数据集。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  开源仓库：https://github.com/SJTU-IPADS/MetaAttention (MIT License)，Zenodo: https://doi.org/10.5281/zenodo.17701680。Docker 环境支持 CUDA (NVIDIA Hopper, Dockerfile.cu128) 和 ROCm (AMD MI200, Dockerfile.rocm)。Functional test: `python testing/test.py`，Performance test 复现 Figure 11 (H100) 和 Figure 14 (MI250X)。Quick-start: `python examples/retention_parallel.py`。

  **算法 pipeline 伪代码**（以 RetNet Parallel Pattern forward 为例）：

  ```
  # 用户定义 attention template（约 22 行 Python）
  pattern: Parallel
  inputs: {Query: [B,H,S,256], Key: [B,H,Skv,256], Value: [B,H,Skv,512]}
  customizable_function:
    def scores_Mod(scores):          # Mod: 元素级 mask
        return scores * mask
    def scores_RowNorm(scores):      # RowNorm: 行归一化
        t = scores.reduceAbsSum()
        t = max(t, 1)
        return scores / t
  ```

  **MetaAttention 内部执行展开**（Parallel Pattern 模板）：
  ```
  for each tile_block of Q, K, V on device:
    # Stage 1: Relevance Scoring (matmul, Tensor Core)
    scores_tile = matmul(Q_tile, K_tile^T)     # [Br,d] × [d,Bc] → [Br,Bc]
    
    # Stage 2: Customizable score modification (Mod, elementwise SIMT)
    scores_tile = scores_Mod(scores_tile)      # e.g., apply mask
    
    # Stage 3: Online row-wise normalization (RowNorm online)
    # 在线计算，避免 scores 写回 HBM:
    #   online_prologue: init running_state
    #   online_forward(scores_tile, prev_state):
    #     更新 running max/sum
    #     scores_tile = normalize(scores_tile, running_state)
    #   online_epilogue: 最终归一化
    weights_tile = scores_RowNorm_Online(scores_tile)
    
    # Stage 4: Aggregation (matmul, Tensor Core)
    output_tile = matmul(weights_tile, V_tile) # [Br,Bc] × [Bc,dv] → [Br,dv]
    
    # Stage 5: Output modification (Mod, elementwise SIMT)
    output_tile = output_Mod(output_tile)
  ```

  **Recurrent Pattern 变体**（Mamba2 SSM）：
  ```
  state h = zeros([B, H, d_state])           # 压缩 hidden state
  for i in 0..seq_len-1:
      output[i] = matmul(Q[i], h)             # Relevance Scoring
      h = h_mod(h + matmul(K[i]^T, V[i]))    # Aggregation + state transform
      # h_mod 是 customizable Mod function
  ```

  与 Baseline（手写 CUDA/Triton kernel）的关键差异：
  - Baseline：每种 attention 变体需手写完整 kernel（FlashMLA 1000+ 行 CUDA, Mamba2 3000 行 Triton），执行策略（fusion, parallelism, pipelining, memory placement）全部 hardcode，不兼容非标准 shape 或不同 hardware
  - MetaAttention：用户定义 template + customizable functions（22-90 LoC），框架通过 IntermediateTensor-based scheduling 自动推导最优 tiling、memory placement 和 pipeline 策略；统一处理 Parallel 和 Recurrent 两种模式；支持 NVIDIA (CUDA/TMA/Tensor Core via TileLang & CUTE) 和 AMD (ROCm/Matrix Core via TileLang) 双后端
