## DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出DSV框架，通过利用Video DiT训练过程中注意力的动态稀疏性来加速训练。核心算法pipeline实现包括：
  (i) **Sparsity Predictor**：为每个self-attention模块引入两个低秩可训练矩阵W_Q^lr和W_K^lr，将输入投影到远小于原始Q/K的inner dimension d_lr（d_lr ≪ d_k），用Q_lr·K_lr^T近似原始attention score分布QK^T，用于预识别critical KV pairs。
  (ii) **Two-Stage Training Algorithm**：Stage 1全量训练+高频sparsity profiling+预测器训练（损失函数：L_approx = 0.95·CosLoss(QK_lr, QK_main) + 0.05·NormLoss(QK_lr, QK_main)），当平均近似损失低于阈值0.01后进入Stage 2；Stage 2激活稀疏计算，低频profiling+连续预测器微调+选择性稀疏OP调度。
  (iii) **Profiler**：周期性测量各attention head和block的稀疏度（定义：critical KV pairs累计贡献90% total attention scores），使用动量更新S^i = α·S^{i-1} + (1-α)·S^i_profiled（α=0.9）。
  (iv) **OP Dispatcher**：基于block级别稀疏度、offline profile的性能特征和当前memory utilization决定哪些block使用稀疏计算。
  实验比较：Baseline为Vanilla 3D Full Attention（基于FlashAttention-2实现）和3D Window-based Sparse Attention（WA-M窗口1/3、WA-L窗口2/3），比较训练收敛、验证loss、FVD、VBench Quality Score和Semantic Score。

- 硬件平台是什么，配置是什么。
  最多64张NVIDIA H100 GPU，节点内通过900 GB/s NVLink双向互联，节点间通过InfiniBand with RoCE（200 Gbps per cross-node GPU pair）。计算精度BF16，梯度通信和分布式optimizer更新使用FP32。默认启用gradient checkpointing。

- 模型是什么。数据集和bench分别是什么。
  模型：0.8B（28层, 12头, head size 96, GeGLU）、2.7B（32层, 16头, head size 128, GeGLU）、30B（42层, 24头, head size 256, GeLU-approximate），架构类似Meta MovieGen，包含3D full self-attention和cross-attention模块。VAE使用stability-ai（8×8空间压缩）。文本编码器：UCF-101/WebVid-10M使用CLIP，VideoGen/OpenVid使用T5-xxl。优化器Adam（lr=1e-4），训练范式flow matching。
  数据集：UCF-101（latent 16×16×16, 4K tokens）、WebVid-10M（latent 16×16×16, 4K tokens）、VideoGen（latent 32×32×32, 32K tokens）、OpenVid（latent 40×56×56, 125K tokens）。
  评测指标：FVD（Fréchet Video Distance）、VBench Quality Score和Semantic Score。

- 开源情况。论文未提供开源代码仓库链接。实现基于PyTorch FSDP框架，扩展了tensor和context parallelism。核心kernel用Triton（稀疏attention）和CUDA（critical KV estimation）编写。截至分析时未在公开平台找到代码发布。

  算法pipeline说明书（伪代码级）：
  ```
  # Stage 1: Full Training with Predictor Training
  for each training iteration:
      for each DiT block:
          Q, K, V = W_Q(H), W_K(H), W_V(H)        # 原始投影
          Q_lr, K_lr = W_Q^lr(H), W_K^lr(H)       # 低秩投影 (d_lr << d_k)
          
          # Full attention computation (FlashAttention)
          O = FlashAttention(Q, K, V)
          
          # Predictor training (detached from main graph)
          L_approx = 0.95 * CosLoss(Q_lr @ K_lr^T, Q @ K^T) \
                   + 0.05 * NormLoss(Q_lr @ K_lr^T, Q @ K^T)
          update(W_Q^lr, W_K^lr) using L_approx
          
          # High-frequency sparsity profiling (sampled)
          if iteration % profile_interval == 0:
              S[i] = 0.9 * S[i-1] + 0.1 * S_profiled[i]
  
  # Stage transition: when avg(L_approx) < 0.01 across all blocks
  
  # Stage 2: Sparse Training
  for each training iteration:
      for each DiT block:
          if block.sparsity > sparsity_threshold:  # OP Dispatcher decision
              # Sparse attention path
              Q_lr, K_lr = W_Q^lr(H), W_K^lr(H)
              crit_indices = FusedTopK(Q_lr @ K_lr^T, k=ceil((1-sparsity)*S))
              # crit_indices shape: [H, num_queries, K_per_query]
              O = SparseFlashAttention(Q, K, V, crit_indices)
          else:
              O = FlashAttention(Q, K, V)
          
          # Low-frequency predictor fine-tuning
          if iteration % fine_tune_interval == 0:
              fine_tune(W_Q^lr, W_K^lr)
  ```

  张量计算说明：假设输入H ∈ R^{S×d}（S为token数，d为hidden dim），标准attention需要计算QK^T ∈ R^{S×S}（O(S²)复杂度）。DSV用Q_lr·K_lr^T ∈ R^{S×S}近似（其中Q_lr, K_lr ∈ R^{S×d_lr}且d_lr ≪ d_k），将低秩乘积通过fused kernel进行top-K选择，直接得到critical KV indices（维度[H, S, K_per_query]），随后仅对这些选中的KV对进行完整的softmax attention计算，将attention复杂度从O(S²d)降至O(S·K·d)（K≈(1-sparsity)·S ≪ S）。

## A Survey of Resource-efficient LLM and Multimodal Foundation Models

- 属于算法pipeline的实现是什么？实验比较什么？
  本文为综述论文，无原创实验。综述范围覆盖资源高效算法，包括：
  (i) **高效注意力机制**（§3.1）——稀疏注意力（Longformer、BIGBIRD）、近似注意力（Linformer、Reformer、Performer）、无注意力架构（Mamba/SSM、RWKV、RetNet、Hyena），复杂度从O(T²d)降至O(Td)或O(T log T d)；
  (ii) **动态神经网络**（§3.2）——Mixture-of-Experts（Switch Transformer、GLaM、V-MoE）、Early Exiting（DeeBERT、PABEE、FREE）；
  (iii) **预训练算法**（§4.1）——训练数据缩减（去重、patch removal如MAE）、NAS（Zero-shot NAS、PASHA）、渐进式学习（StackingBERT、LiGO）、混合精度训练（Mesa、GACT）；
  (iv) **微调算法**（§4.2）——Additive Tuning（Adapter、Prompt Tuning、Prefix Tuning）、Selective Tuning（SAM、SmartFRZ）、Re-parameter Tuning（LoRA及其变体QLoRA、DoRA、PiSSA、LoRA+）；
  (v) **推理算法**（§4.3）——Speculative Decoding（2-3×加速）、KV Cache优化（H2O、FastGen、vLLM PagedAttention）、Prompt压缩（LLMLingua 20×压缩）、Long Context（StreamingLLM、LongNet）；
  (vi) **模型压缩**（§4.4）——剪枝（SparseGPT、Wanda、LLM-Pruner）、知识蒸馏（MiniLLM、GKD、Distilling Step-by-Step）、量化（GPTQ 3-4bit、AWQ、SmoothQuant、QuaRot）、低秩分解（TensorGPT 38.4×压缩）。
  论文使用flops-profiler工具（https://pypi.org/project/flops-profiler/）对GPT-2及Stable Diffusion 2.1进行FLOPs和存储开销分析（§2.1.3、§2.3.3）。

- 硬件平台是什么，配置是什么。
  论文为综述，未进行统一硬件实验。综合分析引用以下平台：NVIDIA A100/H100 GPU、TPU v4、消费级GPU、手机端（iPhone 12 CoreML、安卓NPU）、Raspberry Pi 5等。

- 模型是什么。数据集和bench分别是什么。
  综述覆盖模型：LLM（GPT-1/2/3/4、LLaMA-1/2、BERT、T5、PaLM）、ViT（ViT、DeiT、MAE、Swin Transformer）、扩散模型（Stable Diffusion 1/2）、多模态（CLIP、Flamingo、LLaVA、SAM）。
  Benchmark覆盖：GLUE、SuperGLUE、SQuAD、MMLU、HumanEval、ImageNet-1K/21K、COCO、ADE20K等。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  综述材料全部开源：https://github.com/UbiquitousLearning/Efficient_Foundation_Model_Survey。该仓库包含LaTeX源码及参考文献BibTeX，但不包含可执行代码或benchmark脚本。以下以量化为例说明算法pipeline：

  量化（Quantization）算法pipeline——将FP32权重/激活转为低精度整数（§4.4.3）：
  ```
  // Weight-Only PTQ (以GPTQ为例)
  // 逐层量化，使用逆Hessian信息更新未量化权重
  for layer l in model.layers:
      W = layer.weight  // FP32, shape [d_out, d_in]
      H = inverse_hessian(W, calibration_data)  // [d_in, d_in]
      for i in range(d_in):
          // 量化第i列，使用H[i:,i:]补偿误差
          w_q[:,i] = round(W[:,i] / scale[i])  // INT4
          // 更新未量化权重以补偿量化误差
          W[:,i+1:] -= (w_q[:,i] - W[:,i]) * H[i,i+1:] / H[i,i]
  // 推理时dequantize: W_fp16 ≈ dequant(w_q) * scale
  ```

  以LoRA（§4.2.3）为例的低秩适应pipeline：
  ```
  // W_0 ∈ R^{d×k} 为预训练权重（冻结）
  // A ∈ R^{d×r}, B ∈ R^{r×k} 为可训练低秩矩阵，r << min(d,k)
  h = W_0 @ x + α/r * (B @ A) @ x
  // 训练时仅更新A, B；推理时W = W_0 + α/r * B @ A可fuse回原权重
  ```

## FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism

- 属于算法pipeline的实现是什么？实验比较什么？
  实现Tensorized Online Softmax Algorithm（Algorithm 1）：在fused attention的block-wise在线softmax计算中，引入X-row tile surrogate maximum m̂[i]替代逐行maximum m[i]，使m̂[i]在X行范围内保持uniform（对应HMMA.1688的X=16或HGMMA.64x8x8的X=64），从而满足repurposed tensor MMA scaling instruction对uniform scaling factor的约束。该surrogate保证：(1) no overflow（m̂[i] ≥ m[i]）；(2) 极低的all-underflow概率（对Gaussian等典型attention score分布）。若极端分布触发all-underflow，可选择性回退至vectorized rescaling。
  实验比较：(i) Attention throughput vs FlashAttention-2/3、FlashInfer、Triton；(ii) Ablation——FA2 vs FA2+Max16 vs AllTensor vs ILP FA-T至多18.4% speedup；(iii) Synthetic accuracy——RMSE vs FP64 reference under outlier variance τ²；(iv) Generative benchmark——HumanEval Pass@10和MMLU score功能正确性无退化。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100 80GB SXM4（Ampere SM80）：FP16-FP32 GEMM 312 TFLOPS
  - NVIDIA Jetson AGX Orin 64GB（Ampere SM87, edge-class）
  - NVIDIA H100 80GB PCIe（Hopper SM90A）：FP8-FP32 GEMM ~989 TFLOPS，支持异步WGMMA

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama3.1 8B、Ministral 8B、Qwen3 8B、Llama2 13B、Mistral NeMo 12B、Qwen3 14B
  - Attention变体：MHA、GQA（preliminary results show similar speedups）
  - Benchmark：HumanEval（Pass@10 with pass@k estimation）、MMLU（accuracy score）
  - Synthetic test：Q/K/V sampled from N(0,1) + N(0,τ²)·Bernoulli(1/1000)，h=128, s=4096

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源artifact在Zenodo：https://doi.org/10.5281/zenodo.17673796，基于FlashAttention-2/3代码库构建。

  算法pipeline（Tensorized Online Softmax，Algorithm 1，以warpgroup单次iteration为例，输入S∈R^{n×s}, O∈R^{n×d}, m_old∈R^n, l∈R^n, surrogate tile size X）：
  1. Compute X-row tile maxima: m̂ ← tilemax(S, X) → m̂ ∈ R^{⌈n/X⌉}
     - 用warp all-reduce REDUX指令高效计算每X行最大值
  2. Get old surrogate maximums: m̂_old ← m_old[X·i] for i∈[0, ⌈n/X⌉)
  3. **Tensorized rescale O**：O ← exp(m̂_old - m̂)·O（repurposed tensor MMA scaling，利用m̂在当前X行内的uniformity满足tensor MMA scaling constraint）
  4. Assign surrogate maximums: m[i] ← m̂[⌊i/X⌋] for i∈[0, n)
  5. **Tensorized rescale S**：Z ← log₂(e)·S - (log₂(e)·m)（repurposed tensor MMA FMA，log₂(e)为常量scaling factor）
  6. Vector exponentiation: P̃ ← exp₂(Z)（MUFU.EX2或等价指令，保持vectorized因exp无法tensorize）
  7. **Tensorized row-sum**：l ← exp(m_old - m)·l + rowsum(P̃)（repurposed tensor MMA row-sum reduction，消除explicit thread sync和all-reduce）
  8. Return P̃, O, m, l

  与Baseline safe online softmax的关键差异：
  - Baseline：逐行计算maximum m[i]，导致O rescaling的scaling factor exp(m_old[i]-m[i])逐行不同→无法使用tensor MMA uniform scaling
  - FlashAttention-T：X-row surrogate m̂[k] = max of X consecutive rows → scaling factor exp(m̂_old[k]-m̂[k])在X行内uniform → 满足tensor MMA scaling constraint
  - Numerical safety：m̂[k] ≥ m[i]，no overflow guarantee；all-underflow probability在典型分布下asymptotically small；极端情况可回退vectorized rescaling

- 属于算法pipeline的实现是什么？实验比较什么？
  实现BLASST算法（Algorithm 1）：在FlashAttention的block-wise online softmax过程中，维护running maximum $m_i^{(j)}$。当block的local maximum $\tilde{m}_i^{(j)}$ 满足 $\tilde{m}_i^{(j)} - m_i^{(j)} < \ln(\lambda)$ 时，跳过该block的：(1) softmax指数计算（CUDA core MUFU.EX2指令），(2) attention-value矩阵乘法（Tensor core MMA），(3) Value block的HBM加载。
  实验比较：(i) prefill阶段 vs dense attention（FlashAttention-3 BF16）、MInference、FlexPrefill、XAttention；(ii) decode阶段 vs dense attention、Quest、RocketKV；(iii) prefill+decode联合 vs dense baseline；(iv) sparsity-aware training vs training-free sparsity的accuracy-sparsity trade-off；(v) 与其他sparsity方法的可组合性（XAttention+BLASST、BLASST+RocketKV）。

- 硬件平台是什么，配置是什么。
  - NVIDIA Blackwell B200 GPU：prefill batch=1, 64K seq len; decode batch=148, 32K seq len
  - NVIDIA Hopper H200 GPU：prefill batch=1, 64K seq len; decode batch=128, 16K seq len
  - 端到端serving评估：B200和H200，TensorRT-LLM in-flight batching，concurrency=64，输入平均10K tokens，输出平均6 tokens

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama-3.1-8B-Instruct、Qwen3-8B-Instruct、Qwen3-30B-A3B-Instruct、Llama-3.1-70B-Instruct、DeepSeek-R1（MLA attention验证）、Llama-3.1-8B-Instruct蒸馏自DeepSeek-R1（长生成推理）
  - Attention变体：MHA、GQA、MQA、MLA（DeepSeek-R1）
  - 长上下文benchmark：RULER（4K-128K tokens，合成检索与推理，含NIAH_MULTI/VT/FWE等子集）、LongBench v1/v2（真实世界QA、摘要、代码补全）
  - 推理benchmark：MATH500、AIME 2024、GPQA（研究生级科学推理）、LiveCodeBench（代码生成）
  - 超长序列：RepoQA（16K/200K代码仓库理解）
  - 评估框架：NVIDIA NeMo-Skills（https://github.com/NVIDIA-NeMo/Skills）用于推理任务标准化评估

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源集成到TensorRT-LLM（https://github.com/NVIDIA/TensorRT-LLM）和FlashInfer。Artifact repository: https://github.com/cameronshinn/blasst-ae-mlsys26.git（MLSys 2026 artifact evaluation，Apache 2.0 许可证）。

  算法pipeline（以prefill一次forward pass为例，输入Q/K/V ∈ R^{L×d}，block size B_c）：
  1. 将Q分为T_r个block {Q_i | i=1..T_r}，将K/V分为T_c个block {K_j, V_j | j=1..T_c}，每个block大小为B_c × d
  2. 对每个query block i，初始化 running max m_i^{(0)} = -∞, output O_i^{(0)} = 0, sum l_i^{(0)} = 0
  3. 对每个KV block j（外循环，逐个处理）：
     a. 计算 attention scores: S_{ij} = Q_i × K_j^T ∈ R^{B_r×B_c}（BMM1 / QK^T，tensor core）
     b. 计算block local max: m̃_i^{(j)} = rowmax(S_{ij})（CUDA core reduction，每行取最大值）
     c. 更新running max: m_i^{(j)} = max(m_i^{(j-1)}, m̃_i^{(j)})
     d. 【跳过检查】if m̃_i^{(j)} - m_i^{(j)} < ln(λ):
          - 跳过 softmax: 不计算 P̃_{ij} = exp(S_{ij} - m_i^{(j)})（省CUDA core MUFU.EX2）
          - 跳过 attention-value乘法: 不计算 P̃_{ij} × V_j（省tensor core MMA）
          - decode kernel还跳过: 不加载V_j from HBM（省memory bandwidth）
     e. 若未跳过，计算softmax权重: P̃_{ij} = exp(S_{ij} - m_i^{(j)})
     f. 若未跳过，更新running sum: l_i^{(j)} = exp(m_i^{(j-1)}-m_i^{(j)})·l_i^{(j-1)} + rowsum(P̃_{ij})
     g. 若未跳过，累加输出: O_i^{(j)} = exp(m_i^{(j-1)}-m_i^{(j)})·O_i^{(j-1)} + P̃_{ij} × V_j（BMM2）
  4. 最终归一化: O_i = O_i^{(T_c)} / l_i^{(T_c)}

  校准pipeline（Algorithm 2）：对校准集D中每个样本(x_i, L_i)和每个候选阈值λ_j，测量达到的sparsity s_{ij}，记录数据点(λ_j·L_i, s_{ij})。拟合指数模型 λ·L = α·exp(β·s)。推理时用目标sparsity S计算 λ = α·exp(β·S)/L。校准仅需一次forward pass，因为sparsity对所有λ可从同一次attention scores计算。发现最优阈值与上下文长度成反比 λ = a/L。

  Sparsity-aware training：fine-tuning时forward pass应用BLASST跳过block，backward pass中跳过的block自然不接收梯度（forward未计算），迫使模型将重要信息集中在高attention score block中。采用ProLong的curriculum training方法。

## FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness

- 属于算法pipeline的实现是什么？实验比较什么？
  实现FLASHATTENTION算法（Algorithm 1）和block-sparse FLASHATTENTION（Algorithm 5，Appendix B）：(i) **Tiling**——将Q/K/V矩阵分块，在on-chip SRAM中逐block计算softmax，通过维护running maximum $m_i$和running sum $\ell_i$实现online softmax的增量计算（algebraic aggregation），避免在HBM中materialize $N \times N$的attention矩阵；(ii) **Recomputation**——前向pass存储输出O和softmax归一化统计量$(m, \ell)$，反向pass在SRAM中重计算attention矩阵S和P，避免从HBM读取$O(N^2)$中间值；(iii) **Kernel fusion**——所有attention操作（矩阵乘、softmax、masking、dropout、矩阵乘）融合为单个CUDA kernel。Block-sparse FLASHATTENTION在此基础上跳过预定义稀疏mask中零值block的计算。
  实验比较：(i) 训练速度——BERT-large（MLPerf 1.1 speed record vs FLASHATTENTION）、GPT-2 small/medium（HuggingFace vs Megatron-LM vs FLASHATTENTION）、Long-Range Arena（Transformer vs FLASHATTENTION vs block-sparse FLASHATTENTION vs Linformer/Linear Attention/Performer/Local Attention/Reformer/Smyrf）；(ii) 模型质量——GPT-2 with increased context length（1K/2K/4K）的perplexity、长文档分类（MIMIC-III, ECtHR）在不同sequence length（512-16384）下的micro $F_1$、Path-X和Path-256 benchmark的accuracy；(iii) Attention benchmark——runtime和memory随sequence length（128-64K）变化，对比exact attention（PyTorch）、approximate attention（Linformer/Linear Attention/Performer/Reformer/Smyrf/Local Attention）和sparse attention。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100 GPU (Ampere架构)：40GB或80GB HBM，带宽1.5-2.0TB/s，192KB on-chip SRAM per SM（共108 SMs），SRAM带宽约19TB/s
  - 训练实验：8×A100 GPU（BERT-large，GPT-2 small/medium）
  - Benchmark实验：单卡A100 40GB，dropout + padding mask，head dim 64，16 heads，batch size 64
  - 软件环境：CUDA（自编CUDA kernel，基于Apex FMHA代码https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha作为起点），PyTorch

- 模型是什么。数据集和bench分别是什么。
  - 模型：BERT-large（seq length 512）、GPT-2 small（124M, seq length 1K/2K/4K）、GPT-2 medium（350M, seq length 1K）、Transformer（LRA benchmark, seq length 1K-4K）、RoBERTa（pretrained，长文档分类，seq length 512-16384）
  - 训练数据：
    - Wikipedia（BERT-large pretraining，MLPerf 1.1 benchmark setting）
    - OpenWebText [34]（GPT-2 small/medium pretraining）
    - LLaVA-CC3M-Pretrain + LLaVA-NEXT Instruction Tuning data（论文未明确说明，Dimple相关）
  - Benchmark：
    - MLPerf 1.1 Training（BERT-large，目标accuracy 72.0% masked language modeling）
    - Long-Range Arena (LRA) [83]：ListOps、Text、Retrieval、Image、Pathfinder（seq length 1K-4K）
    - MIMIC-III [49]：重症监护出院摘要，多标签分类（micro $F_1$），平均2395 tokens，最长14562 tokens
    - ECtHR [6,7]：欧洲人权法院法律案例分类（micro $F_1$），平均2197 tokens，最长49392 tokens
    - Path-X（seq length 16K）和Path-256（seq length 64K）：128×128/256×256黑白图像中两点间路径连接分类

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/HazyResearch/flash-attention（Stanford HazyResearch，BSD许可证）。代码库包含CUDA kernel实现、PyTorch接口（`flash_attn_func`）、block-sparse variant、以及训练BERT/GPT-2的示例脚本。论文声明实现以NVIDIA Apex FMHA代码（https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha）为起点。

  算法pipeline（FLASHATTENTION forward pass，输入Q/K/V ∈ R^{N×d} in HBM，SRAM size M）：
  1. 计算block sizes: $B_c = \lceil\frac{M}{4d}\rceil$（K/V列block大小），$B_r = \min(\lceil\frac{M}{4d}\rceil, d)$（Q行block大小）
  2. 将Q分为$T_r = \lceil N/B_r\rceil$个blocks $\{Q_1,...,Q_{T_r}\}$，每block $B_r \times d$
  3. 将K/V分为$T_c = \lceil N/B_c\rceil$个blocks $\{K_1,...,K_{T_c}\},\{V_1,...,V_{T_c}\}$，每block $B_c \times d$
  4. 初始化output $O = (0)_{N \times d}$，running stats $\ell = (0)_N$，$m = (-\infty)_N$ in HBM
  5. for j = 1 to $T_c$:  // 外循环：遍历KV blocks
       Load $K_j, V_j$ from HBM → on-chip SRAM
       for i = 1 to $T_r$:  // 内循环：遍历Q blocks
         Load $Q_i$ from HBM → SRAM
         On chip: $S_{ij} = Q_i K_j^T \in \mathbb{R}^{B_r \times B_c}$  // Tensor core GEMM
         On chip: $\tilde{m}_{ij} = \text{rowmax}(S_{ij})$，$\tilde{P}_{ij} = \exp(S_{ij} - \tilde{m}_{ij})$，$\tilde{\ell}_{ij} = \text{rowsum}(\tilde{P}_{ij})$
         On chip: $m_i^{\text{new}} = \max(m_i, \tilde{m}_{ij})$
         On chip: $\ell_i^{\text{new}} = e^{m_i - m_i^{\text{new}}}\ell_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}}\tilde{\ell}_{ij}$
         Write $O_i \leftarrow \text{diag}(\ell_i^{\text{new}})^{-1}(\text{diag}(\ell_i)e^{m_i - m_i^{\text{new}}}O_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}}\tilde{P}_{ij}V_j)$ to HBM
         Write $\ell_i \leftarrow \ell_i^{\text{new}}, m_i \leftarrow m_i^{\text{new}}$ to HBM
  6. Return O（所有blocks处理完后）

  IO复杂度：FLASHATTENTION需要$\Theta(N^2d^2M^{-1})$次HBM访问，标准attention需要$\Theta(Nd + N^2)$次。对于典型值d=64-128, M≈100KB，$d^2 \ll M$，FLASHATTENTION的HBM访问减少数倍（up to 9× fewer）。

  反向pass：存储O和softmax统计量$(m, \ell)$（仅$O(N)$额外内存），在SRAM中重计算$S_{ij}$和$P_{ij}$来求$\partial Q, \partial K, \partial V$的梯度。比从HBM读取$N\times N$ attention矩阵更快。

  Block-sparse FLASHATTENTION：给定block sparsity mask $M \in \{0,1\}^{N/B_r \times N/B_c}$，仅在$M_{ij}=1$时执行内循环的$S_{ij}$计算和softmax/PV操作。IO复杂度$\Theta(Nd + N^2d^2M^{-1}s)$，其中$s$为非零block比例。

## Flash Multi-Head Feed-Forward Network

- 属于算法pipeline的实现是什么？实验比较什么？
  实现FlashMHF架构：将标准SwiGLU FFN替换为Multi-Head FFN + Parallel FFN Sub-Networks设计。核心：（1）多head分解——将FFN输入切分为H个head，每个head独立执行key-value style的FFN计算（公式8-9）；（2）并行子网络——每head内包含E个并行sub-network，通过sigmoid gating学习加权聚合（公式11-13），维持d_e ≈ 8/3·d_h的平衡ratio解决scaling imbalance；（3）最终concat所有head输出并做Wo投影（公式14）。整体类似dense MoE但不做sparse top-k selection。
  实验比较：（i）128M/370M/1.3B规模上FlashMHF vs SwiGLU baseline的validation loss（PG19）和perplexity（Table 1, Figure 4-7）；（ii）FlashMHF vs MH-FFN naive多head的scaling对比（128M vs 370M性能分化）；（iii）FlashMHF vs PKV baseline（Parametric KV——用multi-head attention替换FFN验证element-wise activation必要性）；（iv）FlashMHF vs Dense-MoE baseline（H=1验证多head必要性）；（v）head dimension ablations（d_h=64/128/256）at 370M和1.3B（Table 1, Figure 5-6）；（vi）下游任务：HellaSwag/SIQA/PIQA/OBQA/WinoGrande/RACE（Table 2）；（vii）memory和latency benchmark vs SwiGLU FFN和MH-FFN（Figure 8, Table 5）。

- 硬件平台是什么，配置是什么。
  - NVIDIA H100 GPU (Hopper架构)，用于memory/latency benchmark和efficiency实验
  - 训练：NVIDIA H100 GPU集群（PyTorch + bfloat16），128M/370M: 245K steps, 1.3B: 409K steps
  - 推理benchmark：单H100，batch size=8, sequence length从192到16128（Table 5）
  - 软件环境：PyTorch, Triton（consumer GPU kernel实现）, ThunderKittens（Hopper kernel实现）, cuBLAS

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama-like architecture（RoPE + multi-head self-attention + SwiGLU FFN / FlashMHF），128M/370M/1.3B参数规模。使用GPT-NeoX tokenizer，vocab size=50,432。config详见Table 4。
  - 训练数据：THE PILE，128M/370M训练60B tokens，1.3B训练100B tokens。context length=4096, batch size=64。
  - 验证集：PG19 validation split（evaluation loss）
  - 下游benchmark：HellaSwag（commonsense reasoning）、SIQA（Social IQA）、PIQA（Physical IQA）、OpenBookQA、WinoGrande、RACE（reading comprehension）
  - Baseline模型：Llama-like SwiGLU FFN、PKV（Parametric KV attention替换FFN）、Dense-MoE（H=1）、Naïve MH-FFN

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文声明代码将公开于 https://anonymous.4open.science/r/FlashMHF-9395（当前为匿名审阅提交状态，403 Forbidden）。出版后将在该URL开源I/O-aware kernel实现、模型配置和训练脚本。

  算法pipeline——FlashMHF forward pass（以H=8 heads, E=7 sub-networks, d_h=128, d_e≈342为例）：
  
  ```
  输入: X ∈ R^{L × d_model}（如1024×1024）
  参数: W_in ∈ R^{d_model × d_model}, W_out ∈ R^{d_model × d_model}
        For h=1..H, e=1..E: K_e^h, U_e^h, V_e^h ∈ R^{d_e × d_h}
        For h=1..H: W^h ∈ R^{d_h × E}
  
  Step 1 - Head-wise split & projection:
    Q = split_H(X · W_in) ∈ R^{L × H × d_h}
    # d_model=1024, H=8, d_h=128
    # Q[:,h,:] 是第h个head的query (L×128)
  
  Step 2 - Per-head gating:
    For h = 1..H:
      P^h = Q[:,h,:] · W^h ∈ R^{L × E}          # 每token E个sub-network的logits
      R^h[:,e] = σ(P^h[:,e]) / (Σ_{e'} σ(P^h[:,e']) + ε)  # sigmoid归一化gating weights
  
  Step 3 - Per-head sub-network aggregation:
    For h = 1..H:
      S_h = 0 ∈ R^{L × d_h}
      For e = 1..E:                                # E=7个并行sub-network
        # 每个sub-network内做SwiGLU-style key-value计算:
        # FFÑ(Q_h; K_e^h, U_e^h, V_e^h) = (SiLU(Q_h · K_e^{hT}) ⊙ (Q_h · U_e^{hT})) · V_e^h
        gate = SiLU(Q[:,h,:] · K_e^{hT})         # ∈ R^{L × d_e}
        up   = Q[:,h,:] · U_e^{hT}               # ∈ R^{L × d_e}
        out  = (gate ⊙ up) · V_e^h                # ∈ R^{L × d_h}
        S_h += R^h[:,e:e+1] ⊙ out                 # gated aggregation
      # S_h ∈ R^{L × d_h}, 平衡的d_e ≈ 8/3·d_h ≈ 342
  
  Step 4 - Head concat & output projection:
    O = concat_H([S_1, S_2, ..., S_H]) · W_out ∈ R^{L × d_model}
  ```
  
  关键设计参数：d_e ≈ 8/3·d_h（维持SwiGLU ratio），E = round(d_ff / d_e)。128M: H=6, E=8; 370M: H=8, E=7; 1.3B: H=16, E=15。
  
  与baseline对比：标准SwiGLU FFN = (SiLU(X·W_gate) ⊙ (X·W_up)) · W_down，单个大中间激活(R^{L×d_ff})。FlashMHF将其分解为H×E个更小的sub-network计算，每sub-network仅需要d_e维中间激活。

## Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现分为两部分：(1) 混合训练范式"Autoregressive-then-Diffusion"：Phase I 使用自回归训练（causal attention mask + next-token prediction loss）进行视觉-语言对齐和指令微调；Phase II 切换为扩散训练（full bidirectional attention mask + masked language modeling loss，仅对answer部分mask）恢复并行解码能力。
  (2) 推理技术：Confident Decoding（基于置信度阈值γ动态选择每步解码token数）、Prefilling（复用prompt的KV cache将注意力复杂度从O((L_prompt+L_answer)²)降至O(L_answer²)）、Structure Prior（通过预置token控制输出结构和长度）。
  实验比较：(i) 13个MLLM benchmark上与LLaVA-1.5-7B、LLaVA-NEXT-7B、Eagle-7B、Qwen-VL-7B、Eagle2-9B、Qwen2.5-VL-7B的性能对比（Table 1）；(ii) 纯扩散训练(DA+DT) vs 自回归后扩散(AA+AT+DT)的消融实验，6种策略比较（Table 2）；(iii) Prefilling对性能和速度的影响（Table 3），batch size=1和32下测量TPS加速比；(iv) Length Bias分析——纯扩散模型对response length敏感度（ChartQA上从42.7%降至8.6%）；(v) Confident Decoding、Structure Prior的定性展示（Table 4-6）。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 GPU集群，训练总计约100 GPU hours。Table 3的Prefilling消融实验在单张H100上进行（batch size=1模拟低GPU利用率，batch size=32模拟高GPU利用率）。

- 模型是什么。数据集和bench分别是什么。
  模型：Dimple-7B。组件：Qwen2.5-VL的vision encoder（ViT）、Dream（从Qwen2.5微调的DLM）作为LLM backbone、随机初始化的两层projector。
  训练数据：Phase I (alignment): LLaVA-CC3M-Pretrain (559k样本)；Phase I (instruction tuning) + Phase II: LLaVA-NEXT Instruction Tuning data (739k样本)。总计约1.3M训练样本，0.8B训练tokens。
  评估库：lmms-eval（https://github.com/EvolvingLMMs-Lab/lmms-eval）。
  Benchmarks：GQA、MMBench_en_test、MME (perception/cognition/total)、POPE、MMMU_val、SQA_img (ScienceQA)、AI2D、ChartQA、TextQA_eval、OCRBench、MathVista_test_mini、MMVet。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/yu-rp/Dimple；模型权重：https://huggingface.co/rp-yu/Dimple-7B；在线体验：https://huggingface.co/spaces/rp-yu/Dimple-7B。

  算法pipeline——训练（Autoregressive-then-Diffusion）：
  1. **初始模型**：Vision Encoder (Qwen2.5-VL ViT, 冻结) + 随机初始化2层MLP projector + Dream DLM (Qwen2.5 fine-tuned)。
  2. **Phase I - Autoregressive Alignment（1 epoch, lr=0.001, batch=256, 数据=LLaVA-CC3M-Pretrain 559k）**：
     a. 将Dream的full bidirectional attention替换为causal attention mask。
     b. 使用标准next-token prediction loss: L_AR = -Σ log p_θ(x_i | x_{<i})，对所有answer tokens计算loss。
     c. 训练projector（vision-language对齐），冻结vision encoder和LLM部分参数。
  3. **Phase I - Autoregressive Instruction Tuning（1 epoch, lr=5e-6, batch=128, 数据=LLaVA-NEXT 739k）**：
     a. 同causal attention，在instruction-following数据上继续训练。
     b. 使用LLaVA-NEXT training recipe和datasets。
  4. **Phase II - Diffusion Tuning（1 epoch, lr=5e-7, batch=128, 数据=LLaVA-NEXT 739k 复用）**：
     a. 恢复Dream的full bidirectional attention mask。
     b. 将[EOS]替换为随机数量的[padding] tokens（公式：n~RandomInteger(n_min, n_max)，其中n_min=l+1，n_max根据l确定）。
     c. Loss: L_D = E_t[(1/t) * E_{q(x_t|x_0)}[-Σ δ_{x_t^n, m} * (x_0^n)^T log f_θ(x_t)^n]]，仅mask answer部分。
     d. 时间步t∈(0,1]随机采样，使用absorbing-state (mask)扩散。

  算法pipeline——推理（Confident Decoding + Prefilling）：
  1. **输入**：图片经过Vision Encoder → visual tokens (projector映射)；文本prompt编码为token序列x。总长度L = L_prompt + L_answer。
  2. **Prefilling（首次forward）**：完整计算attention得到prompt的K/V cache。后续步骤仅需O(L_answer²)的attention计算。
  3. **Confident Decoding迭代（Alg. 1）**：
     输入：当前序列x_t、logits z_t、温度τ、阈值γ、fallback位置数K、masked token数N
     a. 计算pre-revision概率 p_t = softmax(z_t)；post-revision概率 p̃_t = softmax(z_t/τ)。
     b. 对每个masked位置i，计算confidence c_t^(i) = max(p_t^(i))（使用pre-revision概率）；采样候选token x̃_t^(i) ~ Categorical(p̃_t^(i))。
     c. 选择：若∃i满足c_t^(i)≥γ，则更新所有满足条件的位置（batch update）。否则fallback：随机选K个位置更新。
     d. 重复直到所有masked位置被填充或达到最大迭代次数。
  4. **Structure Prior**：在生成前预置特定位置的token（如"Thus, the answer is \box{"），这些位置在解码时标记为已确定，不参与mask/更新。

## Fast-dLLM v2: Efficient Block-Diffusion LLM

- 属于算法pipeline的实现是什么？实验比较什么？
  实现分为训练和推理两部分：(1) **Block Diffusion训练配方**——将预训练AR模型（Qwen2.5-Instruct）适配为block diffusion LLM，仅需~1B tokens微调（vs Dream的580B tokens）。核心设计包括：block-wise组织（block size=32，序列对齐到block整数倍并packing）、complementary masking（每个样本两个互补mask view，确保所有token既在masked又在unmasked上下文中被训练）、token shift（masked位置使用i-1位置的hidden state预测token i，保留AR模型的representation quality）、block-wise attention mask（M_BD block diagonal + M_OBC offset block causal + M_BC block causal，同时支持noised和clean上下文）。训练loss为masked-token-only cross-entropy。(2) **推理pipeline**——逐block自回归解码+block级KV cache复用+块内confidence-aware并行解码（来自Fast-dLLM v1）+ DualCache子块缓存。block size=32, sub-block size=8, 默认threshold=1（关闭并行解码），加速场景threshold=0.9。

  实验比较：(i) 1.5B和7B规模在HumanEval/MBPP/GSM8K/MATH/IFEval/MMLU/GPQA的benchmark性能 vs Qwen2.5-baseline、LLaMA-3.2、SmolLM-2、Dream、LLaDA系列（Table 1）；(ii) A100和H100上不同batch size的吞吐量 vs AR baseline（Figure 5）；(iii) 不同threshold下的accuracy-throughput trade-off（Figure 4, GSM8K）；(iv) 消融实验：naive token shift vs +pad vs +pad+CM（Table 2）；(v) 不同sub-block size的影响（Table 3）；(vi) 不同block size（与训练不一致）的影响（Table 4）；(vii) sub-block cache对accuracy和throughput的影响（Figure 6）。

- 硬件平台是什么，配置是什么。
  - 训练：64 × NVIDIA A100 GPU，DeepSpeed Zero-3，1.5B模型训练约8小时，7B模型约12小时
  - 推理评估：NVIDIA A100和H100 GPU，batch size 1-64，context length=2048，block size=32，sub-block size=8
  - throughput测量使用相同硬件对比AR baseline和diffusion方法

- 模型是什么。数据集和bench分别是什么。
  - 模型：Qwen2.5-Instruct 1.5B和7B（预训练AR模型微调得到Fast-dLLM v2）
  - 训练数据：LLaMA-Nemotron post-training dataset子集（高质量instruction-following数据）
  - Benchmark：HumanEval/MBPP（代码生成，EvalPlus评估）、GSM8K/MATH（数学推理）、IFEval（指令遵循）、MMLU/GPQA（知识密集型QA）
  - 评估框架：LM-Eval（非代码任务）、EvalPlus（代码任务），零样本设定（GPQA使用5-shot），greedy decoding

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文声明Code and model will be publicly released，但未提供具体链接。v1版本已开源：https://github.com/NVlabs/Fast-dLLM。

  **训练算法pipeline（以block size=32, context length L=2048为例）：**
  ```
  输入: 预训练Qwen2.5-Instruct模型参数θ, 训练数据D, block size D=32, mask概率t~Uniform(0,1)
  1: 对每个训练样本：
  2:     将序列右填充至D的整数倍（填充[MASK]不计loss）
  3:     将多个样本拼接packing至长度L，自然对齐为B=L/D个block
  4:     对每个block b采样随机binary mask m_b ∈ {0,1}^D
  5:     生成互补mask m̄_b = 1 - m_b
  6:     两个view放入同一个batch
  7:     对每个masked位置i（m_i=1处）：
  8:         使用position i-1的hidden state预测token i（token shift）
  9:     attention mask = [[M_BD, M_OBC], [0, M_BC]]，同时处理noised x_t和clean x_0
  10:     loss: -Σ 1[x_t^i=[MASK]] · log p_θ(x_0^i | x_{<i}, x_{block(i)})
  11: 使用AdamW优化器，linear warmup 500 steps
  ```

  **推理pipeline（以threshold=0.9，block size=32，sub-block size=8为例）：**
  ```
  输入: prompt p, 目标生成长度L, block size B=32, sub-block size S=8
  1: 将L拆分为K=⌈L/B⌉个block，每个block再分sub-block
  2: 初始化x = [p; [MASK]×L]
  3: for k = 1 to K:                                        // 逐block自回归
  4:     for each sub-block in block k:
  5:         forward pass（仅当前block+prefix，复用block级KV cache）
  6:         计算当前sub-block内[MASK]位置的confidence
  7:         confidence > τ的token并行解码，其余保留[MASK]
  8:         使用DualCache复用sub-block级的prefix/suffix KV
  9:     end
  10:     block k完成，刷新block级KV cache（作为后续block的prefix）
  11: end
  12: return decoded tokens
  ```

  **张量计算关键设计——Attention Mask分解（M_full ∈ {0,1}^{2L×2L}）：**
  - M_BD (Block-diagonal): [M_BD]_{ij}=1 iff i,j同属一个block → 块内双向自注意力
  - M_OBC (Offset block-causal): [M_OBC]_{ij}=1 iff j所在block在i所在block之前 → noised token attends to clean prefix
  - M_BC (Block-causal): [M_BC]_{ij}=1 iff j在i相同或更早的block → clean token间的AR-like因果依赖
  - 使用PyTorch flex-attention实现高效结构化masking

## Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现两个核心算法组件：(1) **Block-wise Approximate KV Cache**：针对Diffusion LLM的双向注意力特性，设计分块生成的近似KV Cache机制（PrefixCache和DualCache），利用相邻推理步之间KV激活的高余弦相似度（≈1），在块内复用缓存的Key和Value，块完成后统一更新所有token的KV Cache。(2) **Confidence-Aware Parallel Decoding**：提出基于置信度阈值的自适应并行解码策略（threshold strategy和factor strategy），仅解码置信度超过阈值/满足因子约束的token，从理论上证明了高置信度条件下贪婪并行解码等价于贪婪顺序解码（Theorem 1: (n+1)ε ≤ 1时二者argmax相同），在保证生成质量的同时实现最多13.3×并行解码加速。

  实验比较：(i) 消融实验：LLaDA baseline vs +Cache vs +Parallel vs +Cache+Parallel (Fast-dLLM) 四组对比，评估GSM8K/MATH/HumanEval/MBPP上accuracy和吞吐量(tok/s)；(ii) Dream模型上的泛化验证；(iii) Cache变体：PrefixCache vs DualCache vs No Cache（Table 4）；(iv) 并行解码策略：threshold vs factor vs 固定token-per-step baseline（Figure 5, Table 11）；(v) Cache block size消融(4/8/16/32)（Figure 4）；(vi) 不同生成长度(256/512/1024)和不同shot数(5-shot/8-shot)下的可扩展性（Table 4-5）；(vii) LLaDA-V多模态模型MathVista/MathVerse评估（Table 3, 9-10）；(viii) 不同batch size (1-32)下PrefixCache vs LLaDA vs LLaMA吞吐对比（Figure 9）；(ix) LLaDA vs LLaDA-1.5对比（Table 12）。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 80GB GPU（单卡），所有实验不使用任何推理加速框架（如vLLM/TensorRT-LLM）。prefill length=256 tokens（batch size scaling实验），生成长度16/32/64/256/512/1024。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaDA-Instruct（7B）、LLaDA-1.5（增强版）、Dream-Base（7B）、LLaDA-V（多模态vision-language变体）
  - Benchmark：GSM8K（5-shot数学推理）、MATH（4-shot竞赛数学）、HumanEval（0-shot代码生成）、MBPP（3-shot代码生成）、MathVista（视觉数学推理）、MathVerse（视觉数学推理）
  - 评估框架：lm-eval（标准化评估库），吞吐量指标为平均每秒生成的输出token数，计算完整序列到<eos>

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/NVlabs/Fast-dLLM（NVIDIA官方，Apache 2.0许可，ICLR 2026录用）。v1目录为本文训练无关加速方法。

  算法pipeline（以PrefixCache + Threshold策略，块大小B=32，阈值τ=0.9为例）：

  ```
  输入: pθ (MDM模型), prompt p0, 答案长度L, 块数K=⌈L/B⌉, 块大小B, 每块步数T, 阈值τ
   1: x ← [p0; [MASK]×L]                                    // 初始化：prompt后接全MASK序列
   2: 首次forward pass计算全序列attention，缓存prefix K/V       // KV Cache Init
   3: for k = 1 to K do                                      // 逐块解码
   4:     s ← |p0| + (k-1)B,  e ← |p0| + kB                  // 当前块起止索引
   5:     for t = 1 to T do
   6:         复用缓存prefix K/V，对x[s:e)（及prefix）计算attention  // 仅计算当前块attention
   7:         对x[s:e)中每个[MASK]位置i: c_i = max_v pθ(X_i=v|x)  // 置信度=最大softmax概率
   8:         找出所有c_i > τ的位置，解码argmax token              // 置信度阈值过滤
   9:         若所有c_i ≤ τ，解码max c_i token                    // 保底：防止死循环
  10:        if x[s:e)全部非MASK: break                         // 当前块完成
  11:     end for
  12:     重新计算全序列attention，更新prefix KV Cache            // 块间Cache更新（复杂度与解码融合）
  13: end for
  14: return x
  ```

  DualCache变体：额外在第2步缓存suffix（全[MASK]位置）的K/V，第6步仅需对当前块B×B的query-key计算自注意力。
  Factor策略：替换第8步为：排序{c_i}按降序{c^(1), c^(2), ..., c^(m)}，找最大n使(n+1)(1-c^(n)) < f，解码top-n tokens。

  张量计算对比（PrefixCache vs 无Cache，单step）：
  - 无Cache: Q × K_full^T, 其中Q ∈ R^{B×d}, K_full ∈ R^{(|p|+L)×d} → 计算量O(B·(|p|+L)·d)
  - PrefixCache: Q × [K_prefix||K_rest]^T, K_prefix ∈ R^{|p|×d}缓存复用, 仅需Q×K_rest^T（rest含当前块+suffix的自注意+交叉注意）→ 减少重复的Q×K_prefix^T计算
  - DualCache: 进一步缓存K_suffix/V_suffix, Q×K_block仅需B×B块内自注意力 → 计算量O(B²·d)


## FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning

- 属于算法pipeline的实现是什么？实验比较什么？
  对FlashAttention v1的online softmax算法做两项tweak以减少non-matmul FLOPs：(i) **前向**——不再对output累加的每一项做`diag(ℓ)^{-1}` rescale，改为维护"un-scaled" output并在所有KV blocks处理完后一次性做`diag(ℓ)^{-1}` rescale，消除每次内迭代对之前已累积output的rescale操作；(ii) **反向**——只存储logsumexp `L = m + log(ℓ)`（每行一个scalar, O(N) extra memory）替代同时存储rowwise max m和rowwise sum ℓ，反向从L恢复softmax denominator。
  实验比较：与FlashAttention v1 forward+backward speed对比（相同算法逻辑，不同non-matmul FLOPs数量）。与FlashAttention Triton实现对比。与xformers cutlass实现对比。与PyTorch标准attention对比。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100 80GB SXM4 GPU：FP16/BF16 matmul 312 TFLOPs/s，non-matmul FP32 19.5 TFLOPs/s（matmul与non-matmul吞吐比16:1，non-matmul FLOP实质比matmul FLOP贵16×）
  - NVIDIA H100 GPU（初步benchmark，未用TMA和FP8）

- 模型是什么。数据集和bench分别是什么。
  - 端到端训练模型：GPT3-1.3B（24 layers, hidden_dim=2048, 16 heads, head_dim=128）和GPT3-2.7B（32 layers, hidden_dim=2560, 32 heads, head_dim=80），sequence length 2k和8k
  - Benchmark数据集：论文未明确说明端到端训练使用的具体数据集（仅测量training throughput TFLOPs/s，非下游任务accuracy）
  - Attention benchmark：变sequence length（512-16K）、head dim（64/128）、causal/non-causal mask

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/Dao-AILab/flash-attention（BSD license）。安装：`pip install flash-attn`。

  **算法pipeline（以forward pass, 2 blocks简化为例, Q/K/V ∈ R^{B_r×d}）**：

  伪代码（对应Algorithm 1）：
  ```
  # FlashAttention-2 Forward (one row block)
  O_tilde = zeros(B_r, d)       # un-scaled output
  l = zeros(B_r)                # running sum of exp
  m = -inf * ones(B_r)          # running max
  
  for j = 1 to T_c:             # loop over KV blocks
      S = Q @ K_j.T             # [B_r, B_c] matmul (Tensor Core)
      m_new = max(m, rowmax(S)) # [B_r]
      m_rescale = exp(m - m_new) # [B_r], rescale factor for old values
      P_tilde = exp(S - m_new)  # [B_r, B_c] pointwise exp (non-matmul)
      l_new = m_rescale * l + rowsum(P_tilde)  # [B_r] (non-matmul)
      
      # Key difference from FlashAttention v1:
      # O_tilde stays un-scaled; no diag(l)^{-1} applied here
      O_tilde = diag(m_rescale) @ O_tilde + P_tilde @ V_j  # matmul
      
      m = m_new; l = l_new
  
  # Final rescale (only once, at the end):
  O = diag(l)^{-1} @ O_tilde
  L = m + log(l)  # logsumexp for backward
  ```
  vs FlashAttention v1（每次迭代都rescale output）：
  ```
  O = diag(l)^{-1} @ (diag(exp(m_old - m_new)) @ (diag(l_old) @ O_old) + P_tilde @ V_j)
  ```
  FlashAttention-2避免了每次迭代中`diag(l)^{-1}`对O_old的rescale操作（non-matmul elementwise multiply），改为最后一次统一rescale。同时反向只需L而非(m, l) pair，减少了register压力和non-matmul运算。

  **Non-matmul FLOPs对比（per iteration, per row block）**：
  - FlashAttention v1: rescale m_old→m_new (B_r mul) + rescale l_old (B_r mul) + exp(S-m_new) (B_r×B_c exp) + rowsum (B_r×B_c add) + rescale O_old by l_old/l_new (B_r×d mul) + rescale P_tilde by 1/l_new (B_r×B_c mul) ≈ B_r×(2 + 2B_c + 2d) non-matmul FLOPs
  - FlashAttention-2: rescale m_old→m_new (B_r mul) + rescale l_old (B_r mul) + exp(S-m_new) (B_r×B_c exp) + rowsum (B_r×B_c add) + rescale O_tilde by m_rescale (B_r×d mul, no l-based rescale) ≈ B_r×(2 + 2B_c + d) non-matmul FLOPs
  - 减少约 B_r×d 次non-matmul multiply per iteration。

## FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision

- 属于算法pipeline的实现是什么？实验比较什么？
  实现三个核心算法创新：(1) **Producer-Consumer asynchrony via warp-specialization**：将CTA内warps划分为producer（仅发射TMA数据搬运指令）和consumer（仅发射WGMMA计算指令）角色，通过setmaxnreg动态重分配寄存器，使用s-stage circular SMEM buffer pingpong调度，隐藏数据搬运延迟；(2) **2-stage GEMM-softmax pipelining**：在consumer warpgroup内部，通过寄存器缓冲$\mathbf{S}_{\text{next}}$打破迭代间的串行依赖，使得第j次迭代的softmax（CUDA core: rowmax FMNMX + EX2 MUFU.EX2 + rowsum FADD）与第j+1次迭代的QK^T WGMMA重叠执行，而第j次迭代的PV WGMMA与第j+1次迭代的softmax重叠；(3) **FP8 block quantization with incoherent processing**：对Q/K/V逐block（B_r或B_c粒度）量化并保持per-block scaling factor，Q和K先乘随机正交矩阵M（Hadamard + random sign diagonal product）进行incoherent processing以消除outlier，再量化为FP8 (e4m3)格式送入FP8 tensor core执行WGMMA。
  实验比较：(i) Forward speed (TFLOPs/s) vs FlashAttention-2、FlashAttention-2 in Triton、cuDNN attention、standard PyTorch attention，在H100 GPU上seqlen 512-16K；(ii) Backward speed vs FlashAttention-2、FlashAttention-2 in Triton；(iii) FP8 forward speed vs BF16 baselines；(iv) 消融实验：warp-specialization和GEMM-softmax pipelining各自对性能的贡献（固定参数batch=4, seqlen=8448, nheads=16, hdim=128）；(v) 数值精度（RMSE）验证：FP16 vs standard attention + FP64 reference，FP8 vs per-tensor quantization baseline，并消融block quantization和incoherent processing各自对精度的贡献。

- 硬件平台是什么，配置是什么。
  - NVIDIA H100 80GB SXM5 GPU (Hopper架构, 700W)：989 TFLOPS FP16/BF16 matmul理论峰值，3.9 TFLOPS special functions (exponential)，80 GiB HBM @ 3.35 TB/s，228 KiB SMEM per SM，132 SMs，GPU boost clock 1830 MHz
  - CUDA 12.3, cuDNN 9.5.0.50, CUTLASS 3.6, FlashAttention 2.6.3, Triton 3.1, PyTorch 2.5.0
  - Benchmark固定GPU clock speed为1830MHz，重复10次取平均以减少variability

- 模型是什么。数据集和bench分别是什么。
  - Attention配置：hidden dim 2048，head dim 64/128/256（对应32/16/8 heads），seqlen 512-16384，batch size使得总token数为16K
  - 支持MHA、MQA (multi-query attention)、GQA (grouped-query attention)
  - 数值精度验证：使用合成数据$\mathbf{Q},\mathbf{K},\mathbf{V} \sim \mathcal{N}(0,1) + \mathcal{N}(0,100) \cdot \text{Bernoulli}(0.001)$模拟LLM中的outlier features和activations，以FP64 reference为ground truth
  - 包含causal mask和无mask两种场景

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/Dao-AILab/flash-attention（BSD许可证），计划集成到PyTorch。
  
  算法pipeline（以FP16前向一次CTA级forward pass为例，输入$\mathbf{Q}_i \in \mathbb{R}^{B_r \times d}$和$\mathbf{K},\mathbf{V} \in \mathbb{R}^{N \times d}$，block sizes $B_r$, $B_c$）：
  1. **Warp-specialization setup**：CTA内warps分为producer warpgroup（使用setmaxnreg释放register）和consumer warpgroup（使用setmaxnreg申请更多register用于WGMMA）。
  2. **Producer mainloop**：TMA async load $\mathbf{Q}_i$ from HBM→SMEM → commit通知consumer。对$j=0..T_c-1$：wait for stage $(j\%s)$ consumed → TMA async load $\mathbf{K}_j$, $\mathbf{V}_j$ from HBM→SMEM at stage $(j\%s)$ → commit通知consumer。
  3. **Consumer mainloop（2-stage pipelining, Algorithm 2）**：
     a. **Prologue (j=0)**：Wait for $\mathbf{Q}_i$, $\mathbf{K}_0$ in SMEM → SS-WGMMA: $\mathbf{S}_{\text{cur}} = \mathbf{Q}_i \mathbf{K}_0^T$ (commit+wait) → 释放K的stage 0 → softmax: rowmax+EX2+rowsum → 计算$m_i$, $\tilde{\mathbf{P}}_{\text{cur}}$, $\ell_i$, rescale $\mathbf{O}_i$。
     b. **Mainloop ($j=1..T_c-1$)**：Wait for $\mathbf{K}_j$ → SS-WGMMA: $\mathbf{S}_{\text{next}} = \mathbf{Q}_i \mathbf{K}_j^T$ (commit, no wait) → Wait for $\mathbf{V}_{j-1}$ → RS-WGMMA: $\mathbf{O}_i += \tilde{\mathbf{P}}_{\text{cur}} \mathbf{V}_{j-1}$ (commit, no wait) → Wait for $\mathbf{S}_{\text{next}}$ WGMMA → softmax on $\mathbf{S}_{\text{next}}$: rowmax+EX2+rowsum, compute $m_i$, $\tilde{\mathbf{P}}_{\text{next}}$, $\ell_i$ → Wait for PV WGMMA → rescale $\mathbf{O}_i$ → release stages → copy $\mathbf{S}_{\text{next}} \to \mathbf{S}_{\text{cur}}$。
     c. **Epilogue**：Wait for $\mathbf{V}_{T_c-1}$ → RS-WGMMA: $\mathbf{O}_i += \tilde{\mathbf{P}}_{\text{last}} \mathbf{V}_{T_c-1}$ (commit+wait) → resize $\mathbf{O}_i = \operatorname{diag}(\ell_i)^{-1}\mathbf{O}_i$, $L_i = m_i + \log(\ell_i)$ → write $\mathbf{O}_i$, $L_i$ to HBM。
  4. **FP8 variant变更**：(a) Q/K必须k-major布局（contiguous in head dimension），V需要m-major布局（contiguous in seqlen dimension），通过in-kernel SMEM→RMEM→SMEM transpose（LDSM + byte_perm + STSM）解决；(b) FP32 accumulator → FP8 operand layout转换通过byte_perm和shfl_sync组合实现register data exchange。

## FlowMM Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  实现FlowMM框架的两个核心组件：(1) **Cross-Modal Information Flow Guided Merging**——通过分析MLLM各层的cross-modal attention比例$\rho^l$（公式6-7），当$\rho^l$超过阈值$\theta$时执行跨模态合并（inter-modal merging），低于$\theta$时执行模态内合并（intra-modal merging），实现层自适应（layer-specific）的KV cache合并策略。同时使用proxy tokens评估token重要性$\mathcal{I}^{l,h}(i)$（公式8），选出top-B pivot tokens保留最关键任务信息，将non-pivot tokens合并到pivot set中。(2) **Sensitivity-Adaptive Token Matching**——通过余弦相似度$u_{i,j}$（公式9）评估token相似度，同时用attention scores作为token敏感度（sensitivity）的近似度量，设置敏感度阈值$\tau$来保护高敏感度token（公式10：$\mathcal{I}_j \le \tau$），仅合并低敏感度token以最小化任务关键信息损失。
  实验比较：(i) 在Qwen2.5-VL-7B、InternVL2.5-8B、MobileVLM-V2-3B三个MLLM上与5个baseline对比——StreamingLLM、H2O（eviction类）、D2O、KVMerge（text-based merging类）、LOOK-M（multimodal-specific merging类），cache budget=20%下的accuracy/ROUGE-L（Table 1）；(ii) 不同cache budget（5%-60%）下的性能对比（Figure 4）；(iii) 效率分析：decoding latency和GPU memory usage vs full cache（Table 2）；(iv) 消融实验：cross-modal merging threshold $\theta$的影响（Table 3）、各组件有效性（Table 4）。

- 硬件平台是什么，配置是什么。
  - 单张NVIDIA A100 Tensor Core GPU（80GB），用于效率分析和消融实验
  - 所有实验在单卡上进行，测量decoding speed（ms/token）和GPU memory consumption（GiB）

- 模型是什么。数据集和bench分别是什么。
  - 模型：Qwen2.5-VL-7B（Bai et al., 2025）、InternVL2.5-8B（Chen et al., 2024b）、MobileVLM-V2-3B（Chu et al., 2024），涵盖不同架构设计的MLLM
  - Benchmark：MileBench（Song et al., 2024）——首个专门测试MLLM长上下文多模态能力的benchmark，平均每样本15.2张图和422.3个词
  - 评估的7个任务：ALFRED（Conversational Embodied Dialogue, ROUGE-L）、IEdit（Visual Relationship Expressing, ROUGE-L）、STD（Visual Change Captioning, ROUGE-L）、MMCoQA（Multimodal Dialogue, Accuracy）、CLEVR-C（Visual Change Captioning, ROUGE-L）、TextNeedle（Text Needle In A Haystack, Accuracy）、ImageNeedle（Image Needle In A Haystack, Accuracy）
  - 评估框架：论文未明确说明（直接使用MileBench提供的评估脚本）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未给出开源链接。论文未在正文或附录中提供代码仓库URL。

  算法pipeline（FlowMM一次完整前向推理，以Qwen2.5-VL-7B推理一个多模态输入为例）：

  **Phase 1 - 信息流分析（离线或首次推理时进行）**：
  ```
  输入: MLLM的L层transformer, 校准样本集D, 阈值θ
  对每个校准样本:
    对每层 l = 1..L:
      对每个attention head h = 1..H:
        计算cross-modal attention: 
          A_{v→t}^{l,h} = Σ_{v∈V} Σ_{t∈T} α_{v→t}^{l,h}  # visual→text attention
          A_{t→v}^{l,h} = Σ_{t∈T} Σ_{v∈V} α_{t→v}^{l,h}  # text→visual attention
      计算cross-modal interaction ratio:
        ρ^l = (1/H) · Σ_h (A_{v→t}^{l,h} + A_{t→v}^{l,h}) / A^{l,h}
    判断每层合并策略:
      if ρ^l ≥ θ: 层l → cross-modal merging（跨模态合并）
      else:        层l → intra-modal merging（模态内合并）
  ```

  **Phase 2 - KV Cache Merging（每层运行时执行）**：
  ```
  输入: 第l层的KV cache K_t, V_t ∈ R^{L_t×d}, head h
  策略: merge_strategy = cross-modal 或 intra-modal（由Phase 1决定）
  预算: 保留比例B（如20%）
  
  Step 1 - Token重要性评估:
    选proxy tokens P = 最后若干prompt tokens
    For each token i:
      I^{l,h}(i) = Σ_{j∈P} α_{j→i}^{l,h}  # proxy attention聚合
  
  Step 2 - 选择pivot set:
    按I排序，选top-B tokens作为pivot set K^p
    其余tokens作为non-pivot set K^n
    # K^p保留最关键任务信息，K^n将被合并
  
  Step 3 - Sensitivity-Adaptive Token Matching（仅在K^n→K^p方向，受merge_strategy约束）:
    设定sensitivity threshold τ
    For each token i in K^n:
      计算与K^p中所有token j的cosine similarity:
        u_{i,j} = k_i^T k_j / (||k_i|| · ||k_j||)
      找最近邻（仅考虑低敏感度pivot）:
        k_*^{nearest} = Argmax_{j∈K^p, I_j≤τ}(u_{i,j})
      合并: 将token i的KV状态合并到k_*^{nearest}:
        K_*^{merged} = weighted_avg(K_*, K_i)  # attention-weighted averaging
        V_*^{merged} = weighted_avg(V_*, V_i)
      # 若merge_strategy=intra-modal，仅在同模态内搜索最近邻
      # 若merge_strategy=cross-modal，允许跨模态搜索
  
  输出: 压缩后的K^{merged}, V^{merged} ∈ R^{L_compressed×d}
  ```

  关键张量计算：
  - Cross-modal ratio $\rho^l$对每层为标量，O(L·H·|V|·|T|)但仅需一次前向即可确定
  - Token importance: $\mathcal{I}^{l,h} \in \mathbb{R}^{L_t}$，每个token一个标量值
  - Cosine similarity矩阵: $u \in \mathbb{R}^{|K^n|\times|K^p|}$，按merge_strategy过滤候选集
  - KV合并: 加权平均，O(d) per merge operation
  - 总计算开销: 相比full cache inference增加约5-10%（主要来自cosine similarity计算和attention score聚合），但无fine-tuning需求

  与Baseline的关键差异：
  - Baseline（KVMerge等）：所有层使用统一合并策略（uniform merging），未区分cross-modal vs intra-modal层
  - FlowMM：每层根据cross-modal attention flow自适应选择合并策略——浅层（$\rho^l$低）做intra-modal保留模态特征，深层（$\rho^l$高）做cross-modal促进跨模态融合
  - Baseline（LOOK-M等）：基于相似度直接合并，不考虑token sensitivity
  - FlowMM：引入sensitivity threshold $\tau$保护高敏感度token（如包含任务关键信息的特殊token），仅合并低敏感度token

## FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos

- 属于算法pipeline的实现是什么？实验比较什么？
  实现一个全自动数据标注pipeline（FoundationMotion Pipeline），包含四个阶段：(1) Video Preprocessing——对视频做5-10秒temporal cropping，使用VGGT检测并过滤显著相机运动的视频；(2) Object Detection and Tracking——使用Qwen2.5-VL-7B做开放词汇目标识别+Grounded-DINO做定位，使用Cascade Mask R-CNN (ViTDet-H)+ViTPose++Hands23做人体/手部检测，使用SAM2做跨帧时序tracking（两阶段：初始tracking每帧传播+每5帧keyframe refinement）；(3) Caption Generation——将tracking输出（归一化bbox轨迹JSON、视频帧、彩色bbox overlay）输入GPT-4o-mini，按7维度prompt生成motion caption；(4) QA Generation——基于caption和视频帧，使用GPT-4o-mini生成5类QA（Motion Recognition, Action Order/Temporal Ordering, Motion-related Objects, Location-based Motion, Repetition Count）。在InternVid的46.7k视频上运行该pipeline，生成467K caption/QA-video pairs作为FoundationMotion Dataset。然后fine-tune NVILA-Video-15B/8B和Qwen2.5-VL-7B在MotionBench、VLM4D和自建zero-shot benchmarks上对比评估，baseline包括Gemini-2.5-Flash、Qwen2.5-VL-72B、PerceptionLM同量数据finetune。

- 硬件平台是什么，配置是什么。
  8×A100 GPUs（training和testing均使用）。论文未明确说明A100的具体显存配置（40GB或80GB）。

- 模型是什么。数据集和bench分别是什么。
  - 被fine-tune的base模型：NVILA-Video-15B、NVILA-Video-8B、Qwen2.5-VL-7B
  - 数据标注pipeline中使用的模型：Qwen2.5-VL-7B（开放词汇检测）、Grounded-DINO（目标定位）、Cascade Mask R-CNN with ViTDet-H backbone（人体检测）、ViTPose+（关键点提取）、Hands23（手部检测+交互分析）、SAM2（视频tracking）、VGGT（相机运动检测）、GPT-4o-mini（caption和QA生成）
  - 训练数据：InternVid（随机采样5秒clips，通过FoundationMotion pipeline标注）
  - 评估benchmarks：
    - 公开benchmark：MotionBench（5385 videos, 8052 QA pairs, 6 motion tasks）、VLM4D（1000 videos, 1800 QA pairs）
    - 自建zero-shot benchmark：AV-Car（NuScenes, 1968 QAs）、AV-Hand（NuScenes, 108 QAs）、Daily（100 Days of Hands, 832 QAs）、Robotics（YouTube, 102 QAs）
  - Baseline对比模型：Gemini-2.5-Flash、Qwen2.5-VL-72B、NVILA-Video-15B/8B base、Qwen2.5-VL-7B base、PLM（PerceptionLM）同量数据finetune

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  全面开源：Code: https://github.com/Wolfv0/FoundationMotion；Dataset: https://huggingface.co/datasets/WoWolf/v2-dev；Model: https://huggingface.co/WoWolf/models。
  
  **Algorithm Pipeline 伪代码**：
  ```
  # Stage 1: Video Preprocessing
  t_s ~ U(5, min(10, t_v))
  t_start = max(0, min(t_v - t_s, t_v/2 - t_s/2 + eps)), eps ~ U(-0.2*t_v, 0.2*t_v)
  clip = video[t_start : t_start + t_s]
  if VGGT.camera_motion_score(clip) > 0.3: skip  # filter high camera motion
  
  # Stage 2: Object Detection & Tracking
  O = Qwen2.5-VL-7B.detect(clip[0])           # open-vocab object categories
  B_obj = GroundedDINO(clip[0], O)              # bounding boxes per category
  B_person = CascadeMaskRCNN_ViTDetH(clip[0])   # person detections (tau=0.8)
  K = ViTPose+(clip[0], B_person)               # keypoints incl. 42 hand kpts
  for each person p:
      hands[p] = Hands23.detect(expand_region(K.hand_kpts[p], 1.5x))
  M_0 = SAM2.initialize(clip[0], B_obj + B_person + hands)  # init masks
  for t in 1..T:
      M_t = SAM2.propagate(M_{t-1})
      if t % 5 == 0:  # keyframe refinement
          B_new = Hands23.detect(clip[t])
          M_t = SAM2.propagate(M_{t-1}, B_new)
  
  # Stage 3: Caption Generation
  json_trajectories = {obj_id: {bbox: [[l,t,r,b]_t for t], object_type, interactions}}
  frames_2fps = sample(clip, fps=2)
  overlay = draw_bbox_overlay(frames_2fps, json_trajectories, color_coded=True)
  caption = GPT4o_mini(frames_2fps, json_trajectories, overlay, prompt_7dim)
  
  # Stage 4: QA Generation (5 types)
  qa_pairs = GPT4o_mini(frames, caption, prompt_5categories)
  # Categories: MotionRecognition, ActionOrder, MotionRelatedObjects,
  #             LocationBasedMotion, RepetitionCount
  # Output: [{"Q":..., "A":..., "B":..., "C":..., "D":..., "type":...}]
  
  # Stage 5: Fine-tuning
  for model in [NVILA-Video-15B, NVILA-Video-8B, Qwen2.5-VL-7B]:
      if model is Qwen-based:
          trainer = llamafactory(model, lr=1e-5, optimizer=Adam)
      else:  # NVILA
          trainer = NVILA_official(model, lr=1.5e-5, optimizer=Adam)
      trainer.fit(FoundationMotion_Dataset, cosine_annealing, no_weight_decay)
  ```

  **张量计算示意（Tracking核心）**：
  - BBox轨迹存储：traj in R^{N_obj x T x 4}，其中4维为 [left/width, top/height, right/width, bottom/height] 归一化坐标
  - SAM2 propagation：M_t = SAM2.predictor.propagate_in_video(M_{t-1})，M_t为frame t的segmentation masks {m_i in {0,1}^{HxW}}
  - ID层级编码：persons ID in [0,99]，left_hand=ID*10+1，right_hand=ID*10+4，objects ID >= 1000

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
