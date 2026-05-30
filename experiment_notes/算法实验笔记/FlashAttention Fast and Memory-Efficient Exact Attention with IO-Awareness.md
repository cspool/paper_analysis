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
