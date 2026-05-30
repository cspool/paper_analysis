## SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：SageAttention，一种面向Attention的INT8后训练量化（PTQ）方法，即插即用加速推理。核心技术：(1) Smooth K — 对K矩阵按token维度减去均值mean(K)消除channel-wise outlier，数学上等价于原始softmax（$q(K-\text{mean}(K))^\top = qK^\top - q\cdot\text{mean}(K)^\top$），不影响attention score；(2) INT8 per-token/per-block量化Q和K，利用INT8 Tensor Core加速$QK^\top$ Matmul；(3) FP16 accumulator for PV Matmul — 保留P和V在FP16精度，使用FP16 accumulator代替FP32 accumulator，2×加速$PV$ Matmul而零精度损失；(4) Adaptive Quantization — 对每个layer自动在SAGEAttn-B（QK INT8 per-block + PV FP16）和SAGEAttn-vB（全INT8，PV也INT8量化）中选择，cosine similarity > 99.8%的层选vB（约4%更快），其余选B（更准确）。
  - 实验比较：speed对比FlashAttention2、xformers、Torch Attention（TOPS和真实模型延迟）；accuracy对比FP16 full-precision attention；quantization method对比FlashAttention3 FP8版本、per-token/per-block/per-tensor INT8量化。End-to-end评估覆盖Llama2-7B（WikiText perplexity, LAMBADA, MMLU accuracy）、CogvideoX（CLIPSIM, CLIP-T, VQA-a, VQA-t, FScore）、Unidiffuser/UltraPixel（FID, sFID, CLIP, ImageReward）、TIMM（ImageNet, Sketch, ImageNet-r accuracy）、Llava1.6（TextVQA, POPE, VQAv2）。消融实验包括smooth K的overhead（<0.2%）和adaptive quantization的收益（+11.7% OPS）。

- 硬件平台是什么，配置是什么。
  - GPU: NVIDIA RTX 4090（PCIE 5.0, 16-core Xeon 6430 CPU, 120GB DDR4 RAM）；NVIDIA RTX 3090（16-core Xeon 8358P CPU, 80GB DDR4 RAM）。
  - 软件环境: Ubuntu 22.04, torch 2.4.0+cu121, triton-nightly (20240816), python 3.11, gcc/g++ 9。

- 模型是什么。数据集和bench分别是什么。
  - 模型: Llama2-7B（text2text）, CogvideoX（text2video）, Unidiffuser（text2image）, UltraPixel（text2image）, TIMM vit_base_patch16_224（image classification）, Llava1.6（visual QA）。
  - 数据集/benchmark: WikiText（perplexity）, LAMBADA（contextual understanding）, MMLU（knowledge）；Open-Sora prompt sets（video generation）；COCO 2014val（前256条annotations用于图像生成FID/sFID计算）；ImageNet, ImageNet-Sketch, ImageNet-Rendition（分类）；TextVQA, POPE, VQAv2（VQA）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接: https://github.com/thu-ml/SageAttention（MIT License）。
  - 算法pipeline（SAGEAttn-B, Algorithm 1）:
    1. Preprocessing: K_smooth = K - mean(K)  # mean over token dim, shape 1×d
    2. Quantization: (δ_Q, Q̂) = ψ_Q(Q/√d), (δ_K, K̂) = ψ_K(K_smooth)  # INT8 per-block
    3. Tiling: 将Q̂分为T_m=N/b_q块{Q̂_i}，K̂和V分为T_n=N/b_kv块{K̂_j},{V_j}
    4. For i in [1, T_m] (并行于SMs):
       For j in [1, T_n]:
         S_i^j = Matmul(Q̂_i, K̂_j^T) × δ_Q[i] × δ_K[j]  # INT8 Tensor Core, dequant via scale multiplication
         (m_i^j, P̃_i^j) = online_softmax(m_i^{j-1}, S_i^j)  # FP16
         O_i^j = diag(e^{m_i^{j-1}-m_i^j})O_i^{j-1} + Matmul(P̃_i^j, V_j, accum=FP16)  # FP16 accumulator
       O_i = diag(l_i^{T_n})^{-1}O_i^{T_n}
    5. Q,K量化粒度选择: per-token (SAGEAttn-T) 或 per-block (SAGEAttn-B)。P使用per-block（因P每行max=1，静态scale=1/127），V使用per-channel（解决channel-wise outlier）。
    6. Adaptive: 对每layer测试SAGEAttn-vB cosine sim，若>99.8%则用vB（全INT8，快4%），否则用B（FP16 PV accumulator）。
