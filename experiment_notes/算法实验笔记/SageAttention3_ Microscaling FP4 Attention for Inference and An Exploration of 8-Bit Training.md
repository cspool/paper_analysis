## SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training

- 属于算法pipeline的实现是什么？实验比较什么？
  - **SageAttention3 (FP4 inference)**：利用 Blackwell GPU 的 NVFP4 Tensor Core（E2M1 数据类型，1×16 量化块，E4M3 scale factor），对 attention 中的两个矩阵乘法 QK^T 和 PV 应用 FP4 microscaling quantization。同时提出 two-level quantization 解决 attention map P 的量化精度问题：先 per-token 归一化到 [0, 448×6]，再做 FP4 microscaling 量化。继承 SageAttention2 的 Smoothing Q 和 Smoothing K 技术。
  - **SageBwd (INT8 training)**：前向对 QK^T 和 PV 做 per-block INT8 量化，对 P 做 per-token INT8 量化（复用 online softmax 的 max 值避免额外 max 操作）。反向中保持 dOV^T 在 FP16（因其误差主导 dQ/dK 的梯度精度），其余四个 MatMul 做 INT8 per-block 量化。
  - 实验比较：与 FlashAttention2、xformers、SageAttention、SageAttention2 对比 kernel speed (TOPS) 和端到端推理/训练指标。

- 硬件平台是什么，配置是什么。
  - SageAttention3 推理：NVIDIA RTX5090 (Blackwell 架构，支持 FP4 Tensor Core)
  - SageBwd 训练：NVIDIA RTX4090
  - 理论对比包含 B300、B200、H100 的 TOPS

- 模型是什么。数据集和bench分别是什么。
  - **文生文模型**：Qwen2.5 (1.5B, 3B)、Llama3.2 (1B, 3B)；数据集：GSM8K、DROP、MMLU、HELLASWAG
  - **文生视频模型**：CogVideoX (2B)、HunyuanVideo、Mochi；数据集：Open-Sora prompt sets；metrics：CLIPSIM、CLIP-T、VQA-a、VQA-t、FScore
  - **文生图模型**：Flux、Stable-Diffusion3.5；数据集：COCO annotations；metrics：FID、sFID、CLIP、ImageReward
  - **预训练**：Llama 400M (hidden=1024, 20 layers, intermediate=3072, 16 heads)；数据集：FineWebEdu

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/thu-ml/SageAttention
  - FP4 算法流程（对应论文 Algorithm 1）：输入 FP16 Q, K, V ∈ R^{N×d}，分块 B_q, B_{kv}。对 Q_i 做 Smoothing（减去均值），然后 φ(Q_i - q̄_i) → (s_Q, Q̂_i) 量化到 NVFP4（1×16 块，E2M1+E4M3）。对 K_j^T 做 φ(K_j^T) → (s_K, K̂_j)，对 V_j 做 φ(V_j) → (s_V, V̂_j)。S_{ij} = FP4MM(Q̂_i, s_Q, K̂_j, s_K) + GEMV(q̄_i, K_j^T)。online softmax 计算 P̃_{ij} = exp(S_{ij} - m_{ij})。Two-level quantization: s_{P1} = rowmax(P̃_{ij})/(448×6)，P̃_{ij} = P̃_{ij}/s_{P1}，然后 s_{P2}, P̂_{ij} = φ(P̃_{ij})。O_{ij} = diag(e^{m_{i,j-1}-m_{ij}})O_{i,j-1} + FP4MM(P̂_{ij}, s_{P2}, V̂_j, s_V) × s_{P1}。最终 O_i = diag(l_{i,T_n})^{-1} O_{i,T_n}。
