## MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出三方面通信优化加速 MoE 大规模训练：(1) 为 attention 和 FFN 分别定制通信高效并行策略——attention 用 Sequence Parallelism (SP，基于 DeepSpeed-Ulysses 的 all-to-all 风格)、FFN/experts 用 Expert Parallelism (EP)，替代传统 Tensor Parallelism (TP)；(2) 通信压缩——BF16 训练中将 DP 梯度同步精度从 FP32 降至 BF16（all-to-all 替代 reduce-scatter + FP32 本地累积），FP8 训练中用 FP8 all-to-all 替代 BF16 reduce-scatter（per-token activation quantization + per-channel/group quantization）；(3) selective activation rematerialization，仅保留计算密集的中间激活，低成本的通过重计算/重通信获得，节省约 50% 激活内存。
  - 实验比较 MegaScale-MoE vs Megatron-LM（commit f1f03922），包括 strong scaling（240-1440 GPU，固定 global batch 720）、weak scaling（480-1440 GPU，batch 360→1080 等比增长）、不同 GPU 平台（H800/A100/H20）性能分解、ablation study 逐步启用 SP+EP → inter-operator overlap → intra-operator overlap。
  - 评估六种 MoE 模型：Internal-352B（60 layers, h=4096, 32 experts, top-k=3）、Mixtral-8×7B、Mixtral-8×22B、Hunyuan-Large、Phi-3.5-MoE、DeepSeekMoE。

- 硬件平台是什么，配置是什么。
  - 主要平台：NVIDIA H800 SXM GPU（Compute 989 TFLOPS, 80 GB HBM, 3.4 TB/s 内存带宽, NVLink 400 GB/s），最多 1,440 GPUs。
  - 对比平台：NVIDIA A100（312 TFLOPS, 80 GB, 2.0 TB/s, NVLink 600 GB/s）、NVIDIA H20（148 TFLOPS, 96 GB, 4.0 TB/s, NVLink 900 GB/s），各 32 GPUs。
  - 训练精度：BF16 mixed-precision 和 FP8（E4M3）。
  - 网络：intra-node NVLink + inter-node RDMA（NIC 50 GB/s 量级）。
  - Sequence length=8192, vocabulary size=65536。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Internal-352B MoE（60 layers, h=4096, 32 experts, top-k=3, GQA m=4, SwiGLU FFN h_ffn=14336），以及 Mixtral-8×7B、Mixtral-8×22B、Hunyuan-Large、Phi-3.5-MoE、DeepSeekMoE 等五个开源 MoE 模型。
  - 数据集：论文未明确说明具体训练数据集名称。用于验证 FP8 收敛性的 35B 和 176B MoE 模型也未指定数据集。
  - Benchmark：训练吞吐量（tokens/s）、MFU（Model FLOPs Utilization）、iteration time、loss curve 收敛性、内存占用。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未提供开源链接（论文本身发表在 EuroSys 2026，未在文中提供 GitHub 仓库地址）。
  - 系统基于 Megatron-LM 构建（开源：github.com/NVIDIA/Megatron-LM）。
  - 算法pipeline 核心计算流程（单 MoE 层 forward，基于论文 §4.1 Figure 8-9）：
    1. Input: hidden [b, s/n, h] → RMSNorm → ln1_out [b, s/n, h]
    2. QKV Projection: qkv = MatMul(ln1_out, qkv_weight) → [b, s/n, h(1+2/m)]
    3. RoPE on q, k → q_rope [b, s/n, h], k_rope [b, s/n, h/m]
    4. SP Attention: All-to-All(q_rope, k_rope, v) → qkv_a2a [b, s, h(1+2/m)/n]
    5. SelfAttention(qkv_a2a) → attn [b, s, h/n]
    6. All-to-All(attn) → attn_a2a [b, s/n, h]
    7. Output Projection: attn_out = MatMul(attn_a2a, out_weight) → [b, s/n, h]
    8. Residual: ln2_in = Add(hidden, attn_out) → RMSNorm → ln2_out [b, s/n, h]
    9. Expert dispatch: All-Gather(ln2_out) → ln2_out_ag [b, s, h] → Scatter → ffn_in [b*s*k/n, h]
    10. SwiGLU FFN: fc1_out = GroupedGEMM(ffn_in, fc1_weight), fc3_out = GroupedGEMM(ffn_in, fc3_weight), fc2_in = SiLU(fc1_out) * fc3_out, fc2_out = GroupedGEMM(fc2_in, fc2_weight) → [b*s*k/n, h]
    11. Gather(fc2_out) → fc2_out_rs [b, s, h] → Reduce-Scatter → ffn_out [b, s/n, h]
    12. Residual: hidden(next) = Add(ln2_in, ffn_out)
  - 当 top-k > n 时，EP 通信从 all-to-all 切换为 all-gather + reduce-scatter（环形通信更高效）。
  - DP 通信压缩：梯度本地 FP32 累积后 cast 到 BF16 → all-to-all（替代 reduce-scatter）→ 本地 FP32 聚合，通信量减半。
