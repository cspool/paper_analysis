## DeepSeek-V3 Technical Report

- baseline方法是什么？
  Baseline 为 DeepSeek-V2 架构（MLA + DeepSeekMoE + auxiliary-loss-based load balancing + next-token prediction only），以及 BF16 训练范式。具体痛点：(1) **Auxiliary Loss 干扰模型性能**：传统 MoE 使用 auxiliary loss 强制负载均衡，但过大的 auxiliary loss 会损伤模型性能（trade-off between load balance and model quality）；序列级 auxiliary loss 要求每个序列内部负载均衡，限制了专家的领域特化能力。(2) **Next-token prediction 训练信号稀疏**：每个 position 仅预测下一个 token，训练信号密度低，数据效率有限。(3) **BF16 训练通信与内存开销大**：BF16 训练的 activation、通信和 optimizer state 占用大量 GPU 内存和带宽；跨节点 MoE 通信开销与计算量之比约为 1:1，成为训练瓶颈。(4) **Pipeline parallelism bubble 大**：传统 1F1B 和 ZB1P pipeline parallelism 仍存在显著的 bubble 和通信-计算串行问题。(5) **推理部署资源需求大**：MoE 推理需要大量 GPU 才能高效运行，小型团队难以部署。

  **Baseline 全栈执行例子（以 DeepSeek-V2, 236B total/21B activated, 单 token decode 为例）**：
  - **算法层**: MLA (d_c=512) + DeepSeekMoE (2 shared + 160 routed, K_r=6, Sigmoid gating with top-K normalization, auxiliary-loss-based load balancing)。Next-token prediction only。
  - **系统框架层**: HAI-LLM 框架，BF16 训练，HAI-LLM 框架上的标准 1F1B/ZB1P pipeline parallelism + expert parallelism + ZeRO-1 data parallelism。All-to-all 通信使用 NCCL，无 overlap。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 标准 FlashAttention kernel，NCCL all-to-all 通信 kernel。BF16 GEMM on Tensor Cores。
  - **硬件架构层**: 8×H800 GPU/节点 × N nodes。NVLink + NVSwitch 节点内，InfiniBand 跨节点。BF16 参数/activation 存储，FP32 optimizer states。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **(1) Auxiliary-Loss-Free Load Balancing**：引入每个 expert 的 bias 项 b_i，仅在 routing 时加到 affinity score 上（s_{i,t}+b_i 决定 Top-K），每 training step 结束时动态调整：过载专家 b_i -= γ(0.001)，欠载专家 b_i += γ(0.001)。Gating value 仍使用原始 s_{i,t}。仅保留极小的 sequence-wise balance loss (α=0.0001) 防止极端不均衡。消除 auxiliary loss 对性能的负面影响，同时通过 batch-wise balancing 允许专家在不同 domain 上特化（Pile-test 证实 auxiliary-loss-free 模型展现更强的 domain-specific expert specialization patterns）。

  **(2) Multi-Token Prediction (MTP)**：增加 1-depth MTP 模块，每个 position 额外预测下下个 token，保持完整 causal chain。MTP 模块包含 shared embedding layer、shared output head、独立 Transformer block TRM_1 和 projection matrix M_1。训练时 λ=0.3 (first 10T) → 0.1 (last 4.8T)。推理时 MTP 模块可丢弃（正常推理）或用于 speculative decoding（第二 token 接受率 85-90%，1.8x TPS 加速）。稠化训练信号，提升数据效率；MTP 消融实验：small MoE (15.7B) 和 large MoE (228.7B) 上均一致提升 benchmark 性能。

  **(3) FP8 Mixed Precision Training**：fine-grained quantization（activation: 1×128 tile-wise, weight: 128×128 block-wise），E4M3 for all tensors, online quantization (per-tile max)。CUDA Core FP32 promotion：每 N_c=128 个 WGMMA 结果拷贝到 CUDA Core 做完整 FP32 累积+dequantization。BF16 optimizer states（first/second moments 不用 FP32），activation cached in FP8 (E5M6 for attention inputs, FP8 for SwiGLU inputs)。低精度通信：MoE up-projection 前将 activation 量化为 FP8 再 dispatch。BF16 → FP8 训练 loss error <0.25%，训练速度理论加倍，GPU 内存显著减少。

  **(4) DualPipe Pipeline Parallelism**：双向流水线调度，将每个 chunk 拆分为 attention/dispatch/MLP/combine 四组件，后向再拆分 backward for input 和 backward for weights。通过手动调整 SM 比例实现 all-to-all 和 PP 通信与计算的完全重叠。Bubble = (PP-1)/(PP)*(F&B-3W)/(F+B-W)，比 1F1B 和 ZB1P 更小。仅需 pipeline stages 和 micro-batches 可被 2 整除。支持跨节点 fine-grained experts 而通信开销近零。

  **(5) Custom Cross-Node All-to-All Kernels**：warp specialization + 20 SMs/10 channels。IB send → IB-to-NVLink forward → NVLink receive 流水线处理 dispatching；NVLink send → NVLink-to-IB forward+accumulate → IB receive+accumulate 处理 combining。PTX 定制指令 + auto-tuned chunk size 减小 L2 cache 污染。配合 node-limited routing (M=4, avg 3.2 experts/node)，仅 20/132 SMs 即可跑满 IB+NVLink 带宽。

  **(6) Inference Deployment Strategy**：Prefill-Decoding 分离部署（prefill: 4 nodes/32 GPUs, TP4+SP+DP8+EP32；decode: 40 nodes/320 GPUs, TP4+SP+DP80+EP320）。冗余专家部署：每 10 分钟检测高负载 expert 并复制，prefill 阶段 32 冗余 expert，decode 阶段 64 GPU 承载冗余+共享 expert。Micro-batch 双流水线重叠（prefill: 重叠两 micro-batch 的 attention/MoE 和 dispatch/combine；decode: 重叠 attention 和 dispatch+MoE+combine）。正在探索 dynamic redundancy（每 GPU 16 experts，仅激活 9 个，运行时全局最优路由计算）。

  **论文方法全栈执行例子（以 DeepSeek-V3, 671B total/37B activated, 跨节点训练一个 forward-backward chunk pair 为例）**：
  - **算法层**: MLA (d_c=512, d_c'=1536) + DeepSeekMoE (1 shared + 256 routed, K_r=8, M=4 nodes, Sigmoid gating, bias-based aux-loss-free routing, no token dropping) + MTP (D=1, λ=0.3, shared emb/head + independent TRM_1)。FP8 E4M3 fine-grained quantized GEMM。
  - **系统框架层**: HAI-LLM framework。16-way PP + 64-way EP (8 nodes) + ZeRO-1 DP。DualPipe 双向调度：正向 chunk [Attn|Dispatch|MLP|Combine]+PP_Comm 与反向 chunk [Attn_BW_In|Attn_BW_W|Disp_BW|MLP_BW_In|MLP_BW_W|Comb_BW]+PP_Comm 重叠。RMSNorm + MLA up-projection recomputation。EMA 异步在 CPU 更新。Shared Embedding+Output Head 部署在首尾相同 PP rank。
  - **编译框架层**: 论文未明确说明（自研 HAI-LLM 为内部训练框架，非开源编译框架）。
  - **Kernel调度层**: Cross-node all-to-all kernel — warp specialization, 20 SMs/10 channels, IB+NVLink fully overlapped pipeline。FP8 GEMM — WGMMA on Tensor Core, N_c=128 interval CUDA Core FP32 promotion, per-group scaling dequantization fused。PTX instruction + auto-tuned chunk size + L2 cache interference minimization。
  - **硬件架构层**: H800 GPU, 132 SMs, Tensor Core (14-bit accumulation hardware limit), NVLink 160 GB/s + NVSwitch intra-node, IB 50 GB/s inter-node。FP8 activation caching (E5M6 for attn inputs), BF16 optimizer states, FP32 master weights+gradients。Cost: 180K H800 GPU hours/trillion tokens → 14.8T tokens = 2.664M GPU hours ($5.328M) pre-training。
