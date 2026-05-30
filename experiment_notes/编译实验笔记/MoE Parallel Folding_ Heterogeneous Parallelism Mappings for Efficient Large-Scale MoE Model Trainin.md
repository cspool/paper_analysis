## MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core

- 属于编译框架的实现是什么？实验比较什么？
  实现是 MoE Parallel Folding 策略和灵活 token dispatcher，在 NVIDIA Megatron-Core 框架中修改并行组初始化逻辑，将 Attention 层和 MoE 层的并行映射解耦：Attention 层使用 TP×CP×DP×PP 四维并行组，MoE 层使用 TP×EP×DP×PP 四维并行组（其中 EP 替代 CP，且 TP 在 MoE 层称为 ETP）。仅约束 Attention 和 MoE 层的 PP group 数量和成员保持一致。同时实现支持 token-dropping 和 token-dropless 两种训练范式的统一 token dispatcher，采用 sub-sequence dropping 策略替代 full-sequence dropping 以减少通信开销。实验比较四种 baseline 方法（FSDP、FSDP+EP、TP+EP+DP、MCore 5D-parallelism）在 Mixtral-8x22B、Llama3-8x70B、Qwen2-57B-A14B、Mixtral-8x22B-G8T8 四个 MoE 模型上的 MFU 性能，以及 strong scaling（至 1024 GPU）和 context scaling（至 128K tokens）表现。还评估 FP8 训练性能。

- 硬件平台是什么，配置是什么。
  NVIDIA Eos 集群：每节点 8 个 NVIDIA H100 GPU（峰值 BF16 989.5 TFLOPS/GPU），2 个 56 核 Intel Sapphire Rapids CPU。节点内 NVLink 4th Gen（450 GB/s 单向带宽），节点间 InfiniBand（400 Gbps 单向带宽）。最多使用 1024 GPU。软件环境：PyTorch 2.5.0 + CUDA 12.6，BF16 精度。

- 开源编译框架是什么。修改了什么。
  开源框架：NVIDIA Megatron-Core（https://github.com/NVIDIA/Megatron-LM）。修改内容：
  1. 并行组生成逻辑：实现 generate_mappings() 函数，为 Attention 层和 MoE 层分别生成两套独立的并行组。Attention rank 布局为 (attn_dp, pp, cp, tp)，MoE rank 布局为 (moe_dp, pp, ep, tp)。从 Attention 到 MoE 层的转换仅需 reshape 操作（将 sequence/subsequence 展平为 batch of tokens），无额外通信开销。
  2. Token Dispatcher：实现统一的 token dispatcher，在 ETP 和 EP 维度上协调 All-to-All-V（跨 EP 组交换 token）、AllGather-V（ETP 组内广播）、ReduceScatter-V（ETP 组内聚合分发）等集合通信操作。支持 sub-sequence-based token dropping（默认策略），无需跨 rank 收集 logits。
  3. 支持 TP、EP、CP、DP、PP 五维混合并行，且 EP 可以折叠（fold）到 Attention 层的任意子组中。

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  代码已在 Megatron-Core 中开源（https://github.com/NVIDIA/Megatron-LM）。使用方式：

  **并行组生成过程**：
  ```
  world_size = 64; tp = 2; cp = 2; ep = 2; etp = 2; pp = 2
  
  # Attention 组: attn_dp = 64 // 2 // 2 // 2 = 8
  attn_ranks = ranks.reshape(8, 2, 2, 2)  # (attn_dp, pp, cp, tp)
  attention_groups = {
    "TP": rearrange(attn_ranks, "dp pp cp tp -> (dp pp cp) tp"),  # 每 tp 维度一组
    "CP": rearrange(attn_ranks, "dp pp cp tp -> (dp pp tp) cp"),  # 每 cp 维度一组
    "PP": rearrange(attn_ranks, "dp pp cp tp -> (dp cp tp) pp"),  # 每 pp 维度一组
    "DP": rearrange(attn_ranks, "dp pp cp tp -> (pp cp tp) dp"),  # 每 dp 维度一组
  }
  
  # MoE 组: moe_dp = 64 // 2 // 2 // 2 = 8
  moe_ranks = ranks.reshape(8, 2, 2, 2)  # (moe_dp, pp, ep, tp)
  moe_groups = {
    "TP(ETP)": rearrange(moe_ranks, "dp pp ep tp -> (dp pp ep) tp"),
    "EP":       rearrange(moe_ranks, "dp pp ep tp -> (dp pp tp) ep"),
    "PP":       rearrange(moe_ranks, "dp pp ep tp -> (dp ep tp) pp"),
    "DP(EDP)":  rearrange(moe_ranks, "dp pp ep tp -> (pp ep tp) dp"),
  }
  ```

  **一个 Transformer Layer 执行全过程（以 Mixtral 8x22B 为例，TP=2, CP=2, DP=2 → Attention 为 TP2CP2DP2）**：
  1. Attention 层：4 GPU 各自处理一个子序列，通过 TP（列/行切分 MLP projection）和 CP（序列分片）协作完成 Self-Attention。
  2. 转换：Attention 输出 reshape 为 batch of tokens（无通信），进入 MoE 层。
  3. MoE 层（EP8, ETP1, DP1）：Router 计算每个 token 的 expert assignment（sub-sequence dropping，仅基于本地 logits）。Permutation 将同一 expert 的 token 连续排列。All-to-All-V 跨 EP 组分发 token 到对应 expert 所在 GPU。各 GPU 独立计算本地 expert FFN。反向 All-to-All-V 将结果送回原 GPU。Un-permutation 恢复原始 token 顺序。
  4. 下一 Attention 层：MoE 输出 reshape 回序列形式，进入 Attention 层。

  关键作用：通过将 MoE 层的 EP 折叠到 Attention 层的 TP/CP/DP 子组中，使 all-to-all 通信尽可能在节点内高带宽 NVLink（450 GB/s）完成，而非跨节点 InfiniBand（400 Gbps），从而降低通信开销、提升 MFU。
