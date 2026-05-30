## GEMM+AR / GEMM+RS / GEMM+A2A Communication Patterns

术语是什么？

GEMM+AR、GEMM+RS、GEMM+A2A 是分布式模型训练和推理中三种典型的"通用矩阵乘法 + 数据依赖的集合通信"组合模式。AR = AllReduce，RS = ReduceScatter，A2A = All-to-All。这些 pattern 广泛存在于 multi-GPU 并行策略中，且通信开销往往是端到端延迟的主要瓶颈（FlashOverlap profiling 显示 GEMM+AR 在 LLM serving 中占 31.6-42.2%、GEMM+RS 在 Llama2-7B training 中占 ~30%、GEMM+A2A 在 Mixtral-8x7B training 中占 >40%）。

从算法pipeline角度拆解术语：

三种 pattern 在模型训练/推理 pipeline 中的位置：

```
(1) GEMM+AR (AllReduce):
发生位置: Tensor Parallelism (TP) 和 Data Parallelism (DP)
流程:
  TP: 每个 GPU 计算 GEMM 的部分结果 (partial sum)
      → AllReduce 将所有 GPU 的 partial sum 求和并广播
      → 每个 GPU 获得完整结果
  DP: 每个 GPU 独立计算 gradient
      → AllReduce 求和所有 GPU 的 gradients
      → 每个 GPU 获得平均 gradient

典型场景: Llama3-70B TP=8, attention proj + FFN 后的 AllReduce
通信量: 2×(N-1)/N × data_size (Ring AllReduce)

(2) GEMM+RS (ReduceScatter):
发生位置: TP training (AllReduce 分解为 RS+AG) + FSDP backward
流程:
  TP training: GEMM partial results → ReduceScatter (沿 row 维 reduce 并 scatter)
              → AllGather (聚合完整结果)
  FSDP: weight gradient GEMM → ReduceScatter → 每个 GPU 持有部分 reduced gradient

典型场景: Llama2-7B FSDP training
通信量: (N-1)/N × data_size

(3) GEMM+A2A (All-to-All):
发生位置: Expert Parallelism (EP) in MoE models
流程:
  每个 GPU 计算其 local experts 的 FFN (GEMM)
  → All-to-All: 每个 GPU 将其计算的 token 发送到 token 原始 GPU
  → 每个 GPU 接收来自所有 GPU 的 token，形成完整 batch

典型场景: Mixtral-8x7B EP=4
特点: 动态 routing 导致 GPU 间 workload imbalance → 通信开销加剧
```

**Annotations**: FlashOverlap 的 GEMM+AR 加速在 RTX 4090 上达 1.02-1.65×、A800 上达 1.30×。GEMM+RS 在 A800 pairwise NVLink 上加速 1.07-1.31×。GEMM+A2A 在 MoE 场景因 workload imbalance 需要 predictor 取所有 GPU max 延迟。

术语一般如何实现？如何使用？

三种 pattern 均通过 NCCL 集合通信 API 实现。FlashOverlap 通过统一的 signaling + reordering 机制支持全部三种 pattern——仅 reordering 粒度不同（tile/subtile/subtoken level），signaling 和 counting table 机制完全复用。在 PyTorch 分布式训练中，TP 使用 `torch.distributed.all_reduce`、FSDP 使用 `torch.distributed.reduce_scatter`、MoE 使用 `torch.distributed.all_to_all`。FlashOverlap 替换这些调用为带 overlap 的实现。

涉及论文标题：
- Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering
