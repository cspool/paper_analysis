## Fine-grained Pipelined Dataflow

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fine-grained Pipelined Dataflow（细粒度流水线数据流）是 HLX 论文提出的核心设计理念。与传统 GPU 上 block-level 同步计算（如 FA-2 的 block-level fusion 和 FA-3 的 2-stage warp-specialized pipeline）不同，细粒度流水线将计算分解为更小的粒度（如 PipeFlash 的 2 行 Q 而非整个 block），形成多级流水线（4 级 for PipeFlash, 3 级 for PipeSSD），使非 MatMul 操作的计算延迟被 MatMul 操作完全隐藏。同时，小粒度意味着更少的中间数据（PipeFlash 的 score/probability 矩阵从 128KB 降为 1KB），从而有效避免 GPU 上的 register spilling 问题。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

细粒度流水线的核心机制——流水线阶段平衡（Pipeline Stage Balancing）：

以 PipeFlash 为例，4 级流水线阶段及其资源需求：

| Stage | Engine | Operation | Compute Pattern | Cycles/Row |
|-------|--------|-----------|-----------------|------------|
| 0 | DPE#0 | $QK^T$ | MatMul (compute-bound) | $\lceil d_{head}/DPU_{size}\rceil \times \lceil (d_{head} \times block_{size})/DPE_{size}\rceil$ |
| 1 | RVPE | Local Softmax | Element-wise + exp (light) | O(1) per element |
| 2 | DPE#1 | PV | MatMul (compute-bound) | same as Stage 0 (when $d_{head}=block_{size}$) |
| 3 | UpE | Update O | Element-wise mul+add (light) | O(1) per element |

平衡策略：以 bottleneck（DPE MatMul）为准，通过控制每级处理的行数实现平衡。当 $block_{size}=d_{head}$ 时，Stage 0 和 Stage 2 的计算量相等，每行处理时间相同，pipeline 达近 100% utilization。当 $block_{size} \neq d_{head}$ 时，通过调整每级处理行数比例（如 Stage 0 处理 1 行时 Stage 2 处理 $\lceil block_{size}/d_{head}\rceil$ 行）最小化 inefficiency，utilisation 变化 <2%。

与 GPU warp-specialized pipeline（如 FA-3）的对比：

```
# FA-3: 2-stage pipeline (Hopper warp specialization)
# Stage 0 (producer warps): TMA load Q,K,V tiles
# Stage 1 (consumer warps): MatMul + Softmax
# Issue: register pressure (2x intermediate data), SIMT constraints

# HLX: N-stage fine-grained pipeline
# Stages: DPE#0 → RVPE → DPE#1 → UpE
# Issue: dedicated engines, no register sharing, direct forwarding
```

FA-3 的 2-stage pipeline 因每个阶段需要独立 warp 的完整寄存器（256KB per SM）导致双倍中间数据，恶化 register pressure。HLX 的细粒度流水线使用专用引擎（每个引擎有固定资源），数据通过 NoC 直接转发而非通过寄存器文件中转，消除了这一瓶颈。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现细粒度流水线数据流的关键硬件要求：(1) 异构专用引擎（而非统一 SIMT/SIMD 单元），每个引擎针对特定操作优化（MatMul、向量、SFU、更新）；(2) 引擎间直接数据转发路径（NoC 或专用总线），避免经过全局内存中转；(3) 灵活的流水线控制逻辑，支持根据操作维度（$d_{head}$, $block_{size}$, $d_{state}$）动态调整各阶段处理行数。GPU 因 SIMT 执行模型、有限寄存器资源和 TMA 的粗粒度特性，难以高效支持细粒度流水线。HLX 通过 URSC 的 DPE→RVPE→DPE→UpE 四级流水线和 mux/demux 数据路由实现了该数据流。

涉及论文标题：
- HLX: A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models
