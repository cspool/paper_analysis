## LongCat-Flash Technical Report

- 属于Serving调度的实现是什么？实验比较什么？
  - LongCat-Flash 推理系统提出多项 Serving 调度创新：
    1. **Single Batch Overlap (SBO) 四阶段 Pipeline**：利用 ScMoE 架构在单个 batch 内实现 module-level computation-communication overlap。Stage 1: MLA_0 单独执行（作为后续阶段的输入）；Stage 2: All-to-All Dispatch 与 Dense FFN + Attn_0 (QKV Projection) 重叠；Stage 3: MoE GEMM 独立执行；Stage 4: Attn_1 (Core Attention + Output Projection) + Dense FFN 与 All-to-All Combine 重叠。SBO 区别于 DeepSeek-V3 的 TBO（需要两个 batch 做重叠），在单 batch 内即可隐藏通信。
    2. **PD-Disaggregated 架构**：Prefill 和 Decode 分离部署，层级别 KV cache 传输减少 TTFT。最小部署单元为 2 nodes × 16 H800-80GB GPUs。
    3. **MTP Speculative Decoding**：单一 dense layer 的 MTP 作为 draft model，接受率约 90%。采用 C2T (Classifier-based Tree Construction) 过滤低接受概率 token 减少 verification 延迟。MTP 选择 dense layer（1.41% params, 92.1% accept rate）而非 ScMoE layer 以优化 draft-to-target cost ratio。
    4. **Multi-Step Overlapped Scheduler**：通过 TVD fusing (Target forward + Verification + Draft forward 融合为单个 CUDA Graph) 减少 kernel launch overhead。进一步引入 multi-step overlapped scheduler，单次 schedule iteration 预启动多个 forward step 的 kernel，隐藏 CPU scheduling 和同步延迟。需动态预分配 KV cache slots，通过数学归纳法保证 KV cache 分配收敛在 [2n, 3n] 范围内（n 为预启动步数）。
    5. **Wide EP Deployment with DeepEP**：修改 DeepEP 和 EPLB 支持 zero-computation experts（zero-comp experts 输出无需通信即可获得），避免传输 identity 结果。EP 可扩展到上千 GPUs 以降低 MoE GEMM 延迟。
    6. **TP Deployment for Dense FFN**：ScMoE 中 Dense FFN intermediate dim 较大（12288），采用 TP2 或 TP4（非 TP8）减少通信开销。Dense FFN 的 intra-node NVLink 通信（all-gather/reduce-scatter）与 MoE 的 inter-node RDMA 通信（all-to-all）通过 GPUDirect RDMA 并发执行。
  - 实验比较：
    - LongCat-Flash vs DeepSeek-V3 在吞吐量和延迟上的对比（Table 6）：在各种配置下（bf16/fp8, 不同 context length），LongCat-Flash TGS（generation throughput per GPU）和 TPS/u（per-user generation speed）均显著高于 DeepSeek-V3。在 5000 avg context, bf16, 128 H800 GPUs 下达到 100.5 TPS/u。
    - 理论 TPOT（Table 7）：LongCat-Flash SBO 理论 TPOT 16ms, DeepSeek-V3 TBO 30ms, Qwen3-235B TBO 26.2ms。理论 cost: LongCat-Flash \$0.09/1M output tokens, DeepSeek-V3 \$0.17, Qwen3-235B \$0.15。
    - 实测 TPOT 约 26ms（batch size 96），达到理论值的 61.5%。
    - 成本：\$0.70 per million output tokens（基于 H800 \$2/hour）。

- 硬件平台是什么，配置是什么。
  - **推理部署集群**：NVIDIA H800-80GB GPUs，NVLink intra-node + RDMA inter-node（GPUDirect RDMA），200Gb/s per accelerator RDMA。典型部署单元：2 nodes × 16 GPUs（Prefill + Decode 各一 node），按需扩展到 128 GPUs。Wide EP deployment 支持上千 GPUs 扩展以降低解码延迟。
  - **通信库**：DeepEP（修改版支持 zero-computation experts），EPLB（修改版）。NVLink Sharp 硬件加速 broadcast (multimem.st) 和 in-switch reduction (multimem.ld_reduce)。

- 开源Serving框架是什么。修改了什么。
  - 论文未明确说明基于哪个开源 Serving 框架（如 vLLM、SGLang 等），以自研推理系统实现。
  - 核心修改/实现：
    1. **SBO Pipeline Scheduler**：四阶段 module-level overlap 调度器，实现单个 batch 内的 computation-communication overlap。非 DeepSeek-V3 的 TBO（需要两个 batch）。
    2. **PD-Disaggregation with Layer-wise KV Transmission**：层级别传输 KV cache（而非等待全部 KV cache 完成后再传输），显著降低高 QPS 下的 TTFT。
    3. **MTP Speculative Decoding Engine**：dense MTP head 作为 draft model + C2T rejection filter + TVD CUDA Graph fusion。
    4. **Multi-Step Overlapped Scheduler**：CPU 端调度器预分配多步 KV cache slots 并批量 launch kernel，隐藏 CPU-GPU 同步延迟。
    5. **DeepEP/EPLB 修改**：支持 zero-computation experts 的路由和负载均衡，zero-comp expert 输出跳过通信直接返回。
    6. **Custom Communication Kernels**：基于 NVLink Sharp PTX 的 all-gather/reduce-scatter kernel，直接使用 multimem 指令，比 NCCL 和 MSCCL++ 更快（4KB-96MB message size 范围内），仅需 4 thread blocks。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：模型权重和 GitHub 仓库开源（https://github.com/meituan-longcat）。推理系统层面（SBO scheduler, multi-step scheduler, custom kernels）论文未明确说明是否全部开源，但引用和使用了 DeepEP、FlashMLA、DeepGEMM 等开源组件并做了修改。
  - **推理全过程（SBO 单 token 解码）**：
    ```
    [请求输入]
    Prompt tokens → Router 计算 → Token-to-Expert Dispatch

    [SBO Pipeline 四阶段 - 以单个 token 为例]
    Stage 1: MLA_0 (单独执行)
      - Input 过 MLA_0 → 产生 q,k,v cache entry + MoE input
    Stage 2 (并行执行):
      - Dense FFN(chunk_a)
      - Attn_0: QKV Projection(chunk_a)
      - All-to-All Dispatch(chunk_b) → 通过 DeepEP RDMA 跨节点发送 token 到 expert 所在 GPU
    Stage 3: MoE GEMM(chunk_b)
      - SwapAB MoE GEMM kernel → 各 GPU 计算分配的 expert
    Stage 4 (并行执行):
      - Attn_1: Core Attention(chunk_a) + Output Projection(chunk_a)
      - Dense FFN(chunk_b)
      - All-to-All Combine(chunk_b) → 通过 DeepEP RDMA 回收 expert 计算结果

    [Speculative Decoding 增强]
    Target model forward → MTP Draft forward → C2T classifier filter →
    Target Verification (TVD fused CUDA Graph)

    [Multi-Step Overlapped Scheduler]
    CPU 预分配 4 步 KV cache slots → 批量 launch 4 个 SBO pipeline 的 CUDA kernels →
    GPU 连续执行不被 CPU 打断
    ```

    **作用**：SBO 将单 batch 内 All-to-All 通信（约 708us dispatch+combine 总计）大部分隐藏于 Dense FFN + Attention 计算（约 264us）中，non-overlapping communication 从 25.3% 降至 8.4%。MTP 以接受率 90% 将有效生成速度提升约 1.8x。Multi-step scheduler 消除 CPU launch 瓶颈。最终实现 100+ TPS/user，\$0.70/1M output tokens。
