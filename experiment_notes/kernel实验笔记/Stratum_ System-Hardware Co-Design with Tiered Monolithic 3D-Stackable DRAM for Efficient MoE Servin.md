## Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Stratum 的 kernel 调度/运行时计算涵盖 Expert Processing 和 Attention Processing 两大类的算子映射和执行调度。(1) **Expert Processing 执行流程**（§4.1）：MoE layer 的 3 阶段 GeMM 操作（projection-up GeMM1 + GeMM2 → SiLU/Hadamard → projection-down GeMM3），采用 tensor parallelism 策略——沿不同维度分区矩阵（GeMM1/2 垂直分片，GeMM3 水平分片），避免 expert weight 复制，通过 all-gather 复制输入 token 再每个 PU 独立计算。Executed experts 顺序处理（非并行），所有 PUs 协作处理一个 expert。(2) **优化执行 Pipeline**（Figure 9）：输入 token 分片发送到各 DRAM channel → sub-ring all-gather 重建完整矩阵 → GeMM2 与 activation function evaluation 重叠 → GeMM3 的 reduce-scatter 与下一 expert GeMM1 并行 → weighted-sum 由 special function engines 在 expert 输出就绪后立即执行。(3) **Attention Processing**（§4.2）：利用 head-level parallelism + PU groups 分区执行，每个 PU group 处理多个 attention heads，interleaved Softmax 与其他算子执行。Query×Key + Softmax 与 Attn×Value 在多个 heads 间交错。Key/Value 矩阵沿 sequence length 维度分片，每 PU 独立计算 local max/sum 后仅交换标量进行全局 Softmax 归一化。(4) **On-chip Ring Network 通信调度**：16 PUs 通过双向 ring 互联，支持 all-gather、reduce-scatter、scalar exchange 等 collective 通信原语，ring router 内含 aggregator 实现 in-situ data reduction。
  - 实验比较：(a) Per-layer MoE latency vs hot expert hit rate（Figure 17a）；(b) Overall system throughput vs hot expert hit rate（Figure 17b）；(c) Decoding throughput scaling vs batch size (1-32)，different sequence lengths (256-4096)（Figure 18a）；(d) Throughput per area vs Mono3D DRAM layers (64/256/1024)（Figure 18b）；(e) Expert swap time/energy overhead per benchmark（Table 4）；(f) 与 Duplex 对比——2.2-3.0× throughput, 1.9-2.9× energy improvement。

- 后端平台是什么，配置是什么。
  - Stratum NMP Logic Die Processor：7nm process, 0.7V supply, 121 mm² die area, FP16 arithmetic. 16 PUs, 每 PU: 16 PEs（16×16 MAC tensor core each），64k MAC units total, 1 GHz operating frequency, 128 TFLOPS peak performance. 36 MB on-chip SRAM, aggregated ring bandwidth 2.048 TB/s. Aggregated Mono3D DRAM bandwidth 19.01-34.34 TB/s（8 tiers）. Peak power 43W（logic die only）.
  - xPU：NVIDIA H100 SXM5 HBM3（Stratum-L）、RTX A6000（Stratum-S）。
  - SystemVerilog 实现，Cadence Genus 综合（ASAP7 7nm PDK），post-synthesis 网表仿真。

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研 in-house cycle-level simulator——接受 tensor size、parameter tier assignments、attention head mappings、routed expert IDs 以及各组件 delay/energy 参数作为输入，输出总体执行时间和 component-level 能耗分解。
  - System-level simulator——包含 Request Generator（Poisson arrival）、SLO-Aware Scheduler、Memory/Computation Mapper、Stratum NMP interface。
  - 组件 delay/energy 参数来源：(a) Cadence Genus synthesis reports for area/timing/power；(b) post-synthesis netlist simulation with annotated switching activity for energy；(c) FinCACTI for SRAM modeling (shared memory, psum memory)。
  - GPU baseline 评估使用 vLLM 0.8.1 benchmark throughput mode, GPU energy from NVIDIA-SMI。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未开源 Stratum 仿真器代码。
  - NMP 执行全过程（single MoE layer on Stratum-L, Mixtral 8×7B）：
    ```
    输入：batch tokens X_t [M×K] 从 xPU 发送到 Mono3D DRAM
    xPU 完成 Gating → expert routing IDs → switch Mono3D DRAM to NMP mode
    
    [Step 1] xPU → DRAM data transfer:
      - X_t partitioned into slices → each sent to distinct DRAM channel
      - Sub-ring all-gather: each PU reconstructs full X_t
    
    [Step 2-7] Sequential Expert Processing (tensor-parallel across all 16 PUs):
    For each activated expert e:
      [Step 2] GeMM1: Z_1 = X_t @ W_1[i] (projection-up, matrix partitioned vertically)
               → PE Tensor Cores: 16×16 MAC array, k-tap dot-product engines
               → Data loaded from Mono3D DRAM via hybrid bonding (19-34 TB/s)
      [Step 3] GeMM2: Z_2 = X_t @ W_2[i] (projection-up, parallel with Step 2)
               → Overlapped with Step 4 activation
      [Step 4] Activation: SiLU(Z_1) via Special Function Engine (256-way SIMD)
      [Step 5] Hadamard: SiLU(Z_1) ⊙ Z_2 = X_2  (no inter-PU communication needed)
      [Step 6] GeMM3: Z_3 = X_2 @ W_3[i] (projection-down, matrix partitioned horizontally)
               → Partial sums accumulated in intra-channel reducer tree
      [Step 7] Reduce-Scatter: aggregate Z_3 partial sums across PUs via ring network
               → Overlapped with next expert's GeMM1 (pipeline optimization)
    
    [Step 8-9] Post-Processing:
      [Step 8] Go to next expert (repeat Steps 2-7)
      [Step 9] Weighted Sum: Σ gate_score_e * expert_output_e via Special Function Engine
    
    [Step 10] Write back results to DRAM designated address → exit NMP mode → xPU reads
    
    Performance output:
    - MoE layer latency = Σ expert compute times + max(communication, compute) overlaps
    - Energy = Σ (PE energy + ring network energy + DRAM access energy)
    - Tiering impact: fast tier tRCD=2.29ns vs slow tier tRCD=22.88ns
    ```
  - Attention 执行全过程：
    ```
    输入：xPU 写入 new KV pairs 到 DRAM, queries Q via DRAM channels
    
    [PU Group Assignment]:
      - 8 attention heads → 4 PU groups × 2 heads/group
      - Each PU group: neighboring PUs on ring topology
    
    [Per-Head Execution (within PU group)]:
      Step A: Sub-ring all-gather Q (replicate to all PUs in group)
      Step B: S = Q @ K^T (partitioned along sequence length dim)
              → Each PU computes local S slice → scalar exchange local max/sum
              → Global softmax normalization
      Step C: O = softmax(S) @ V (partitioned along sequence length dim)
      Step D: Reduce-scatter aggregated O across PUs in group
    
    [Interleaved Pipeline (2 heads per group)]:
      Head1: Q@K → Softmax(steps 1,2,3 with inter-PU scalar comm) → Attn@V
           while Head1 Softmax: Head2 Q@K overlaps
           while Head2 Attn@V: Head1 next decode step
    
    Key optimization: Softmax decomposed into 3 steps with 2 rounds inter-PU comm,
    interleaved with other head's MatMul for latency hiding.
    ```
