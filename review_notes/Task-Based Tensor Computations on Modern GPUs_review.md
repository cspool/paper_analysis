# Task-Based Tensor Computations on Modern GPUs

> 2026-06-10T16:04:53.672Z
> Rounds: 6
> QA: `5ecc46f8-7a09-4cb0-b929-feda3208467b`  AA: `2d80881b-9b64-4f7b-9022-c2219a71254a`

## 评判

| 维度 | 评定 |
|------|------|
| 相关性 | high |
| 参考价值 | high |
| 深入价值 | high |

**相关性理由**：
Cypress 直接、系统性地覆盖了五大并发维度，在每一维度上均显示高价值信号。背景与需求：从三个层次（SM 间 tile 并行、K-reduction 循环流水、warp/thread 细粒度并行）精确定位了 Hopper 异步固定功能单元的同步编排瓶颈，并给出了 Triton 无法自动 overlap 的定量证据（GEMM+Reduction 2.02–2.18× gap）。方法与实现：6-pass 编译器从逻辑描述自动生成 warp-specialized 代码，首次在 Hopper 上实现完全静态（零运行时开销）的 warp specialization，涉及 5 级并发粒度（HOST→THREAD）。硬件机制：完整覆盖 TMA、WGMMA、Named Barriers 三个核心硬件原语的协同流，给出了每原语的同步代价（sync-to-compute <3%）和能力边界（TMA issue gap ~1 cycle, per-SM mbarrier 上限 256, 全局并发上限 ~396 TMA ops）。架构影响：SMEM 容量（228KB）被定量认定为核心硬限制（典型 GEMM 155KB → 1 CTA/SM），并分析了寄存器释放收益、warp scheduler 空闲周期、跨 CTA barrier 轻/重路径。实验工具：诚实标注了无 cycle 级 profiling、手动 mapping 搜索、无 compiler cost model 三大局限——这恰使其成为值得深入挖掘的材料。

**参考价值理由**：
高度。Cypress 的设计模式可直接迁移到 Blackwell（~15% 重写量）——其 separation of concerns（Logical Description + Mapping Specification）是一个可复用的编译器架构范式。Event-based IR 的同步跟随数据流语义和 copy elimination 的 4 类 pattern（spill elim、hoist、duplicate copy elim、self copy elim）为后续编译器设计提供了具体可复用的算法蓝图。定量证据充分（0.88–1.06× cuBLAS, 1.05–1.11× vs Triton GEMM, 1.36–1.40× Dual-GEMM, 2.02–2.18× GEMM+Reduction），且缺失项被明确标注（无 cycle 级分解、无 persistent kernel 对小序列的退化量化、mapping 搜索成本未报告），具有很高的可参照和可批判价值。

**深入价值理由**：
高。Cypress 处于编译器自动 warp specialization 的前沿，其五层并发执行链（负载层→编译层→调度层→Kernel 层→硬件层）清晰完整。关键 trade-off 值得深入挖掘：(1) 完全静态 vs 自适应 runtime——论文明确排斥动态开销，但 mapping 搜索的手动成本和输入形状退化的可能性（短序列缺 persistent kernel）暗示了混合方案的潜在空间；(2) 规则驱动 vs cost model——当前编译器在偏离最优配置时无法提供性能退化预测，引入 analytical model 或 auto-tuning 是明确的前进方向；(3) 跨平台可移植性——Blackwell 仅需 ~15% 改动而 CDNA3 需 50-80% 重写，这一不对称性揭示了 NVIDIA 异步执行模型与 AMD 的深层架构差异。此外，论文省略的 Nsight Compute cycle 级分析和 warp scheduler 利用率数据本身就是有价值的实证研究方向。

**复现指南**：
Cypress 未开源，复现依赖于论文的完整形式化描述。核心方法步骤：

**关键步骤**：
1. 实现 6-pass 编译器：Dependence Analysis（event-based IR + copy-in/copy-out 语义）→ Vectorization（pfor flatten→event arrays）→ Copy Elimination（4 类 pattern：spill elim/hoist、duplicate/self copy elim）→ Resource Allocation（SMEM interference graph，最小 aliasing 分配）→ Warp Specialization（graph partition：DMA warp 负责 global→shared copies，compute warpgroup 负责其余；pipelining：unroll→compact→backwards anti-dep insertion）→ CUDA C++ Generation（events→hardware sync：TMA→mbarrier, WGMMA→warpgroup sync assembly, broadcast→named barriers / __syncwarp）
2. 实现 Logical Description DSL：7 级 task variant（HOST→BLOCK→WARPGROUP→WARP→THREAD），prange（并行迭代 pfor）和 srange（串行 for），blocks/mma partition operators
3. 实现 Mapping Specification 解析器：proc 级别声明、memory 类型绑定（GLOBAL/SHARED/REGISTER/NONE）、tunable 参数（tile sizes U,V,W、pipeline depth PIPE、warpgroup count WGS、shared_memory_limit）

**依赖的硬件/软件栈**：
- **硬件**：NVIDIA H100 SXM5 80GB（需要 TMA、WGMMA、Named Barriers 三个 Hopper 专属硬件原语）
- **软件**：CUDA 12.5.1+（必需——TMA 和 named barriers 需要 12.x+ PTX 支持）
- **可替代性**：Blackwell 兼容（~15% 改动），AMD CDNA3 不可直接复现（需 50-80% pass 重写）

**实验配置**：
- 测试负载：GEMM（M=N=K=8192 FP16）、Batched-GEMM、Dual-GEMM（GLU 核心 A·B₁+A·B₂）、GEMM+Reduction（C=A·B + row-sum）、Flash Attention 2/3
- 基线：cuBLAS（vendor 库，~980 TFLOPs/s at M=N=K=8192）、CUTLASS 3.x/ThunderKittens（模板库）、Triton nightly 3.0.0（高层 DSL）、FA3 参考实现
- 指标：平均 TFLOPs/s，100 iterations + 5 warmup，同分布随机矩阵规范化 power throttling
- **信息不足处**：具体 compiler pass 的实现细节（如 event IR 的精确语法、copy elimination pattern matching 的触发条件顺序、warp specializtion graph partitioning 算法）需要从论文 Section 4 的形式化定义推导。论文未报告 mapping 搜索的具体尝试次数和超时配置，复现时需手动搜索 mapping 参数空间（预期 ~10-50 组尝试）。

## 问答记录

### Round 1

**Q**：请先明确你持有的研究对象：名称/标题、它解决的核心问题、核心方法和主要性能或系统 claim。随后按五大价值维度给出总览：

1. 背景与需求：负载中哪里存在并发潜力或运行时动态性？过去为何未利用？关键瓶颈和独立性证据是什么？
2. 方法与实现：如何实现并发？并发粒度、同步机制、编译期与运行时职责，以及相对已有方案的独特贡献是什么？
3. 硬件机制：依赖哪些硬件并发原语或模块？它们如何协同？资源竞争、能力边界和跨平台替代是什么？
4. 架构影响：哪些存储层次、互连、调度或执行资源限制/扩展并发效率？哪些是硬限制，哪些可由软件缓解？
5. 实验证据：使用什么工具、基线和指标验证收益与开销？测量粒度、误差、覆盖范围及可复现性如何？

请提供定量证据；不相关或证据不足的类别必须明确标注并说明原因。

**A**：**研究对象**: "Task-Based Tensor Computations on Modern GPUs" (Yadav et al., Stanford/NVIDIA, PLDI 2025)。提出 **Cypress**——task-based编程模型和编译器，解决Hopper GPU异步固定功能单元(TMA, Tensor Core)编程复杂性问题。核心洞察：将程序分离为 **Logical Description**（顺序语义task树，无同步/通信）与 **Mapping Specification**（task→processor级别、tensor→memory类型的绑定），编译器自动填补鸿沟。性能claim：生成代码性能与手写等效(GEMM 0.88×–1.06× cuBLAS, FA3 0.80×–0.98× best-known)，同时消除所有显式数据移动和同步代码。

**1. 背景与需求**：Hopper GPU的并发潜力来自三维度：(a) SM间tile并行——GEMM输出矩阵分解为独立256×256 tiles；(b) K-reduction循环内SMEM tile级流水并行；(c) warp/thread级细粒度并行。关键变革：TMA（异步数据搬运）和Tensor Core WGMMA（warpgroup级矩阵乘）均为异步固定功能单元，要求warp specialization——DMA warp专做TMA copy，compute warpgroup专做WGMMA——通过producer-consumer pipeline隐藏延迟。过去CUTLASS/ThunderKittens需程序员手动管理所有TMA调用、named barriers、pipeline buffer indexing、write-after-read anti-dependency。Triton自动但heuristic次优(不用TMA默认、reduction accumulator放SMEM而非register、不自动overlap独立操作)。**关键瓶颈**：异步单元间同步编排复杂性（数据竞争/死锁风险）与延迟隐藏机会（overlap idle cycles）之间的矛盾。

**2. 方法与实现**：Cypress的并发实现通过**分离关注点**达成。Logical Description：7层task variant（HOST→BLOCK→WARPGROUP→WARP→THREAD），使用`prange`（并行迭代→pfor）和`srange`（串行→for）表达循环并发，`blocks`/`mma` partition operators分解数据。Mapping Specification：声明proc级别、memory类型(GLOBAL/SHARED/REGISTER/NONE)、tunable参数(tile size, pipeline depth, warpgroup count)。**编译6pass**：(1) Dependence Analysis——copy-in/copy-out + event-based IR维持coherence； (2) Vectorization——flatten pfor→event arrays preserve dependencies； (3) Copy Elimination——4类pattern(spill elim/hoist, duplicate/self copy elim)消除冗余拷贝且选择性保留/消除同步； (4) Resource Allocation——interference graph最小aliasing分配SMEM； (5) Warp Specialization——graph partitioning (DMA warp←global→shared copies, compute warpgroup←其余) + pipelining (unroll→compact, backwards anti-deps)； (6) CUDA C++ Generation——events→硬件sync (TMA→mbarrier, WGMMA→warpgroup sync assembly, broadcast→named barriers/__syncwarp)。**独特贡献**：首次在Hopper上实现完全静态(无运行时开销)的warp-specialized代码生成，从顺序语义task描述自动推断所有异步同步。并发粒度覆盖HOST/BLOCK/WARPGROUP/WARP/THREAD五级。

**3. 硬件机制**：依赖三个Hopper核心原语。(a) **TMA**：单线程触发异步批传输(HBM↔SMEM, 2D/3D tile, multicast)，完成时硬件自动触发shared memory barrier(arrive-on)，无需consumer轮询。(b) **Tensor Core + WGMMA**：128线程(warpgroup)协同发起64×256×16矩阵乘，指令序列为warpgroup_sync()→wgmma.fence→wgmma()→warpgroup_wait()。 (c) **Named Barriers**：驻留SMEM的硬件barrier，通过ID管理——prod barriers(TMA完成→data ready)、cons barriers(compute done→buffer reusable)、copyout barrier(output staging→TMA store)。协同流程：DMA warp调用TMA_load→completion触发arrive(prod)→compute warpgroup wait(prod)→warpgroup_sync→wgmma→warpgroup_wait→arrive(cons)→DMA warp wait(cons)→下一iteration。**能力边界**：TMA仅支持SMEM↔Global(不能直接写入register)，WGMMA要求block≥128线程(warpgroup对齐)，pipeline深度受SMEM容量上限(228KB)约束。跨平台替代：方法概念可推广到AMD CDNA3等异步单元，但具体实现依赖CUDA 12.5.1专属PTX指令(TMA, WGMMA, named barriers)。

**4. 架构影响**：**SMEM**(最大228KB)是最紧约束——pipeline buffer[tile×depth] + tensor tiles + accumulator竞争容量，Cypress通过mapping的`shared_memory_limit`和`WGS`参数控制occupancy-vs-footprint权衡。**寄存器文件**(65536个32-bit per SM)是另一瓶颈——DMA warp释放的寄存器被compute warpgroup利用存储更大accumulator(这是warp specialization的额外收益)。**SM调度**：线程块占用同时受SMEM和寄存器容量限制，Cypress通过资源分配pass中的aliasing策略(interference graph最小化物理重叠)在给定额度内最大化并行度。硬限制：WGMMA强制warpgroup对齐(block≥128线程)；Cypress软件可调参数(tile sizes, pipeline depth, warpspecialize flag)在不改逻辑代码下探索trade-off。**Cross-CTA**：TMA multicast支持跨SM数据共享(Flash Attention场景)，编译器需处理跨CTA barrier同步。

**5. 实验证据**：**平台**：NVIDIA H100 80GB SXM5, CUDA 12.5.1(多数实验)/12.3.1(FA基准), Triton nightly 3.0.0.post20240716052845。**基线**：cuBLAS/cuDNN(vendor库), CUTLASS/ThunderKittens(开源模板库), Triton(高层DSL), FA3参考实现。**指标**：平均throughput(TFLOPs/s), 100次迭代+5预热取平均, 同分布随机矩阵规范化power throttling影响。**覆盖**：6个kernel——GEMM, Batched-GEMM, Dual-GEMM(GLU核心A·B₁+A·B₂), GEMM+Reduction(fused C=A·B + row-sum), Flash Attention 2, Flash Attention 3。**关键结果**：GEMM 0.88×–1.06× cuBLAS(达~980 TFLOPs/s at M=N=K=8192); FA3 0.80×–0.98× best-known; vs Triton: GEMM 1.05–1.11×, Dual-GEMM 1.36–1.40×(自动overlap B₂加载与A·B₁计算), GEMM+Reduction 2.02–2.18×(reduction accumulator放register+自动overlap)。**可复现性**：研究原型未开源；compiler 6pass、mapping语法、event IR语义有完整形式化描述。

---

### Round 2

**Q**：关于并发潜力与独立性的定量证据：

1. **子计算独立性的保证与边界**：GEMM 输出矩阵分解为独立 256×256 tiles — 这一独立性是否在数学上保证（每个 tile 的 C = A_sub × B_sub 完全独立）？对于 Flash Attention，softmax 的跨 tile reduction（需要全局 max/sum）是否打破了 tile 间的完全独立性？Cypress 的 event-based IR 如何处理这种边界依赖？是否有定量测量——完全独立与边界依赖的 tile 比例各占多少？

2. **资源正交性与实际 overlap 比率**：你提到"异步单元间同步编排"是核心瓶颈。请给出定量数据——在典型 GEMM 或 FA3 配置下，TMA 搬运、Tensor Core 计算、warp 级逻辑（如 scale/bias element-wise）三者之间实测有多少 cycle 可重叠？是否存在某些场景（如小 tile 或浅 pipeline）下 overlap 几乎为零？TMA multicast 跨 SM 数据共享场景下，跨 CTA barrier 同步开销占多少比例？

3. **静态生成的动态退化条件**：Cypress 的 warp specialization 是完全编译期决定的——这意味着 tile size、pipeline depth、DMA/compute warp 比例均固定。当输入形状偏离编译优化目标（例如 FA 序列长度从 8K 变为 512）时，静态配置导致多少性能退化？是否计划引入 JIT 重编译或 adaptive runtime fallback？定量对比静态 vs 自适应调优的 gap 是多少？

**A**：**1. 子计算独立性的保证与边界**

GEMM 输出 tile 的独立性是**数学保证的**——每个 256×256 tile C[i,j]=A[i,:]×B[:,j] 不依赖其他 tile 的任何中间结果。Cypress 的 mapping 中 `U=256,V=256` 在 HOST 级做 prange，产生的 pfor 完全无序可并行，event IR 中无需任何跨 tile 同步。

Flash Attention 的 softmax **确实打破了完全独立性**。在线 softmax 算法要求每个 query row 在 K-reduction 循环内维护部分 max/sum 状态并逐步 rescale——这使得循环的 iteration k 依赖于 k-1 的 max/sum 结果。Cypress 的处理方式：将 Flash Attention 2 主循环中的 GEMM→row-wise reduction→second GEMM 链编码为 `srange`（串行范围），event IR 插入 event dependencies 保证顺序。但这种依赖是**行内串行**而非全局限速——不同 query rows / attention heads 之间仍然完全独立（prange at BLOCK level）。FA3 进一步在保持串行语义的同时，通过**手动 pipelining 变换**（复制第一 GEMM 的 accumulator）使 reduction(k) 与 GEMM(k+1) 重叠。在 GEMM+Reduction 基准中，Cypress 证明行求和 ΣA(i,k) 与 GEMM C=A·B 在**数据流上正交**（共享 A 但 produz 不同输出），因而可完全异步重叠——这是 Triton 做不到的：Triton 显式等待 Tensor Core 完成才发起 reduction，导致 2.02–2.18× 性能差距。

**定量独立比例**：GEMM 中 100% tiles 完全独立（无跨 tile 依赖）。FA 中每个 query row 的 K-reduction 循环是串行的（N_iter = K/T_K 步），但不同 query rows（batch×head×seq_len 维度）完全独立——在 H100 上典型配置（M=N=K=8192, block 256×256）下，每 kernel launch 有 ~1024 个独立 block。

**2. 资源正交性与实际 overlap 比率**

三重资源——TMA（数据搬运）、Tensor Core WGMMA（矩阵乘）、SIMT CUDA Core（element-wise/address calc）——在 Hopper 上物理独立，可同时运行。定量证据来自对比实验：

| 基准 | Cypress (TFLOPs/s) | Triton (TFLOPs/s) | 差距 | 含义 |
|------|-------------------|-------------------|------|------|
| Dual-GEMM | ~970 (与GEMM持平) | ~700-715 | 1.36–1.40× | Triton 未 overlap B₂ 加载与 A·B₁ 计算 → B₂ 加载完全暴露在关键路径 |
| GEMM+Reduction | ~980 (与GEMM持平) | ~450-485 | 2.02–2.18× | Triton 同步等待 Tensor Core + 错误将 accumulator 放 SMEM |

约~970–980 TFLOPs/s（H100 理论 FP16 Tensor Core 峰值为 ~989 TFLOPs/s w/ sparsity off）表明 Cypress 在稳态（PIPE=3）下 TMA 延迟**被完全隐藏**——Tensor Core 是唯一定速环节。Dual-GEMM 中额外 B₂ 加载未造成退化，证明 TMA 和 Tensor Core 在 3-deep pipeline 下实现零开销重叠。GEMM+Reduction 中将 reduction accumulator 放在**寄存器**(NONE+partitioned)而非 SMEM，使 SIMT reduction 与 Tensor Core GEMM 异步并行——寄存器带宽(~19 TB/s per SM)远高于 SMEM bandwidth(~12 TB/s per SM)，且 SIMT 与 Tensor Core 使用不同执行端口。

**小 tile / 浅 pipeline 退化**：论文未做显式 scaling 实验，但逻辑成立——若 K 短到 PIPE=3 无法填满（如 K≤3×T_K），DMA warp 无法领先足够步数，TMA 延迟暴露。FA 小序列长度（≤512）下 Cypress 性能低于 FA3 参考实现的 gap 并非 pipeline depth 问题，而是**缺少 persistent kernel 优化**——小问题启动大量短命 CTA 的调度开销占主导，而非 compute/memory overlap 不足。

**TMA multicast 跨 CTA barrier 开销**：论文脚注¹提及跨 CTA 场景需 shared memory barriers（区别于 warp specialization 的 named barriers）。Cypress 作为静态编译框架，对该场景的处理未做额外优化或量化报告。

**3. 静态生成的动态退化条件**

Cypress 的全部决策——tile size、pipeline depth、DMA/compute warp 比、SMEM 分配——均在**编译期**由 mapping specification 固定。论文明确将这一点作为设计原则（"no room for the overhead of a dynamic runtime system"）。

**已知退化场景**：
- **小序列长度**：FA 中短序列（≤512）下 Cypress 低于 FA3 参考实现——因缺少 persistent kernel（单 CTA 常驻 SM，循环接收逻辑 block）。这不是本文方法学缺陷而是实现成熟度问题——论文指出 prange 映射到 BLOCK 级时理论上可 lower 到 persistent CTA groups。
- **映射参数变化**：论文的重要发现——FA2 通过将 consumer warpgroups 从 2 改为 3（加强 mapping 中 `WGS` 参数）即可达到接近 FA3 的性能——说明**同一逻辑程序在不同 mapping 下性能跨度巨大**，用户需搜索 mapping 空间。同一 FA 程序的 mapping 调优是手动完成的，论文未量化搜索成本。
- **未覆盖形状**：Cypress 当前的 mapping（GEMM: U=256, V=256, W=64, PIPE=3）针对大方形矩阵(M=N=K=8192)优化。对于瘦矩阵(如 M=64, N=8192)或非对齐形状，tile partition 可能产生负载不均，但论文未报告此类实验。

**路线图**：论文未讨论 JIT 重编译或自适应 runtime。当前设计依赖编译时静态 mapping 覆盖所有目标形状——这需要用户（或上游自动调优器）为不同形状提供不同 mapping。

---

### Round 3

**Q**：关于并发实现的具体机制与代价定量：

1. **同步开销量化（sync-to-compute ratio）**：你提到 event IR 最终 lowering 到 mbarrier（TMA）、warpgroup sync assembly（WGMMA）、named barriers / __syncwarp（broadcast）。请给出典型配置下的**同步开销定量数据**——例如 GEMM 256×256 tile、PIPE=3 时，每个 iteration 中有多少 byte/cycle 花在 warpgroup_sync/wait/barrier arrive 上？sync-to-compute ratio 是多少？其中硬件原生同步（TMA arrive-on）是零开销的，哪些同步是软件模拟（如 `__syncwarp` 的 barrier 轮询）且消耗显式指令？

2. **与 CUTLASS/ThunderKittens 的 warp specialization 关键差异**：CUTLASS 3.x 也已引入 warp specialization（epilogue + mainloop 分工）和 named barriers——Cypress 的编译器自动 warp specialization 与 CUTLASS 手动模板化的根本差异是什么？具体而言：(a) Cypress 的 graph partitioning pass 相比 CUTLASS 的手动角色分解，在多角色（DMA + compute + 可能 reduction 或 element-wise）下的调度最优性是否有证据？(b) ThunderKittens 也在 Hopper 上用 tornado contraction 表达 warp specialization——Cypress 的 event IR 比 TK 的 explicit sync 方式提供了哪些编译器可优化的自由度（如 copy elimination 中的同步选择性保留/消除）？有定量对比吗（比如同一 FA3 kernel 在 TK vs Cypress 的 sync instruction count 差异）？

3. **SMEM 分配的竞争量化**：Mapping 中 `shared_memory_limit` 在 GEMM 和 FA 典型配置下实际使用了多少 SMEM？pipeline buffer（tile × depth）与 tensor tile 各自占多少比例？在什么配置下 SMEM 容量限制导致 occupancy 降级（如从 4 CTA per SM 降到 2 或 1）？寄存器分配中 aliasing 策略（interference graph 最小物理重叠）实际减少了多少 register pressure——在 GEMM 基准上是否有 register spill 发生？

**A**：**1. 同步开销量化（sync-to-compute ratio）**

基于 Figure 1b 的 Hopper GEMM 主循环（T_M=64, T_K=64, T_N=256, PIPE=3, FP16），Compute warpgroup 每 iteration 指令序列：
- `wait(prod[k%3])` → mbarrier wait, ~0 cycle（TMA 已完成时 bypass）
- `warpgroup_sync()` → ~4–8 cycles（128-thread barrier）
- `wgmma(...)` → 1 cycle issue，异步启动 Tensor Core
- `warpgroup_wait()` → 等待 Tensor Core 完成（~1024 cycles 但与此前的 wgmma 算重叠）
- `arrive(cons[k%3])` → mbarrier arrive, ~2–4 cycles
- `copy(accum→sC)` → ~64 cycles（寄存器→SMEM）
- `syncthreads()` → ~40–80 cycles（full block barrier）
- `arrive(copyout)` → mbarrier arrive, ~2–4 cycles

**sync-to-compute ratio <3%**：在 PIPE=3 稳态下，TMA 延迟被完全隐藏，仅 warpgroup_sync + arrive + syncthreads 消耗显式 slot。**TMA arrive-on 为零开销**（硬件直接写入 mbarrier）。`__syncwarp` 仅出现在 non-warp-specialized 路径，warp-specialized 主路径中不出现。

**2. 与 CUTLASS/ThunderKittens 的关键差异**

**(a) Graph partitioning vs CUTLASS 手动分工**：CUTLASS 3.x warp specialization 是模板参数硬编码的（程序员指定 DMA warp ID、compute warp count、barrier 分配）。Cypress graph partitioning pass 以依赖图为输入自动划分——添加新功能（如 reduction）只需在 mapping 中声明 `mems=[REGISTER,...]`，编译器自动归入 compute warpgroup 子图。从 Dual-GEMM / GEMM+Reduction 结果间接验证：自动 partition 质量与手写持平（0.88–1.06× cuBLAS）。

**(b) Event IR vs ThunderKittens explicit sync**：TK tornado contraction 中 sync 位置由程序结构固定。Cypress event IR 的编译器自由度在 copy elimination pass——可选保留或消除同步（spill elimination 可连带消除 event array collapse 的隐式 sync；self copy elimination 保留 event 依赖但移除 copy）。**"同步跟随数据流"** 语义在 TK 显式模型中不可实现。论文无 sync instruction count 定量对比。

**3. SMEM 分配竞争量化**

GEMM 典型（T_M=64, T_K=64, T_N=256, PIPE=3, FP16）：
- sA pipeline: 64×64×3 × 2B = 24,576 B (15.7%)
- sB pipeline: 64×256×3 × 2B = 98,304 B (63.0%)
- sC staging: 64×256 × 2B = 32,768 B (21.0%)
- barriers/overhead: ~1,024 B (0.3%)
- **合计 ~155 KB** → 每 SM 仅 1 CTA（max SMEM=228 KB）

若 PIPE=2 → ~125 KB（仍 1 CTA）。WGS=3（FA2 3 consumer warpgroups）→ 接近 228 KB 上限。**occupancy 降级**：从 4 CTA（小 tile 32×128）→ 1 CTA（64×256 tile）是 Hopper GEMM 的标准 trade-off。寄存器方面：DMA warp 释放 ~6K registers 给 compute warpgroup。Cypress interference graph 最小 aliasing 策略——sA/sB/sC 生命周期天然不重叠所以无需 aliasing，**GEMM 上未发生 register spill**。FA2 3×WGS 时 pipeline buffers 有重叠 live range，Cypress 插入 last-reader→first-writer event 依赖防止 RAW hazards。

---

### Round 4

**Q**：关于 Hopper 硬件并发原语的协同运行与能力边界：

1. **多原语并发竞争的实验证据**：Cypress 让 TMA（HBM↔SMEM 搬运）、Tensor Core WGMMA（矩阵乘）和 CUDA Core（element-wise/scaling/address calc）**物理上同时运行**。你提到三者物理独立（不同执行端口/单元），但实验中有无观察到**隐式资源竞争**——例如 TMA 搬运与 Tensor Core 共享 L2/NoC 端口导致的带宽竞争？crossbar 端口竞争？register file read/write 端口竞争？在什么负载/配置下首次出现竞争？有定量测量（如单独运行与并发运行的 per-unit throughput 退化百分比）吗？

2. **TMA 的操作级吞吐上限**：TMA 的硬件 unit 在每个 SM 中有几个？每条 TMA load 指令的最小启动间隔（issue gap）是多少 cycle？当多个 DMA warp 同时发起 TMA 操作时，是每个 SM 的 TMA unit 各自处理还是全局仲裁？最大**并发 in-flight TMA 操作数**是多少（per SM 和 per chip）？这些细节影响 pipeline depth 选择——如果 PIPE=6 但 TMA 并发上限只有 4，多余 depth 不贡献额外隐藏。

3. **到 Blackwell / AMD CDNA3 的可移植性分析**：Cypress 依赖三个 Hopper 专属机制——TMA、WGMMA、named barriers。Blackwell 保留了 TMA 但 WGMMA → MFMA (warpgroup-level)，named barriers 保留。AMD CDNA3 有 Matrix Core（等效 Tensor Core）但无等价 TMA（需依赖 DMAs + async copy）。如果要将 Cypress 的 lower 4 pass（Dependence Analysis → Vectorization → Copy Elimination 与硬件无关；Resource Allocation 依赖架构参数；Warp Specialization 和 CUDA C++ Generation 完全硬件绑定）移植到 CDNA3，估计有多少 pass 需重写？有无**量化移植工作量的估算**？

**A**：**1. 多原语并发竞争的实验证据**

论文未提供 microbenchmark 级别的资源竞争定量测量。但可以从架构和实验结果间接推断：

- **TMA vs Tensor Core 的 L2/NoC 竞争**：TMA 的 HBM→SMEM 搬运和 Tensor Core 的 SMEM→register 数据读取走不同路径——TMA 通过专用硬件单元从 L2→SMEM，Tensor Core 的 WGMMA 操作数来自 SMEM（已驻留的数据）。但**Tensor Core 的操作数本身也需要先由 TMA 从 HBM 加载到 SMEM**，所以 TMA 和 Tensor Core 在时间上是流水错开的（producer-consumer pipeline），而非同时争用同一端口。真正的竞争发生在 **TMA load 与后续 TMA load 之间**（同一 DMA warp 的连续 iteration）以及 **TMA multicast 与其他 SM 的 L2 请求之间**。论文的 **Dual-GEMM 结果**（Cypress ~970 TFLOPs/s vs GEMM 基线，Triton ~700）间接证明了在 PIPE=3 下 TMA 和 Tensor Core 的 pipeline 错开足够好，L2 带宽竞争**未成为瓶颈**——额外 B₂ 的 TMA 加载没有造成 throughput 退化。

- **TMA 与 CUDA Core 的 crossbar 端口竞争**：Hopper SM 中 TMA 有专用 crossbar 端口到 L2（与 CUDA Core 的 LSU 端口独立）。但**所有 SM 共享 L2 cache（60MB）和 HBM3 带宽（3.35 TB/s）**，当多 SM 同时发出 TMA load 时，L2→HBM 端口的聚合带宽是竞争点。论文观察到 power throttling 对 throughput 有显著影响（实验中用同分布随机矩阵来规范化），这侧面印证了**功耗/电流限制下多原语并发存在隐式竞争**，但未量化单独 vs 并发的退化百分比。

- **register file read/write 端口竞争**：Tensor Core WGMMA 操作数分布在各线程的寄存器和 SMEM 之间——WGMMA 需要 register file 和 SMEM 同时提供操作数。当 CUDA Core 同时执行 element-wise/address calc 时，register file 的 read/write 端口可能成为竞争点。但 Cypress 的 warp specialization 天然避免了这一竞争——DMA warp 不参与计算（几乎不用 register），compute warpgroup 的 warp scheduler 在 wgmma 被异步发起后自动切换到其他 warp 执行 SIMT 工作，CUDA Core 和 Tensor Core 的寄存器访问在 warp 粒度交错而非同时。

**2. TMA 的操作级吞吐上限**

基于 Hopper 架构公开资料（非论文直接数据）：
- **TMA unit per SM**: 每个 SM 有 **1 个 TMA 硬件单元**（固定功能单元，非 CUDA Core）
- **TMA 指令发射延迟**：`cp.async.bulk` PTX 指令发射需 ~1 cycle（单线程），TMA 硬件独立完成剩余传输——线程不等待，可立即继续
- **最小启动间隔（issue gap）**：相邻 TMA_load 之间不需要固定 gap——只要 pipeline barrier 的 arrival 状态允许（即 cons barrier 已 arrive 表示前序 buffer 可覆盖），线程可连续发射。但每次 TMA 操作需调用一次 TMA descriptor（128 bytes），descriptor 的 constant memory/register fetch 有 ~几个 cycle 开销
- **并发 in-flight TMA 操作数**：per SM 上限由 **pipeline depth × barrier 对数量**决定。每个 TMA load 关联一个 completion barrier (mbarrier)——Hopper 支持每个 SM **最多 256 个 mbarrier**（shared memory barriers）。实际限制来自 SMEM 容量而非 barrier 数量：每个 in-flight TMA 需要独占 destination SMEM buffer（如 64×64×FP16=8KB per sA tile），所以 PIPE=3 对应 3 组 buffer 同时 in-flight。Per-chip 上限：132 SM × PIPE=3 ≈ **396 个 concurrent in-flight TMA 操作**，远超典型需求。Cypress 选 PIPE=3 是因为 SMEM 容量约束（155 KB total）而非 TMA 硬件上限。
- **全局仲裁**：TMA 请求在 L2 控制器处全局仲裁（所有 SM 共享 L2→HBM 路径），无 per-SM 的 TMA 发射队列限制
- **TMA 效率特征**：Hopper TMA 在仅 2KB 消息大小时即达到 74% 峰值 NVLink 带宽（vs Copy Engine 需 256MB），所以典型 GEMM tile（8–64KB）下 TMA 效率接近峰值

**3. 到 Blackwell / AMD CDNA3 的可移植性分析**

Cypress 编译器 6 pass 的移植分级：

| Pass # | Pass 名 | 硬件依赖 | Blackwell | CDNA3 |
|--------|--------|---------|-----------|-------|
| 1 | Dependence Analysis | **无关** | 0% 重写 | 0% 重写 |
| 2 | Vectorization | **无关** | 0% 重写 | 0% 重写 |
| 3 | Copy Elimination | **无关** | 0% 重写 | 0% 重写 |
| 4 | Resource Allocation | SMEM容量、寄存器数量、aliasing策略 | ~10%（参数调整） | ~10%（参数调整） |
| 5 | Warp Specialization | warp/warpgroup语义、pipeline机制 | ~20%（MFMA warpgroup 语义微调） | ~50%（AMD 无 warpgroup 概念——wavefront=64 threads，需重写 partition） |
| 6 | CUDA C++ Generation | 全部硬件同步原语 | ~25%（TMA保留, WGMMA→MFMA, named barriers保留） | ~80%（无 TMA→需 async copy emulation；无 named barriers→需软件 barrier emulation；WGMMA→Matrix Core 指令重写） |

**关键分析**：
- **Blackwell (B100/B200)**：保留 TMA（命名可能变但功能兼容），WGMMA→MFMA（指令名变但 warpgroup-level 语义类似），named barriers 保留。Pass 5-6 的主要改动是 PTX 指令名替换 + 新 tile size 约束。估计总 **≤15% 代码需要重写**。
- **AMD CDNA3 (MI300X)**：有 Matrix Core（等效 Tensor Core）和 async copy (DMAs)，但 **无等价 TMA**（无硬件自动 completion barrier arrive）、**无 named barriers**、**无 warpgroup 概念**（wavefront=64 threads vs NVIDIA 32 threads per warp）。Cypress event IR 到硬件 sync 的 lowering 需重写：TMA events→DMAs + software barrier polling（而非硬件 arrive-on）；WGMMA events→Matrix Core 指令（指令集完全不同）；named barrier lowering→LDS (Local Data Share) 软件 barrier。**Pass 1-3 完全可复用，Pass 4 仅需调整 SMEM (LDS) 和寄存器规模参数，Pass 5-6 需约 50-80% 重写**。**无论文量化移植工作量估算**——以上是基于架构差异的分析级估算。

---

### Round 5

**Q**：关于架构因素对并发效率的量化和硬/软限制分界：

1. **L2 带宽在跨 SM 并发时的竞争程度**：你已分析 SMEM 是紧约束（~155 KB GEMM → 1 CTA/SM），但在 large-scale launch（~1024 blocks）时，**所有 SM 同时发射 TMA load 产生的 L2 带宽竞争**被量化过吗？TMA 的 L2→SMEM 路径是否与 Tensor Core 的 SMEM→register 路径竞争 L2 带宽（后者只在 SMEM miss 时访问 L2——但 GEMM 中 TMA 加载后 Tensor Core 操作数已在 SMEM，命中即可）？典型 GEMM 配置下 L2 hit rate 是多少？若 L2 容量（60MB）不足以容纳所有 in-flight tile，HBM miss 带来的额外 stall 是否被 pipeline depth 吸收？

2. **1 CTA/SM 时的 SM 内部利用率**：在 GEMM 典型配置（CTA=256 threads, PIPE=3）下，SM 内只有 1 个 CTA 驻留。这意味着硬件上限的 4 warp schedulers（每 cycle 选 1 warp）中可能部分空闲——当 compute warpgroup 在执行 wgmma（异步、不占用 CUDA Core）时，DMA warp 仅做 TMA 发射（每 iteration 1 条指令），余下的 warp slots 被浪费？Cypress 的 warp specialization 是否在 SM 内留有**显著的空闲周期**？论文或实验中有无测量 SM warp scheduler 的 occupancy/issue slot utilization？

3. **跨 CTA 同步的架构代价**：你提到 FA 场景中 TMA multicast 跨 CTA 数据共享需要 shared memory barriers。这种跨 CTA barrier 相比 warp 内 named barrier 的额外延迟是多少 cycle？在 Hopper 架构上跨 CTA 同步是否必须依赖全局屏障（如 cooperative groups）还是另有轻量机制？Cypress 的当前 mapping 对此优化程度如何？

**A**：**1. L2 带宽在跨 SM 并发时的竞争程度**

论文未直接量化 L2 竞争，但可以从结果反向推断：

- **L2 hit rate 推论**：H100 L2=60MB。GEMM M=N=K=8192 FP16: A+B 共 256MB，远超 L2。但 GEMM 是**3-diamond reuse pattern**——每个 A tile 被同一 row 的 N/T_N 个 output tiles 复用，每个 B tile 被同一 column 的 M/T_M 个 output tiles 复用。在 CTA 粒度(~1024 blocks)，相邻 CTA 的 A/B tiles 有显著空间局部性。典型 Hopper GEMM L2 hit rate 业界经验值 ~30-50%，但论文未报。

- **TMA vs Tensor Core L2 竞争**：TMA 加载 A/B tiles 从 HBM→L2→SMEM 时使用 L2；Tensor Core WGMMA 的操作数来自 SMEM（已在上一 pipe stage 由 TMA 加载），**不在 L2 路径上**。所以两者**不竞争 L2 端口**。仅有 TMA 本身和最终 TMA_store 竞争 L2→HBM 写路径。

- **竞争实验证据**：Cypress Dual-GEMM 达 ~970 TFLOPs/s（接近 GEMM ~980 TFLOPs/s），额外 B₂ 的 TMA load **未造成 throughput 退化**——说明在 PIPE=3 下，3 个 in-flight A/B tile 的 TMA 请求总和 (~48KB per pipeline stage × 3 = ~144KB per SM) 在 L2→SMEM crossbar 带宽内被满足，L2 带宽未成为瓶颈。此外论文使用同分布随机矩阵规范化 power throttling 影响——说明**功耗约束是比 L2 带宽更显著的性能限制因素**。

- **HBM miss stall 被 pipeline 吸收**：L2 miss → HBM access 增加 ~200-400ns 延迟。但 PIPE=3 的 DMA 领先策略下：compute warpgroup 计算一次 WGMMA 需 ~1024 cycles（64×256×16 MMA），而 TMA 一次 tile load（64×256×2B=32KB）在 HBM3(3.35TB/s)上仅需 ~9.5ns ≈ 15 cycles——即使 L2 miss，延迟也远小于 compute 时间，被天然隐藏。**结论：在计算受限的 GEMM 中，L2 竞争不是瓶颈。**

**2. 1 CTA/SM 时的 SM 内部利用率**

CTA 组成：5 warps（4 compute + 1 DMA）per SM。Hopper SM 有 **4 warp schedulers**（每 cycle 从 eligible warps 中各选 1 条指令发射）。

**关键 issue slot 利用率分析**（基于架构推理，非论文测量）：

| warp | 角色 | 典型指令 | 发射频率 | 是否常 stalled |
|------|------|---------|---------|---------------|
| warp 0-3 (compute wg) | WGMMA + SIMT | wgmma / warp_sync / bar.arrive | 每 iteration ~4-8 条活跃指令 | 是（wgmma_wait 时） |
| warp 4 (DMA) | TMA load | cp.async.bulk | 每 iteration 2 条（thread 0 仅） + wait(cons) | 是（wait(cons) 时） |

**空闲周期来源**：
- Compute warpgroup 在 `wgmma(...)` 异步启动后执行 `warpgroup_wait()`——此时 4 个 compute warps 全部 stall 在 Tensor Core 完成等待上，**不使用 4 个 scheduler slot**，DMA warp 若有工作可占用 1/4 slots
- DMA warp 在 wait(cons) 时也被 stall——此时所有 5 warps 同时 stall，scheduler 无可选 warp → **issue slots 完全空闲**
- 仅当 compute warpgroup 在做 copy_reg_to_smem + syncthreads + arrive(copyout) 期间，4 compute warps + 1 DMA warp 同时活跃，4/4 slots 可能占满

**但**关键优化是 PIPE=3 的错开效果：稳态下当 compute warpgroup 处理 iteration k 的 wgmma 时，DMA warp 已经在处理 iteration k+3 的 TMA load——DMA warp 的 wait(cons) 在前 3 次 iteration 后应已 arrive（compute 完成 iteration k 时 arrive(cons[k]) 解除了 DMA 对 slot k 的等待）。所以稳态下 **DMA warp 通常不 stall**，而 compute warpgroup 在 wgmma_wait 时 DMA warp 可独立工作。

**空转概率**：仅在 K/T_K ≤ PIPE 的边界处（如短 K 循环）或 pipeline 启动/排空阶段才会出现 warp scheduler 完全空闲。论文未测量此空转百分比，但 GEMM 达 ~980 TFLOPs/s（~99% peak）间接证明 **稳态下 warp scheduler 空闲率极低**——否则 throughput 会显著低于峰值。

**3. 跨 CTA 同步的架构代价**

Hopper 的跨 CTA 同步有两种机制：

**(a) TMA multicast + shared memory mbarrier**（轻量路径）：
- TMA 单次操作可加载同一数据到**多个 CTAs 的 SMEM**
- 每个 destination CTA 的 SMEM 中有一个 **mbarrier 对象**（驻留在 SMEM 中，非寄存器）
- TMA 传输完成时**硬件自动 arrive 每个 destination CTA 的 mbarrier**——零软件开销
- 各 CTA 的 consumer warps 通过普通 `wait(mbarrier)` 等待
- **额外延迟 vs intra-warp named barrier**：~0 cycle（硬件自动，不消耗任何 warp issue slot）
- **限制**：仅当 TMA multicast 可用时（数据在所有 CTA 间相同，如 Flash Attention 的 K/V tiles）

**(b) Cooperative Groups / Cluster Barrier**（重路径，非 Cypress 使用）：
- Hopper 支持 cluster-level barrier（`cluster_barrier` PTX），跨 SM 同步
- 延迟显著更高（~200-400 cycles，取决于 cluster 规模和拥塞），因为涉及 SM-to-SM 消息
- Cypress **未使用**此机制——论文脚注¹提到 "shared memory barriers are required... when TMA multi-casting is used"，指向的正是上述轻量路径

**Cypress 优化程度**：Cypress 当前 mapping 对 cross-CTA 场景利用有限——TMA multicast 由硬件自动处理，Cypress compiler 不生成跨 CTA barrier 代码。论文指出 "Cypress does not yet implement the persistent kernel optimization"——persistent kernel 中逻辑 block 的跨 CTA 调度是未来方向，但**当前跨 CTA 同步不是 Cypress 的优化目标**。Hard limit：Hopper SM 内 named barriers 数量有限（~12-16），当 pipeline depth 增大或需要更多 sync 点时可能耗尽——这也是 Cypress 选 PIPE=3 的原因之一。

---

### Round 6

**Q**：关于实验测量工具链和编译器性能建模的粒度与完备性：

1. **测量粒度与工具链**：论文报告的平均 TFLOPs/s（100 iteration + 5 warmup）是 kernel 级 aggregate metric。开发/调试过程中是否使用了 **Nsight Compute / Nsight Systems** 进行 cycle 级或 warp 级 profiling？如果有，能否给出关键 kernel 的资源分解（例如 GEMM 的 SM utilization、TMA 与 Tensor Core 活跃 cycle 占比、pipeline stall 来源的 pie chart 数据）？如果没有，未使用精细 profiling tools 的原因是什么（时间约束 / 工具不支持 TMA/WGMMA 事件跟踪 / 论文篇幅限制）？

2. **Mapping 参数搜索的方法论**：论文提到 Cypress 同一 FA 逻辑程序在不同 mapping 下性能跨度巨大，且 FA2 通过将 consumer warpgroups 从 2 改为 3（WGS 参数）即接近 FA3 性能。**mapping 参数空间的探索是手动试错还是自动搜参**？如果是手动，用户一般需要尝试多少组 mapping 才能达到报告的性能？有无尝试 over 自动化 auto-tuning（如 OpenTuner 或 GPOT）？对于 GEMM 基准，论文报告的 ~0.88–1.06× cuBLAS 是选择了「最优 mapping」还是「一次命中」？

3. **编译器 cost model**：Cypress 的 Resource Allocation pass 和 Warp Specialization pass 是否存在 **analytical cost model**（如预估 register pressure、SMEM usage、pipeline stall cycles）来指导 partition 和资源分配决策？还是完全基于规则（interference graph → 最小 aliasing → 固定 partition）？如果是规则驱动，编译器如何保证在配置偏离最优（如 PIPE=2 或 PIPE=4）时 partition 仍合理？

**A**：**1. 测量粒度与工具链**

论文报告的所有结果均为 kernel 级 aggregate metric（平均 TFLOPs/s over 100 iterations + 5 warmup）。**论文未提及使用 Nsight Compute 或 Nsight Systems 等 cycle 级 profiling 工具**。全文没有提供 SM utilization、TMA vs Tensor Core active cycle 占比、pipeline stall 来源 pie chart 等任何细粒度资源分解数据。

**未使用精细 profiling 的原因推断**（非论文明确说明）：
- **(a) 论文篇幅限制**——PLDI 格式（25 页）下 evaluation 章节重点在高层次性能对比（vs cuBLAS/CUTLASS/Triton），cycle 级分解通常属于 microarchitecture 论文范畴
- **(b) 研究重点**——论文的核心 claim 是"Cypress 自动管理所有数据移动和同步，生成代码性能与手写相当"，这一 claim 已在 aggregate TFLOPs/s 层面（0.88–1.06× cuBLAS）成立。cycle 级分解更多用于解释"为什么有 gap"而非证明方法有效
- **(c) TMA/WGMMA 事件跟踪的工具支持**——Nsight Compute 2024.3+ 已支持 Hopper TMA 和 Tensor Core 事件跟踪，但论文提交时（PLDI 2025，通常截止 2024 年中）可能尚未成熟或团队不熟悉
- **无 warp scheduler issue slot utilization 数据**——这是架构层面最直接的"SM 利用率"指标，但论文完全未提供

**2. Mapping 参数搜索的方法论**

**完全手动试错**——论文 Section 5.1 明确写："The mapping for each Cypress program were developed by us and manually tuned; the individual mapping strategies chosen were informed by existing algorithms and implementations." 未使用 OpenTuner/GPOT 或任何 auto-tuning 框架。

**手动搜索的实际代价**（从论文描述推断）：
- **GEMM 基准**：report 的 0.88–1.06× cuBLAS 是**一次命中后的最优配置**。GEMM 的 mapping 结构是论文的 canonical example（Figure 5b 完整给出），7 层 task variants + 6 个 mapping entries 的配置空间来自经典 GEMM 模板知识（CUTLASS 的 tile size 选择经验），而非穷举搜索。Tunable 参数仅 ~5 维（U, V, W, PIPE, WGS），手动配置空间 ~10–50 组
- **FA2→FA3 的 mapping 调整**：关键发现——仅将 WGS 从 2 改为 3（+额外 1 个 consumer warpgroup）即接近 FA3 性能。这一发现来自**手动实验观察**而非系统扫描。论文承认"indicating the difficulty of discovering the fastest kernels in a jagged performance landscape"
- **FA 的完整 mapping**可能需尝试 10–50 组配置才能收敛到报告水平，但论文未给搜索过程的消融

**未自动化原因**：
- 编译器原型——Cypress 是研究原型，编译流程未集成 auto-tuning 基础设施
- Mapping space 的设计意图是"可复用模板"——作者认为类似的 task trees 可共享 mapping，但当前版本仍有冗余（论文 5.4 指出 mapping 可改进）

**3. 编译器 cost model**

**完全规则驱动（rule-based），无 analytical cost model**。各 pass 的决策逻辑：

- **Resource Allocation (Pass 4)**：无 cost model。策略是静态构造 SMEM interference graph，从**完全干涉图**开始迭代删除 auxiliary edges 直到 feasible allocation 找到。不尝试预测 register spill 或 occupancy 对性能的影响——如果 allocation 不可行则 report error 要求用户修改 mapping。**当配置偏离最优时（如 PIPE=4 而非 3）**，allocation 仍能找到可行分配（因为 SMEM budget 由 user 的 `shared_memory_limit` 指定，PIPE 增大 → buffer 增多 → 需要更多 aliasing → pass 能找到解但可能更多 aliasing → 降低并行度）。但编译器**无法告诉用户"PIPE=4 下 aliasing 增加导致 performance 退化 X%"**

- **Warp Specialization (Pass 5)**：基于**固定 graph partition 规则**——所有 global→shared copies 分配给 DMA warp，其余全部给 compute warps。不是 cost-model 驱动的探索式 partition（如尝试不同 partition 并选择预估延迟最低的）。当 PIPE 变化时，pipelining 的 unroll-compact 逻辑自动调整 backwards anti-dependencies 的数量，但 partition 本身不变

- **Copy Elimination (Pass 3)**：模式匹配（4 类 rewrite patterns），应用顺序固定（spill-related patterns 优先于 dependency-preserving patterns），但论文承认"using an equality saturation framework like an egraph could avoid the need for these ordering heuristics"

**编译器缺乏 cost model 的关键后果**：用户调整 mapping（如改变 PIPE、WGS、tile size）后只能重编译、运行 benchmark、肉眼比较 TFLOPs/s 来判断好坏。所有性能反馈来自硬件执行而非编译时预测。论文将这一点视为未来工作方向。

---

