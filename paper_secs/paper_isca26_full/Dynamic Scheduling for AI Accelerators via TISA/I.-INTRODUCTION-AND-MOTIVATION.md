# I. INTRODUCTION AND MOTIVATION

Deep learning's rapid growth has driven the development of increasingly sophisticated accelerators [11], [35]. Representative platforms include GPUs (both NVIDIA and AMD [26]), TPUs [20], Ascend [25], and Tenstorrent [37], [38]. These systems employ heterogeneous-unit architectures that integrate tensor, vector, and DMA engines to exploit massive parallelism, as exemplified in Table I.

TABLE I: Heterogeneous execution units across vendors. We unify vendor-specific terms into "Tensor Units" and "Vector ALUs". "DMA Engines" denotes DMA / copy engines / onchip packet movers. Mappings follow vendor docs and prior work [20], [25], [26], [37], [38].

| Vendor      | Tensor Units | Vector ALUs | DMA Engines |
|-------------|--------------|-------------|-------------|
| NVIDIA      | Tensor       | CUDA cores  | TMA         |
| AMD         | Matrix       | SIMD        | SDMA        |
| TPU         | MXU          | V-ALU       | async DMA   |
| Ascend      | Cube         | VU          | on-chip DMA |
| Tenstorrent | SFPU         | FPU         | on-chip DMA |

Current AI accelerators rely primarily on compile-time orchestrated pipeline templates to coordinate heterogeneous units. While modern GPUs employ dynamic scheduling at warp issue, thread-block assignment, and asynchronous memory execution, these mechanisms operate below tile-level operator coordination and do not dynamically reorder crossunit execution based on semantic readiness. This approach remains prevalent because it matches the long-standing softwarehardware contract: compilers emit deterministic instruction streams, and hardware executes them predictably with minimal control overhead. This model simplifies verification, toolchain design, and aligns with bulk-synchronous programming paradigms (e.g., CUDA [27], XLA [36], TensorRT [28]).

However, as both model and hardware complexity scale, static tile-level pipeline scheduling exposes some bottlenecks. Although overlapping tensor, vector, and DMA units promises instruction-level parallelism (ILP) [8], [22], [35], [39], [41], compile-time decisions cannot respond to runtime variability such as DMA backpressure, cache-bank conflicts, or thermal throttling. Consequently, execution deviates from compile-time assumptions: tiles cannot be retimed or rebalanced across units, leaving idle bubbles and underutilized hardware. Tiles arise naturally here because modern compilers already partition operators into tiles to match on-chip memory and bandwidth constraints. Hence, tiles represent the finest semantically meaningful granularity shared across compute, memory, and DMA, making them an ideal unit for adaptive scheduling.

Despite its ubiquity, static tile-level pipeline scheduling suffers from four fundamental limitations that constrain scalability and efficiency:

- Compile-time complexity and engineering burden. Coordinating cross-unit (tensor/vector/DMA) dependencies and exploring reordering opportunities pose optimization challenges related to the minimum makespan and bin packing problems [18], both of which are NPhard problems. Consequently, developers often resort to architecture-specific heuristics and manual tuning, which scale poorly with system complexity.
- Inadequate abstraction granularity. CUDA streams and graph-level runtimes manage kernels at coarse granularity, obscuring tile boundaries and data dependencies. In contrast, instruction-level (warp) scheduling [46] operates at a very fine granularity where semantic meaning is largely lost. Tile-level scheduling strikes an effective balance: it preserves operator context and exposes crossunit overlap while remaining hardware-schedulable [43].
- Inability to adapt to runtime variability. Static tilelevel pipeline schedules assume fixed latencies and bandwidth, but real systems exhibit dynamic fluctuations such as bandwidth contention, Cache/SPM conflicts, and unit

desynchronization due to thermal or operating system effects. These shifts alter the true critical path, yet static approaches cannot retime or re-overlap tiles accordingly.

• Historical precedent: static approaches fail under uncertainty. Superscalar CPUs [7], [15] achieve high ILP via dynamic scheduling, whereas static VLIW/IA-64 [5], [13] architectures failed to generalize across workloads and microarchitectural variation. This history suggests that moving a bounded, semantics-aware portion of scheduling to runtime is both robust and scalable.

Collectively, these limitations highlight the need for a runtime-visible semantic layer that enables adaptive reordering at an appropriate granularity.

Dynamic scheduling provides a path forward, but applying it at arbitrary instruction granularity would be impractical for deep learning accelerators. The tile abstraction provides the sweet spot: each tile encapsulates a semantically coherent computation with well-defined data ranges and resource requirements. This granularity enables the runtime to (1) detect dependencies precisely, (2) arbitrate heterogeneous resources safely, and (3) adapt execution in response to runtime conditions. Tile-granular scheduling thus combines the adaptivity of dynamic systems with the semantic safety of compile-time reasoning, forming the foundation of our design.

We propose a semantics-aware, dynamic scheduling framework at the tile granularity that unifies software semantics and hardware scheduling through the co-design of three synergistic components: a semantics-preserving compiler that maintains operator identity, typed dependencies, and resource affinities through the compilation pipeline, preventing the semantic erosion that limits current runtimes; a tile-level ISA (TISA) that acts as an orthogonal *scheduling-semantics* layer atop existing per-unit execution ISAs, encoding each tile's operator type, dependency descriptors, resource intents, and memory ranges, giving the runtime enough information to reason about legality, readiness, and overlap without expensive static analysis; and a semantics-guided runtime scheduler that consumes TISA semantics to dynamically reorder tiles across tensor, vector, and DMA units, resolving contention and exploiting cross-operator and cross-iteration parallelism under runtime variability.

This workflow directly addresses the weaknesses of static tile-level pipeline scheduling: (1) semantic preservation removes the need for combinatorial compile-time reordering; (2) tile-level granularity balances scheduling visibility and hardware tractability; and (3) dynamic decisions adapt execution to runtime drift while ensuring correctness.

We evaluate our dynamic scheduling framework across representative workloads, including ResNet50 [17], BERT [10], GPT-J [31], LLaMA2 [42], and DeepSeek-R1 [16], on multiple accelerators including Epoch, H100, and A100 GPUs. Our work achieves 1.52–1.92× speedups over the baseline, delivering 1.14–1.63× additional improvement over strong static tilelevel pipeline scheduling. For FlashAttention operators in the mainstream head-dim-128 configuration, it delivers approximately 26.4% higher hardware utilization than the state-of-theart H100 FlashAttention-3 [33]. Ablations further show that semantic preservation alone yields 1.2× gains, underscoring the independent value of restoring runtime-visible semantics. In summary, the key contributions of this work are:

- We design a semantics-aware dynamic tile scheduling framework that bridges static compilation and runtime adaptability for heterogeneous AI accelerators.
- We introduce TISA, a tile-level instruction set that exposes typed dependencies, operator semantics, and resource requirements for safe dynamic reordering.
- We build a semantics-preserving compiler that maintains operator context from high-level frameworks to hardwareschedulable TISA instructions.
- We present comprehensive experimental validation showing improved utilization, adaptability, and portability across accelerators and workloads.

The paper is organized as follows. Section II identifies specific scheduling challenges. Section III analyzes fundamental gaps and presents the overall architecture. Section IV formalizes TISA semantics, followed by the introduction of our dynamic scheduler in Section V and the semantics-preserving compiler in Section VI, respectively. Section VII presents the implementation of our framework, while Section VIII provides comprehensive experimental validation. Finally, Section IX discusses related work, and Section X concludes.

# I. INTRODUCTION AND MOTIVATION

Deep learning's rapid growth has driven the development of increasingly sophisticated accelerators [11], [35]. Representative platforms include GPUs (both NVIDIA and AMD [26]), TPUs [20], Ascend [25], and Tenstorrent [37], [38]. These systems employ heterogeneous-unit architectures that integrate tensor, vector, and DMA engines to exploit massive parallelism, as exemplified in Table I.

TABLE I: Heterogeneous execution units across vendors. We unify vendor-specific terms into "Tensor Units" and "Vector ALUs". "DMA Engines" denotes DMA / copy engines / onchip packet movers. Mappings follow vendor docs and prior work [20], [25], [26], [37], [38].

| Vendor      | Tensor Units | Vector ALUs | DMA Engines |
|-------------|--------------|-------------|-------------|
| NVIDIA      | Tensor       | CUDA cores  | TMA         |
| AMD         | Matrix       | SIMD        | SDMA        |
| TPU         | MXU          | V-ALU       | async DMA   |
| Ascend      | Cube         | VU          | on-chip DMA |
| Tenstorrent | SFPU         | FPU         | on-chip DMA |

Current AI accelerators rely primarily on compile-time orchestrated pipeline templates to coordinate heterogeneous units. While modern GPUs employ dynamic scheduling at warp issue, thread-block assignment, and asynchronous memory execution, these mechanisms operate below tile-level operator coordination and do not dynamically reorder crossunit execution based on semantic readiness. This approach remains prevalent because it matches the long-standing softwarehardware contract: compilers emit deterministic instruction streams, and hardware executes them predictably with minimal control overhead. This model simplifies verification, toolchain design, and aligns with bulk-synchronous programming paradigms (e.g., CUDA [27], XLA [36], TensorRT [28]).

However, as both model and hardware complexity scale, static tile-level pipeline scheduling exposes some bottlenecks. Although overlapping tensor, vector, and DMA units promises instruction-level parallelism (ILP) [8], [22], [35], [39], [41], compile-time decisions cannot respond to runtime variability such as DMA backpressure, cache-bank conflicts, or thermal throttling. Consequently, execution deviates from compile-time assumptions: tiles cannot be retimed or rebalanced across units, leaving idle bubbles and underutilized hardware. Tiles arise naturally here because modern compilers already partition operators into tiles to match on-chip memory and bandwidth constraints. Hence, tiles represent the finest semantically meaningful granularity shared across compute, memory, and DMA, making them an ideal unit for adaptive scheduling.

Despite its ubiquity, static tile-level pipeline scheduling suffers from four fundamental limitations that constrain scalability and efficiency:

- Compile-time complexity and engineering burden. Coordinating cross-unit (tensor/vector/DMA) dependencies and exploring reordering opportunities pose optimization challenges related to the minimum makespan and bin packing problems [18], both of which are NPhard problems. Consequently, developers often resort to architecture-specific heuristics and manual tuning, which scale poorly with system complexity.
- Inadequate abstraction granularity. CUDA streams and graph-level runtimes manage kernels at coarse granularity, obscuring tile boundaries and data dependencies. In contrast, instruction-level (warp) scheduling [46] operates at a very fine granularity where semantic meaning is largely lost. Tile-level scheduling strikes an effective balance: it preserves operator context and exposes crossunit overlap while remaining hardware-schedulable [43].
- Inability to adapt to runtime variability. Static tilelevel pipeline schedules assume fixed latencies and bandwidth, but real systems exhibit dynamic fluctuations such as bandwidth contention, Cache/SPM conflicts, and unit

desynchronization due to thermal or operating system effects. These shifts alter the true critical path, yet static approaches cannot retime or re-overlap tiles accordingly.

• Historical precedent: static approaches fail under uncertainty. Superscalar CPUs [7], [15] achieve high ILP via dynamic scheduling, whereas static VLIW/IA-64 [5], [13] architectures failed to generalize across workloads and microarchitectural variation. This history suggests that moving a bounded, semantics-aware portion of scheduling to runtime is both robust and scalable.

Collectively, these limitations highlight the need for a runtime-visible semantic layer that enables adaptive reordering at an appropriate granularity.

Dynamic scheduling provides a path forward, but applying it at arbitrary instruction granularity would be impractical for deep learning accelerators. The tile abstraction provides the sweet spot: each tile encapsulates a semantically coherent computation with well-defined data ranges and resource requirements. This granularity enables the runtime to (1) detect dependencies precisely, (2) arbitrate heterogeneous resources safely, and (3) adapt execution in response to runtime conditions. Tile-granular scheduling thus combines the adaptivity of dynamic systems with the semantic safety of compile-time reasoning, forming the foundation of our design.

We propose a semantics-aware, dynamic scheduling framework at the tile granularity that unifies software semantics and hardware scheduling through the co-design of three synergistic components: a semantics-preserving compiler that maintains operator identity, typed dependencies, and resource affinities through the compilation pipeline, preventing the semantic erosion that limits current runtimes; a tile-level ISA (TISA) that acts as an orthogonal *scheduling-semantics* layer atop existing per-unit execution ISAs, encoding each tile's operator type, dependency descriptors, resource intents, and memory ranges, giving the runtime enough information to reason about legality, readiness, and overlap without expensive static analysis; and a semantics-guided runtime scheduler that consumes TISA semantics to dynamically reorder tiles across tensor, vector, and DMA units, resolving contention and exploiting cross-operator and cross-iteration parallelism under runtime variability.

This workflow directly addresses the weaknesses of static tile-level pipeline scheduling: (1) semantic preservation removes the need for combinatorial compile-time reordering; (2) tile-level granularity balances scheduling visibility and hardware tractability; and (3) dynamic decisions adapt execution to runtime drift while ensuring correctness.

We evaluate our dynamic scheduling framework across representative workloads, including ResNet50 [17], BERT [10], GPT-J [31], LLaMA2 [42], and DeepSeek-R1 [16], on multiple accelerators including Epoch, H100, and A100 GPUs. Our work achieves 1.52–1.92× speedups over the baseline, delivering 1.14–1.63× additional improvement over strong static tilelevel pipeline scheduling. For FlashAttention operators in the mainstream head-dim-128 configuration, it delivers approximately 26.4% higher hardware utilization than the state-of-theart H100 FlashAttention-3 [33]. Ablations further show that semantic preservation alone yields 1.2× gains, underscoring the independent value of restoring runtime-visible semantics. In summary, the key contributions of this work are:

- We design a semantics-aware dynamic tile scheduling framework that bridges static compilation and runtime adaptability for heterogeneous AI accelerators.
- We introduce TISA, a tile-level instruction set that exposes typed dependencies, operator semantics, and resource requirements for safe dynamic reordering.
- We build a semantics-preserving compiler that maintains operator context from high-level frameworks to hardwareschedulable TISA instructions.
- We present comprehensive experimental validation showing improved utilization, adaptability, and portability across accelerators and workloads.

The paper is organized as follows. Section II identifies specific scheduling challenges. Section III analyzes fundamental gaps and presents the overall architecture. Section IV formalizes TISA semantics, followed by the introduction of our dynamic scheduler in Section V and the semantics-preserving compiler in Section VI, respectively. Section VII presents the implementation of our framework, while Section VIII provides comprehensive experimental validation. Finally, Section IX discusses related work, and Section X concludes.

