# V. DYNAMIC TILE SCHEDULING

With the semantic context (OpType), resource mapping (UnitMap), and memory range descriptors (TileMem), the dynamic scheduler can safely exploit cross-operator overlap and adaptive resource allocation at runtime. We now detail how typed dependency analysis and heterogeneous resource management work together to realize this capability.

## *A. Typed Dependency Analysis*

Conventional instruction schedulers rely on register-based or conservative memory-based dependency tracking, which either fail to capture higher-level semantics or conservatively block parallelism. In contrast, our dynamic tile scheduler performs *typed dependency analysis* that explicitly leverages semantic annotations in TISA instructions. This approach eliminates artificial dependencies while ensuring correctness, enabling safe reordering and parallel execution across operators and iterations. Each TISA instruction exposes its data interfaces through TileMem fields that describe address ranges, memory scopes (e.g., L1-private, L2-local, or HBM channel), and access types (READ/WRITE). Accesses to disjoint memory scopes or non-overlapping ranges are treated as independent, allowing the scheduler to issue instructions concurrently without risking data hazards.

To track runtime state, each execution unit u maintains an *in-flight semantic table*:

$$\mathcal{F}_{u} = \left\{ \begin{array}{c} (idx, start\_addr, end\_addr, \\ access\_type, unit, inst\_ptr) \end{array} \right\}$$

Here, idx indexes the instruction's operand; start\_addr/end\_addr denote the address interval accessed by that operand; access\_type is the access mode (READ/WRITE); unit is the target execution unit; and inst\_ptr is a pointer to the instruction. This table allows the scheduler to reason about partial completion (e.g., when sub-tiles or memory regions become ready) and to wake up dependent tiles immediately once their required data is available. Unlike traditional scoreboards that track register tags, F<sup>u</sup> encodes semantic and spatial context, enabling fine-grained, scope-aware dependency resolution.

When evaluating whether a candidate instruction I can be issued to unit u, our dynamic scheduler performs rule-based hazard detection:

$$\operatorname{Hazard}(I,\mathcal{F}_u) = \exists r \in \mathcal{F}_u : \operatorname{SemanticConflict}(I,r)$$

where SemanticConflict is resolved through the following typed rules:

- Data dependencies (RAW/WAR/WAW): use interval overlap tests on TileMem to identify true conflicts; disjoint ranges are considered independent.
- Memory-scope isolation: operations targeting distinct memory levels or banks (e.g., L1 vs. L2, or separate HBM channels) are non-aliasing and can safely proceed in parallel.
- Semantic compatibility: instructions with compatible OpTypes (e.g., GEMM vs. Softmax) may overlap if their data ranges do not conflict and their unit classes differ.
- Resource feasibility: if current unit capacity (e.g., local buffer size or bandwidth) cannot accommodate I, the scheduler defers its issue until sufficient resources are released.

These rule checks extend traditional hazard models by combining data, structural, and semantic dimensions in one unified framework. They allow the scheduler to distinguish true hazards that affect correctness from benign overlaps that can be exploited for parallel execution. Currently, TISA's TileMem lacks native strided array access, requiring compilers to transpose columns into contiguous addresses or emit conservative boundaries yielding false dependencies. However, since dense LLMs and CNNs intrinsically operate on large contiguous blocks, precision loss remains negligible.

# V. DYNAMIC TILE SCHEDULING

With the semantic context (OpType), resource mapping (UnitMap), and memory range descriptors (TileMem), the dynamic scheduler can safely exploit cross-operator overlap and adaptive resource allocation at runtime. We now detail how typed dependency analysis and heterogeneous resource management work together to realize this capability.

## *A. Typed Dependency Analysis*

Conventional instruction schedulers rely on register-based or conservative memory-based dependency tracking, which either fail to capture higher-level semantics or conservatively block parallelism. In contrast, our dynamic tile scheduler performs *typed dependency analysis* that explicitly leverages semantic annotations in TISA instructions. This approach eliminates artificial dependencies while ensuring correctness, enabling safe reordering and parallel execution across operators and iterations. Each TISA instruction exposes its data interfaces through TileMem fields that describe address ranges, memory scopes (e.g., L1-private, L2-local, or HBM channel), and access types (READ/WRITE). Accesses to disjoint memory scopes or non-overlapping ranges are treated as independent, allowing the scheduler to issue instructions concurrently without risking data hazards.

To track runtime state, each execution unit u maintains an *in-flight semantic table*:

$$\mathcal{F}_{u} = \left\{ \begin{array}{c} (idx, start\_addr, end\_addr, \\ access\_type, unit, inst\_ptr) \end{array} \right\}$$

Here, idx indexes the instruction's operand; start\_addr/end\_addr denote the address interval accessed by that operand; access\_type is the access mode (READ/WRITE); unit is the target execution unit; and inst\_ptr is a pointer to the instruction. This table allows the scheduler to reason about partial completion (e.g., when sub-tiles or memory regions become ready) and to wake up dependent tiles immediately once their required data is available. Unlike traditional scoreboards that track register tags, F<sup>u</sup> encodes semantic and spatial context, enabling fine-grained, scope-aware dependency resolution.

When evaluating whether a candidate instruction I can be issued to unit u, our dynamic scheduler performs rule-based hazard detection:

$$\operatorname{Hazard}(I,\mathcal{F}_u) = \exists r \in \mathcal{F}_u : \operatorname{SemanticConflict}(I,r)$$

where SemanticConflict is resolved through the following typed rules:

- Data dependencies (RAW/WAR/WAW): use interval overlap tests on TileMem to identify true conflicts; disjoint ranges are considered independent.
- Memory-scope isolation: operations targeting distinct memory levels or banks (e.g., L1 vs. L2, or separate HBM channels) are non-aliasing and can safely proceed in parallel.
- Semantic compatibility: instructions with compatible OpTypes (e.g., GEMM vs. Softmax) may overlap if their data ranges do not conflict and their unit classes differ.
- Resource feasibility: if current unit capacity (e.g., local buffer size or bandwidth) cannot accommodate I, the scheduler defers its issue until sufficient resources are released.

These rule checks extend traditional hazard models by combining data, structural, and semantic dimensions in one unified framework. They allow the scheduler to distinguish true hazards that affect correctness from benign overlaps that can be exploited for parallel execution. Currently, TISA's TileMem lacks native strided array access, requiring compilers to transpose columns into contiguous addresses or emit conservative boundaries yielding false dependencies. However, since dense LLMs and CNNs intrinsically operate on large contiguous blocks, precision loss remains negligible.

