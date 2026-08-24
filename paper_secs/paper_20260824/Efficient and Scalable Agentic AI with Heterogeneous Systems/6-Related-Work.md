# 6 Related Work

Recent advances in large-scale machine learning systems have led to the development of specialized infrastructure for model serving, disaggregated execution, compiler optimization, and multi-agent orchestration. In this section, we review representative work across each of these areas. While prior efforts offer important building blocks—ranging from low-level kernel optimization to high-level agent abstractions—they typically operate in isolation, without a unifying system that optimizes execution across heterogeneous compute. Our work builds on these foundations and introduces an optimization framework that integrates cost, performance, and hardware diversity into a cohesive planning model for AI agent workloads.

#### 6.1 Model Serving

Recent advances in model serving have primarily targeted enhancing the efficiency and performance of Large Language Models (LLMs) through specialized software infrastructures. Prominent examples include the vLLM [\[12\]](#page-22-11) and TensorRT-LLM [\[44\]](#page-24-4) frameworks, which have introduced significant software-level optimizations to enhance inference throughput, latency, and memory management.

vLLM introduces an innovative technique called paged attention, which substantially improves batched inference efficiency by effectively managing key-value (KV) caches. This design facilitates continuous batching, minimizes memory fragmentation, and is particularly suited for high-throughput, low-latency deployments. However, vLLM's design is inherently model-centric and assumes a homogeneous hardware environment, thereby limiting its applicability to heterogeneous computing scenarios and comprehensive agentic workloads.

SGLang [\[45\]](#page-24-5) represents a recent effort to provide a high-level programming interface for LLM serving, combining structured prompt orchestration with system-level performance optimizations. It incorporates a custom runtime and memory-aware scheduling to support latency-sensitive applications. However, like vLLM, SGLang primarily targets homogeneous infrastructure and single-model workloads, and does not address broader agentic or heterogeneous execution contexts.

TensorRT-LLM employs optimized CUDA kernels, quantization strategies, and operator fusion techniques to maximize GPU utilization. Specifically tailored to NVIDIA hardware, TensorRT-LLM achieves notable performance by closely aligning model structures with hardware-specific optimizations. However, this hardware-software tight coupling significantly restricts cross-vendor portability and flexibility.

In contrast to prior approaches focused primarily on maximizing throughput and minimizing latency within isolated runtime contexts, our research proposes a more generalized optimization framework that explicitly incorporates operational costs, hardware heterogeneity, and the comprehensive efficiency of entire AI agent workloads.

### 6.2 Disaggregated Serving

Recent studies have explored disaggregated inference architectures, where scheduling, execution, and memory management functionalities are decoupled and distributed across a heterogeneous set of computing resources.

Splitwise [\[46\]](#page-24-6) exemplifies this approach by explicitly decomposing inference workloads into prefill and decode stages, executed across distinct nodes. Splitwise also illustrates practical heterogeneous deployment by employing two different NVIDIA accelerators, selected based on distinct performance-cost trade-offs, demonstrating the potential efficiency benefits of adaptive resource allocation.

NVIDIA's comprehensive inference stack, including NVIDIA Dynamo [\[47\]](#page-24-7), provides an integrated solution designed explicitly for disaggregated inference workloads. Components such as NVIDIA Dynamo Planner, NVIDIA Dynamo Smart Router, NVIDIA Dynamo Distributed KV Cache Manager, and NVIDIA Inference Transfer Library (NIXL) address various stages from workload compilation and scheduling to execution. However, despite the stack's completeness, it remains deeply embedded within NVIDIA's proprietary hardware and software ecosystem, limiting its applicability to broader, vendor-neutral contexts.

The llm-d platform [\[48\]](#page-24-8), an extension of the vLLM framework, offers disaggregated inference by separating prefill and decode operations across individual nodes. Its scheduler determines optimal workload placement based on KV cache state, service-level agreements (SLAs), and system load. Nevertheless, llm-d inherits fundamental constraints from its vLLM foundation, notably restricting deployment to one model per node, which can limit efficient resource utilization.

Mitra et al. [\[42\]](#page-24-2) present an extensive empirical analysis of disaggregated inference, systematically evaluating numerous configurations across diverse workloads and hardware settings. Their findings highlight that disaggregated serving yields substantial benefits, particularly for workloads characterized by high prefill demands and larger model sizes. Moreover, they emphasize the necessity of dynamic rate matching and elastic resource scaling as critical strategies to achieve Pareto-optimal balances between throughput and interactivity.

Our optimization framework generalizes these approaches, integrating both disaggregated and monolithic serving strategies as specific instances within a unified optimization formulation. By explicitly considering cost, performance, and hardware heterogeneity, it facilitates effective optimization of AI agent workloads across diverse computational environments.

#### 6.3 MLIR-Based Efforts

Several recent efforts leverage Multi-Level Intermediate Representation (MLIR) to optimize machine learning workloads across heterogeneous hardware. MLIR serves as a foundational tool enabling hardware-agnostic optimizations and transformations that facilitate efficient code generation for diverse computing architectures.

IREE [\[49\]](#page-24-9) and MHLO [\[50\]](#page-24-10) are prominent examples demonstrating MLIR's potential for portable, high-performance compilation. IREE supports comprehensive end-to-end compilation and execution, accommodating various backend targets including CPUs, GPUs, and accelerators. MHLO offers a standardized representation for tensor operations, streamlining the compilation and optimization pipeline across multiple hardware platforms. However, existing MLIRbased frameworks primarily target individual model execution and do not explicitly optimize across complex agentic workloads with disaggregated execution scenarios.

Triton [\[51\]](#page-24-11) represents a differentiated yet complementary approach to MLIR-based systems. Rather than exposing a general IR for graph-level transformations, Triton offers a Python-based programming model focused on writing highly efficient GPU kernels. Triton has been used effectively to optimize dense linear algebra and memory-bound kernels within LLM workloads, and integrates well with PyTorch through custom operations. However, Triton's scope is primarily focused on kernel-level optimization rather than end-to-end graph compilation, and lacks intrinsic mechanisms to target heterogeneous or disaggregated systems.

Our work similarly leverages MLIR but extends its use to optimize across entire agentic workloads, specifically addressing heterogeneous hardware and disaggregated execution contexts.

#### 6.4 Agent Frameworks

A growing number of frameworks have emerged to structure, coordinate, and execute agentic workloads. Lang-Graph [\[16\]](#page-22-15) provides a graph-based programming model for composing agent behaviors as stateful transitions over tool and memory nodes, enabling fine-grained control over execution flow. CrewAI [\[52\]](#page-24-12) and Autogen [\[53\]](#page-24-13) introduce structured abstractions for collaborative multi-agent systems, with an emphasis on division of labor, role assignment, and tool integration. These systems facilitate modular composition of agents and streamline orchestration, though they often focus on the programming abstraction and rely on general-purpose runtimes.

In contrast, our work complements these abstractions by introducing a cost- and performance-aware execution planning layer. While existing frameworks provide high-level semantics for agent interaction, they do not address optimal taskto-hardware assignment or the underlying systems challenges associated with heterogeneous execution environments.

