# *A. Large Language Models (LLMs)*

Large Language Models (LLMs) have scaled to unprecedented levels due to the effectiveness proven by the scaling law [\[29\]](#page-14-5). These models, trained on various datasets, span billions or even trillions of parameters [\[48\]](#page-15-10). For the current LLMs, the decoder-only transformer is adopted by many popular models such as LLaMA [\[57\]](#page-15-0) and GPT [\[6\]](#page-13-0). The general architecture of the decoder-only transformer consists of repeated blocks, each of which includes a series of main operations such as Layer-Norm, Multihead-Attention, MLP. Within Multihead Attention and MLP layers, the computation is further decomposed into finer-grained operations, such as matrix multiplications (e.g., Linear and MatMul), activation functions (e.g., Softmax and GeLU), and regularization components like Dropout and LayerNorm.

#### <span id="page-1-2"></span>*B. Multi-dimensional Parallelization Strategies*

To support large-scale LLM training, sufficient memory is required to store both model weights and input activations, along with adequate computational resources to complete training within a reasonable timeframe. Consequently, the following parallelization strategies are used in practice.

- Data Parallelism (DP): Splits input data across devices with replicated weights; synchronizes gradients after backward pass [\[48\]](#page-15-10)
- Fully Sharded Data Parallelism (FSDP): Shards both input batches and model parameters across devices, reducing memory use but adding communication to gather parameters during training [\[68\]](#page-15-11).
- Tensor Parallelism (TP): Shards model weights across devices while replicating input data; requires AllReduce to exchange activations after each layer.
- Sequence Parallelism (SP): Splits input sequences into tokens; complements TP by replacing AllReduce with more efficient AllGather and ReduceScatter.
- Pipeline Parallelism (PP): Divides model into stages and pipelines microbatches for concurrent execution across devices.
- Expert Parallelism (EP): For MoE models, uses AllToAll to route tokens to specialized experts after attention layers.

<span id="page-1-0"></span><sup>2</sup>For terminology purposes, we define execution *graphs* to refer to the structure (nodes and dependencies) of the distributed workload, while execution *traces* capture an EG with timing after execution on a real system.

![](_page_2_Figure_0.jpeg)

<span id="page-2-1"></span>Fig. 2. An Example of Execution Graph for GPU Operations.

Each strategy introduces unique patterns for computation, memory access and network communications for a large-scale system [48], [54]. To maximize efficiency and scalability, LLM frameworks today combine multiple parallelism strategies for training different workloads, such as DP, TP, SP, PP and EP within a single model. More details are covered in Sec. III.

#### C. Execution Traces

Modern large-scale machine learning systems consist of interleaved compute and communication operations, often with complex execution order and data flow. Graph-based formats are particularly useful to represent this behavior. They represent compute and communication operations as nodes, and encode data and control dependencies as edges. This structure enables analysis of execution order, critical paths, operator overlap, and performance bottlenecks. Fig. 2 illustrates a simplified execution graph depicting GPU computations, communications, and their inter-operation dependencies. Operation specific parameters, such as the tensor size of a GeMM operation, are encoded as attributes within each node. Following the control and data dependencies from left to right, it reveals which operations must run sequentially and which can execute in parallel across different tensor objects. Labeled tensor sizes are updated following each computation or communication operation between dependent tensors.

Execution traces (ETs) provide a structured view of these operations by capturing actual runtime behavior. They record the execution graph, along with additional metadata such as device type, execution time, and memory usage. Tools like PyTorch profiler [46], Kineto [56], PARAM [52], and Chakra [38] collect these traces at various abstraction levels.

#### III. MOTIVATION

<span id="page-2-0"></span>In this section, we will identify a few challenges of the current approach, then summarize the STAGE design principle.

#### A. Challenges

Challenge 1: Limitations of Real-System ETs. Access to high-fidelity workloads is crucial for optimization and DSE efforts. However, obtaining real ETs is extremely prohibitive in practice due to the high computational and financial cost of running LLMs over large clusters. Moreover, data-sharing limitations prevent organizations with resources from making internal ETs publicly available due to security concerns. Furthermore, even if system designers/optimizers have access to real ETs, their properties are inherently tied to the model architecture, parallelism strategy, and underlying hardware platform (e.g., fused operators depend on compiler support

<span id="page-2-3"></span>TABLE II
CATEGORIZATION OF MODERN LLM COMPONENTS AND PARALLELISM
STRATEGIES WITH NATIVE SUPPORT IN STAGE

| Category                                                                                  | Component                                                                | Origin Source                                                                                                                      |
|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| Attention Mechanism                                                                       | Multi-head<br>Group Query Attention<br>Multi-latent<br>State Space Model | Transformer [60]<br>LLaMA [57]<br>DeepSeek-V2 [1]<br>Mamba [21]                                                                    |
| Feedforward Network                                                                       | Up-down FFN<br>Gate-up-down FFN                                          | GPT [6]<br>LLaMA [57]                                                                                                              |
| Normalization                                                                             | RMSNorm<br>Elem-wise Norm                                                | LLaMA [57]<br>BERT [16]                                                                                                            |
| Mixture-of-Experts MoE MoE with Shared Experts                                            |                                                                          | Gshard [31], Switch Transformer [18]<br>DeepSeek-MoE [12]                                                                          |
| Data Parallelism Tensor Parallelism Pipeline Parallelism FSDP (ZeRO-3) Expert Parallelism |                                                                          | PyTorch DDP [32]<br>Megatron-LM [54]<br>GPipe [24], PipeDream [22]<br>DeepSpeed [48], PyTorch-FSDP [68]<br>Switch Transformer [18] |

within the platform, compute and communication volumes are tied to the system size, and so on). This makes it challenging to extend the ETs or perform DSE for hypothetical future platforms.

Challenge 2: Limitations of Graph Capture from ML Frameworks. Today ML frameworks enable the capture of pre-execution graphs, such as PyTorch's FX graph representation. Unfortunately, relying purely on ML frameworks to obtain workload representations is also quite restrictive. First, capturing a distributed workload's pre-execution graph still requires access to a real cluster. This also limits the degree of parallelization to the cluster size. Second, and more importantly, the dependency on frameworks limits the generated representations to the set of AI models and parallelization strategies that are supported by the frameworks. This significantly restricts co-design opportunities to existing concepts that already made it to mainstream software stacks<sup>3</sup>.

Challenge 3: Limitations of Manually Describing AI Workloads. The wide range of LLM architectures and parallelization strategies makes synthetic modeling of distributed ML workloads particularly challenging. Table II summarizes commonly used model components and parallel strategies optimized for specific training or inference objectives. Previous efforts that have performed synthetic workload generation (such as Calculon [25], MadMax [23], SimAI [62]) rely on customized templates or analytical first-order equations to describe AI workloads. While this approach has demonstrated promising performance analysis of distributed AI systems for realistic workloads, which was their goal, enabling arbitrary AI workload modeling was not. As a result, their templates are over-optimized for specific target workloads / operators and require deep understanding with the codebase for extensions. Moreover, their analytical nature limits the ability to capture realistic system and hardware behaviors, such as compute-communication dependencies.

Challenge 3.1: Diverse and Rapidly Evolving Model Architectures. Modern LLM architectures exhibit substantial diversity. For instance, LLaMA incorporates Group Query Attention

<span id="page-2-2"></span><sup>&</sup>lt;sup>3</sup>As an anecdotal example, while FSDP made sense conceptually when it was developed, it could not be evaluated until support was added in PyTorch.

(GQA) in its attention mechanism alongside a unique threelayer feed-forward network, differing substantially from traditional GPT architectures. Recent models, such as DeepSeek-R1 [\[14\]](#page-13-10), further increase complexity by employing MoE layers with shared experts and Multi-head Latent Attention (MLA). Additionally, non-transformer architectures, exemplified by Mamba [\[21\]](#page-14-4), replace conventional attention with selective state-space models. Emerging hybrid architectures combining transformer and state-space models, such as Zamba [\[19\]](#page-13-11) and Jamba [\[35\]](#page-14-11), further compound the complexity.

*Challenge 3.2: Complexity and Variability in Parallelization Strategies.* Practical deployments of LLMs often employ a hybrid mix of parallelization strategies [\(Sec. II-B\)](#page-1-2) to optimize system performance and resource utilization.

Furthermore, in practice, LLM developers rarely rely on a single model architecture or parallelization strategy. Instead, they often combine multiple design components, resulting in compositional and complex workloads that existing templated / analytical workload generators struggle to systematically represent and evaluate, highlighting a critical gap in current distributed AI workload modeling capabilities.

### <span id="page-3-1"></span>*B.* STAGE *Design Principles*

Design Principles 1: Decoupling Workload Generation from Simulation. Existing synthetic workload generators, such as Calculon, MADMAX, and SimAI, couple workload construction with performance modeling and simulation assumptions. As a result, workload structure and communication behavior are often derived from analytical models that implicitly encode system characteristics.

In contrast, STAGE treats the workload as a first-class, standalone artifact. Generation of execution graphs is independent of any specific simulators, performance model, or system topology. Simulation and profiling tools operate strictly downstream of the generated workload representation. By decoupling them, we enable: 1) reuse of the same workload across different simulators, 2) isolation of workload modeling from performance modeling assumptions, and 3) extensibility to future simulation backends, with no redundant work for each simulator backend. In the evaluation, we show that STAGE can be adapted to multiple simulators, including AstraSim [\[64\]](#page-15-5), Genie [\[66\]](#page-15-18), ScaleSim [\[53\]](#page-15-19), and SimAI [\[62\]](#page-15-17).

Design Principle 2: Decoupling Workload Semantics from System Realization. Execution traces collected from real systems inevitably embed system-specific realizations, including hardware characteristics, topology constraints, communication library implementations, compiler transformations, and framework-level optimizations. As a result, such traces are tightly bound to a particular system instantiation. Any change in topology, hardware generation, software stack, or parallel runtime typically requires re-collecting traces, making them unsuitable for systematic cross-system exploration.

STAGE instead separates workload semantics from system realization. We model distributed workloads at the level of symbolic tensor operations and parallelization semantics,

![](_page_3_Figure_8.jpeg)

<span id="page-3-0"></span>Fig. 3. STAGE Generation Flow Overview.

independent of any specific hardware platform, topology, or runtime stack.

The underlying system is abstracted as a collection of devices connected via a configurable multi-dimensional topology model. This abstraction captures communication structure without embedding hardware-specific execution artifacts. While STAGE supports optional modeling of system-specific optimizations, these are treated as extensible components rather than baked-in assumptions.

By maintaining a system-agnostic workload representation, STAGE enables a fair comparison across heterogeneous systems, while providing portability to future hardware platforms. Furthermore, it makes a separation between algorithmic operators and hardware-specific implementations, like kernel implementation or collective scheduling, which introduce transferability between systems when running simulations.

