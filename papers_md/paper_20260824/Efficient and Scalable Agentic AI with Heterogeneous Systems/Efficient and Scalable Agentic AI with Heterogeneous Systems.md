# EFFICIENT AND SCALABLE AGENTIC AI WITH HETEROGENEOUS SYSTEMS

#### A PREPRINT

Zain Asgar Stanford University, Gimlet Labs Inc. Stanford, CA zasgar@stanford.edu

Michelle Nguyen Gimlet Labs, Inc San Francisco, CA michelle@gimletlabs.ai

Sachin Katti Stanford University, Intel Stanford, CA skatti@stanford.edu

July 25, 2025

# ABSTRACT

AI agents are emerging as a dominant workload in a wide range of applications, promising to be the vehicle that delivers the promised benefits of AI to enterprises and consumers. Unlike conventional software or static inference, agentic workloads are dynamic and structurally complex. Often these agents are directed graphs of compute and IO operations that span multi-modal data input and conversion (e.g. speech to text), data processing and context gathering (e.g privacy filtering, vector DB lookups), multiple LLM inferences, tool calls, etc. To scale AI agent usage, we need efficient and scalable deployment and agent-serving infrastructure. Today, however, the vast majority of these workloads are deployed on homogenous, high-end, single-vendor infrastructure, which can often be quite expensive and limits broad rollout.

To tackle this challenge, in this paper, we present a system design for dynamic orchestration of AI agent workloads on heterogeneous compute infrastructure spanning CPUs and accelerators, both from different vendors and across different performance tiers within a single vendor. The system delivers several building blocks: a framework for planning and optimizing agentic AI execution graphs using cost models that account for compute, memory, and bandwidth constraints of different HW; a MLIR based representation and compilation system that can decompose AI agent execution graphs into granular operators and generate code for different HW options; and a dynamic orchestration system that can place the granular components across a heterogeneous compute infrastructure and stitch them together while meeting an end-to-end SLA. Our design thus performs a systems level TCO optimization and our preliminary results show that leveraging a heterogeneous infrastructure can deliver significant TCO benefits. A preliminary surprising finding is that for some workloads a heterogeneous combination of older generation GPUs with newer accelerators (H100 and Gaudi 3 respectively) can deliver similar TCO as the latest generation homogenous GPU infrastructure design (such as clusters of B200s), potentially allowing us to leverage deployed GPU infrastructure for longer periods than previously assumed.

# 1 Introduction

Agentic AI is experiencing rapid growth, with market research indicating significant adoption across various industries. Recent surveys suggest that over 75% of enterprises are actively deploying or evaluating agentic AI solutions due to their ability to augment or automate complex workflows [\[1,](#page-22-0) [2,](#page-22-1) [3\]](#page-22-2). This rapid adoption is driven by agentic AI's capability to integrate large language models (LLMs), multimodal models (e.g., speech, text, images), intricate data processing techniques, database queries, and external API integrations. Unlike traditional AI applications, which typically involve straightforward model serving scenarios, agentic AI dynamically orchestrates multiple models and heterogeneous tasks, creating complex execution patterns and interdependencies. Efficiently scaling infrastructure to support these multifaceted workloads is critical for unlocking agentic AI's full transformative potential, delivering clear operational benefits to both enterprises and consumers.

Prior work has extensively explored frameworks and techniques for building AI agents, aiming to enhance their accuracy, safety, and security. Notable examples include ReAct [\[4\]](#page-22-3), AutoGPT [\[5\]](#page-22-4), and CAMEL [\[6\]](#page-22-5). Researchers have also addressed significant risks related to security and safety, such as unpredictable user inputs and interactions with untrusted external systems [\[7\]](#page-22-6), and have proposed effective designs for multi-agent systems requiring standardized protocols [\[8,](#page-22-7) [9\]](#page-22-8). Despite these advances, research specifically focused on developing efficient and scalable infrastructure tailored for agentic AI workloads remains nascent.

While there has been limited research dedicated specifically to building scalable infrastructure for agentic AI, complementary efforts have extensively optimized individual model serving. Notable developments include kernel-level innovations like Flash Attention, significantly improving inference performance and memory efficiency for transformer models. Additionally, advanced optimization techniques such as prefill and decode disaggregation have emerged, allowing more efficient utilization of compute resources. Other significant research has explored operational efficiency through model optimization techniques like sparsity and quantization [\[10,](#page-22-9) [11\]](#page-22-10), optimized execution engines such as vLLM [\[12\]](#page-22-11), and dynamic batching methods [\[13\]](#page-22-12). However, these approaches predominantly target static inference scenarios involving a single model and thus do not adequately address the dynamic nature of agentic AI workloads. Agentic AI execution consists of multiple, varying models and data processing tasks that interact dynamically, necessitating fundamentally new systems-level approaches tailored specifically to their inherent complexity and variability. Recent studies have further highlighted that assessing agents solely on accuracy can lead to complex and expensive workloads, underscoring the importance of cost-aware benchmarks [\[14\]](#page-22-13).

A central insight of this paper is that efficiently executing agentic AI workloads requires moving beyond traditional homogeneous GPU deployments to heterogeneous systems. These heterogeneous systems are composed of accelerators across different vendors and performance tiers within a single vendor. Our analysis shows that agentic AI workloads can be decomposed into granular components, each exhibiting sensitivity to distinct hardware resource specifications such as TFLOPS, memory bandwidth and capacity, network bandwidth, disk capacity, and general-purpose compute. By aligning these granular computational tasks—such as LLM prefill, LLM decoding, data processing, and API interactions—with specifically optimized hardware capabilities, we can significantly reduce the Total Cost of Ownership (TCO) and enhance the efficiency of AI inference deployments.

However, exploiting this insight requires framing the solution space for building efficient infrastructure tailored specifically for agentic AI workloads. Effective AI infrastructure must be designed with the capability to maintain a holistic and granular view of the entire agentic pipeline, comprehensively understanding the interactions and dependencies between all its components. It must also have the flexibility to deconstruct pipelines into even more granular tasks. Furthermore, a dynamic, cost-aware orchestration layer is essential to intelligently place individual components onto suitable hardware resources, seamlessly integrate them, and optimize overall system cost. This orchestration must simultaneously ensure compliance with application-level Service Level Agreements (SLAs), such as user-perceived latency and throughput. Achieving these capabilities necessitates reimagining every layer of the AI infrastructure stack—from model runtimes and dynamic compilers to orchestration frameworks and observability solutions.

In this work, we propose a comprehensive systems-level approach designed explicitly for efficient execution of agentic AI workloads. Our contributions include:

- MLIR based Dynamic Dataflow Representation & Compilation: Modeling and compiling agentic workloads as dynamic, granular execution graphs leveraging an MLIR based toolchain, explicitly capturing the complexity and variability inherent to these workflows.
- Cost-Aware Optimization Framework: Formulating an optimization strategy that schedules and executes agentic AI workloads efficiently under practical latency, throughput, and resource constraints.

- Heterogeneous Hardware Integration: Incorporating heterogeneous hardware across different price points and vendors to leverage task-specific computational strengths, improving overall system efficiency and reducing cost.
- Dynamic AI Agent Orchestration: Developing an agentic AI orchestrator that integrates these software- and hardware-level optimizations, enabling scalable, efficient, and cost-effective deployment of agentic AI systems at enterprise scale.

# 2 Defining AI Agent Workloads

AI agents can broadly be defined as computational entities that perceive their environment, process information, and act toward a specified goal. Unlike traditional software governed by static control flow, agents operate based on data, models, and dynamic policies. Their design spans a wide spectrum, ranging from simple, rule-based systems to complex, autonomous frameworks capable of multi-step planning, memory management, and external tool integration.

#### 2.1 Autonomy and Complexity

AI agents can differ significantly in their level of autonomy and internal complexity. At one end of the spectrum are reflex based agents that follow simple, condition-based logic such as spam filters or thermostats that map input to output using predefined rules [\[15\]](#page-22-14). More advanced agents exhibit autonomy by reasoning over multiple steps and selecting actions through a closed loop of perception, decision-making, and tool use. For example, AI agents can be leveraged to answer open-ended questions with real data by dynamically querying external sources, such as Wikipedia or search APIs [\[4\]](#page-22-3).

These workflows demand richer architectural structure to effectively coordinate control flow and memory over time. As agent behaviors become more complex, systems have evolved from single agent pipelines to multi-agent architectures with distinct hierarchies. Figure [1](#page-3-0) presents a taxonomy of common agentic structures, illustrating how control can flow through single agents, coordinated peers, hierarchical layers, or fully custom graphs. For example, in a peer-to-peer agent network, multiple agents operate concurrently on different sub-tasks and collaborate by exchanging information to achieve a shared goal. In contrast, a hierarchical architecture introduces structured layers of control, where higher level agents handle planning and decision making, while lower level agents focus on specialized execution tasks.

#### 2.2 Interactivity

Another dimension of variation is interactivity, which refers to how agents engage with their environment or users throughout task execution. Interactive agents operate in a feedback loop, adjusting behavior in response to real-time inputs. These agents often expose APIs, hold state across turns, and issue clarifying queries to disambiguate user intent. This design is well-suited for customer service bots, coding copilots, or multi-step reasoning tasks. In contrast, noninteractive agents execute in a single pass, producing outputs without runtime adaptation. Such agents are commonly used for document summarization, batch inference, or fixed automation pipelines.

# 2.3 Functional Capabilities

Agents are equipped with a diverse set of functional capabilities, encompassing the concrete operations an agent can perform. Some operate in closed environments, relying solely on static model parameters to generate responses. Others can dynamically invoke tools, such as search engines, APIs, databases, or code execution environments, to augment their reasoning or access real time information. This tool-using behavior enables agents to go beyond static knowledge, making them more robust and useful in open ended or evolving environments.

### 2.4 Agents as a compute graph

To understand and optimize the behavior of AI agents, we need a representation that captures both their modular composition and dynamic control flow. Such a representation enables systematic analysis of execution dependencies, identification of bottlenecks, and opportunities for optimization and scheduling.

A natural way to express agent workloads is as a directed, potentially cyclic, graph of tasks. This graph represents the dataflow and execution dependencies between components. Each node in this graph corresponds to a discrete operation or module. These nodes are hierarchical, where the node may itself be an agent composed of further subgraphs. This allows us to represent agent workloads following the taxonomy of common agentic architectures in Figure [1.](#page-3-0)

> **[图片提取文字 (无描述)]:**
> (a) Single Agent (b) Network (c) Supervisor LLM (d) Supervisor as Tools (e) Hierarchical (f) Custom
![](_page_3_Picture_2.jpeg)

Figure 1: Comparison of agentic architectural patterns, inspired by LangGraph's taxonomy [\[16\]](#page-22-15). (a) A single LLM agent invoking external tools directly. (b) A peer-to-peer network of agents coordinating actions. (c) A supervisor agent dispatching work to multiple subordinate agents. (d) A single agent that uses another agent (e.g., a supervisor) as a tool. (e) A hierarchical architecture with clear delegation across layers, a generalized version of the supervisor pattern. (f) A custom, arbitrarily structured agent graph enabling flexible planning.

<span id="page-3-1"></span><span id="page-3-0"></span>> **[图片提取文字 (无描述)]:**
> Web search Speech to text LLM Text to speech User voice input Agent voice output
![](_page_3_Picture_4.jpeg)

Figure 2: Directed graph for a conversational voice agent

We outline the tasks of the dataflow graph in detail in Table [1,](#page-4-0) where we provide example inputs, outputs, and implementations for each node type. At a high level, these nodes include:

- Agent: A nested or composite controller with its own task graph.
- Tool Call: An external API or function invoked as part of execution.
- Model Execution: A transformer-based inference step.
- Memory: Access to external context or database.
- General Purpose Compute: Lightweight CPU-side processing for logic, parsing, or transformation.

While nodes in the agent graph represent discrete operations, the edges define the dataflow and control dependencies between those operations. An edge typically indicates that one node's output is required by another, such as text, embeddings, tool outputs, or control signals. Edges can represent either synchronous or asynchronous execution. In conditional or cyclic graphs, they may also encode feedback loops or branching behavior.

### 2.4.1 Dataflow Graph Example

To illustrate how a simple agent can be modeled as a dataflow graph, consider a conversational voice agent designed to answer user questions using web search. The graph begins with a user's spoken query as the input node. This signal is transcribed using a Speech-to-Text model and passed to a language model for processing. If the LLM determines that additional context is needed, it triggers a branch that issues web search queries to retrieve relevant information. This process may repeat until the model has enough context to generate a complete response. The final output is converted to speech using a Text-to-Speech model and returned to the user. The complete computation graph, including conditional control flow, is shown in Figure [2.](#page-3-1)

### 2.4.2 Systems-Level Optimizations of Dataflow Graphs

Representing agents as dataflow graphs provides a natural abstraction for modeling the flow of information and computation across discrete operations. This structure not only makes agent behavior more interpretable and modular but also illustrates the space for system-level optimization such as parallelism. Since many agent workloads involve multiple independent or loosely coupled operations, graph structure can reveal natural concurrency across tasks or stages of execution. Leveraging this parallelism is critical for improving latency, throughput, and resource utilization. The primary forms of parallelism exposed by agent graphs include:

- Pipeline parallelism: Decomposing a workload into sequential stages, where each stage processes a different input concurrently. This enables overlapping execution and improved throughput.
- Task parallelism: Executing independent operations or sub-tasks in parallel, often across separate threads, processes, or nodes.

A concrete instance of pipeline parallelism is disaggregated inference, where LLM execution is partitioned into prefill and decode stages and scheduled across distinct hardware resources. This staged architecture enables overlapped execution, allowing the system to initiate prefill for a new input while performing decode on a prior output.

In contrast, an example of task parallelism is expert parallelism. In expert parallelism, different model components are specialized for specific sub-tasks and invoked concurrently. Each expert operates independently on its assigned portion of the workload, enabling the system to process multiple computations simultaneously.

| Task Type               | Inputs                                  | Outputs                                 | Example                                                            |
|-------------------------|-----------------------------------------|-----------------------------------------|--------------------------------------------------------------------|
| Agent                   | Messages, context                       | Sub-tasks or output mes<br>sage         | Recursive agent graph                                              |
| Model Execution         | Token sequence                          | Token logits, hidden state              | Llama<br>[17],<br>GPT<br>[18],<br>BERT [19]                        |
| Model KV Cache          | Write from prefill; Read<br>from decode | KV tensors                              | Device-local memory, re<br>mote memory, or disaggre<br>gated store |
| Tool Call               | Query or structured input               | Tool response or data                   | API request (e.g., calcula<br>tor, search)                         |
| Memory Lookup           | Key, retrieval prompt                   | Retrieved<br>documents<br>or<br>values  | Vector DB (e.g., FAISS,<br>PGVector)                               |
| General Purpose Compute | Data blob, parameters                   | Transformed data                        | JSON<br>parsing,<br>routing<br>logic                               |
| Control Flow / Planner  | Graph state, input tokens               | Execution<br>plan<br>or<br>sub<br>graph | Agent planner module                                               |
| Observation Store       | Event, result                           | Updated memory state                    | Logging, episodic memory                                           |

<span id="page-4-0"></span>Table 1: Common Agent Task Types

# 2.5 Agent Task Workload Characteristics

Each node in an AI agent's dataflow graph exhibits distinct workload requirements across a set of key hardware dimensions. For instance, a node performing LLM inference may be GPU-bound, with high demands on memory capacity and floating-point throughput. Meanwhile, a tool invocation node is likely to be I/O-bound and dominated by network latency.

Concretely, we define these key hardware dimensions as:

- High Performance Compute: Captures the ability of specialized hardware, such as GPUs or AI accelerators, to run compute-intensive operations with high FLOP requirements.
- Memory Bandwidth: Measures the rate at which data can be read from or written to memory.
- Network Bandwidth: Captures the capacity to move data across nodes or services. High network bandwidth supports low latency communication between distributed components.
- Memory Capacity: Refers to the total available memory on a device or system.
- Disk Capacity: Measures the amount of persistent storage available.
- General Purpose Compute: The ability to process scalar CPU-based operations, such as logic, parsing, orchestration, or preprocessing.

To illustrate how system requirements vary across different AI workloads, Figure [3](#page-7-0) presents radar plots of seven representative workload profiles. Each plot highlights the dominant hardware demands across critical system dimensions. We examine each workload in detail in Table [2.](#page-6-0)

<span id="page-6-0"></span>Table 2: Representative AI Agent Workloads and System Characteristics

| Workload                        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LLM Inference (Single Node)     | This workload executes a transformer-based language model on a single ma<br>chine to generate completions. It encompasses embedding, attention, feedfor<br>ward layers, and output projection. The large matrix operations and continuous<br>access to high-dimensional weight tensors result in high demand for compute<br>and GPU memory capacity. Since execution is localized, network bandwidth<br>requirements are negligible. Disk access is limited to model loading at initial<br>ization. |
| LLM Prefill (Disaggregated)     | The prefill phase processes the full input sequence to compute hidden states and<br>populate the KV cache. It involves full attention across all tokens, requiring<br>high compute throughput. Distributed execution amplifies demands on memory<br>capacity, memory bandwidth, and network bandwidth.                                                                                                                                                                                              |
| LLM Decode (Disaggregated)      | During decode, the model generates one token per step using cached attention<br>states. Although compute intensity is lower than prefill due to reduced matrix<br>size, each step incurs frequent KV cache access.<br>This results in sustained<br>memory bandwidth usage. Depending on placement, cache accessibility also<br>imposes significant demands on memory capacity or network bandwidth.                                                                                                 |
| Diffusion Models                | Diffusion models iteratively transform noise into structured outputs over dozens<br>to hundreds of inference steps. Each step involves a complete forward pass<br>through a neural network, producing sustained and high compute utilization.<br>The repeated loading of model parameters from high-bandwidth memory to<br>on-chip SRAM across steps places sustained pressure on memory bandwidth,<br>while intermediate storage is required for activations and state.                            |
| KV Cache Storage                | The KV cache holds layer-wise attention states for reuse during decoding.<br>Long context windows increase memory footprint, while memory pressure<br>can necessitate offloading to disk. In distributed settings, remote cache access<br>introduces network overhead. Although compute is minimal, cache I/O latency<br>is critical to maintaining low end-to-end response times under concurrent load.                                                                                            |
| Tool Calls                      | Tool invocation involves calling external APIs or structured data sources. Since<br>computation occurs externally, local compute and memory usage remain low.<br>However, these steps introduce significant and variable network latency, as<br>well as high outbound bandwidth requirements. General-purpose compute is<br>needed for serializing requests, validating responses, and transforming results<br>for downstream tasks.                                                                |
| General Purpose Data Processing | This workload includes input/output formatting, control logic, and other auxil<br>iary operations. Due to their general-purpose nature, These steps typically are<br>typically executed on CPUs rather than specialized accelerators. Tasks such<br>as document merging may require maintaining large in-memory buffers. Disk<br>and network activity can arise when interfacing with external sources or storage<br>systems.                                                                       |

> **[图片提取文字 (无描述)]:**
> General Purpose Compute Blisk Capacity General Purpose Compute Disk Capacity General Purpose Compute **Bisk Capacity** High Performand Memory Capacity High Performance ry Capacity High Performance Compute Memory Capacity Nowfork Bandwidth Memory Banda ork Bandwidth Memory Bandy ork Bandwidth Memory Bandwidt (a) LLM Inference (Single Node) (b) LLM Prefill (Disaggregated) (c) LLM Decode (Disaggregated) General Purpose Compute **Bisk Capacity** General Purpose Compute Disk Capacity General Purpose Compute Disk Capacity y Capacity High Performance Compute High Performance Memory Capacity High Performance Compute Memory Capacity Compute Notwork Bandwidth Memory Bandwidtt Network Bandwidth Memory Banda Notwork Bandwidth Memory Banda (d) Diffusion Models (e) KV Cache Storage (f) Tool Calls Risk Capacity General Purpose Compute High Performance Compute Memory Capacity Memory Bandwi fork Bandwidth (g) General Purpose Data Processing
![](_page_7_Figure_2.jpeg)

<span id="page-7-0"></span>Figure 3: Radar plots comparing system resource demands across various AI workloads. Each subplot visualizes the relative importance (on a normalized scale from 0 to 10) of six key hardware dimensions: memory capacity, disk capacity, general purpose compute, high performance compute, memory bandwidth, and network bandwidth. These plots reflect qualitative estimates intended to illustrate workload characteristics, rather than results from direct performance measurements. (a) LLM Inference (Single Node) is compute- and memory-intensive but operates within a single server, reducing network demands. (b) LLM Prefill (Disaggregated) requires high memory and network bandwidth due to distributed token processing. (c) LLM Decode (Disaggregated) has lower compute demand than prefill but still exhibits high memory and network usage. (d) Diffusion Models are broadly intensive across all dimensions, especially compute and memory bandwidth. (e) KV Cache Storage emphasizes memory and disk usage, with elevated network I/O for remote retrieval. (f) Tool Calls involve low compute but higher network bandwidth for accessing external tools or APIs. (g) General Purpose Data Processing is characterized by strong general-purpose compute and balanced use of disk, memory, and bandwidth.

# 3 Design Framework for Heterogeneous Systems

Today's typical AI deployments rely heavily on racks of homogeneous GPUs interconnected via high-bandwidth, scale-up networks. These racks usually consist of identical nodes featuring high-performance GPUs such as NVIDIA's GB200 NVL72[\[20\]](#page-23-3), each offering substantial computational resources (e.g., PFLOPS), large memory bandwidth (upwards of several TB/s), and substantial networking bandwidth facilitated by high-speed interconnects like NVIDIA's NVLink[\[21\]](#page-23-4) or InfiniBand[\[22\]](#page-23-5). NVIDIA's future roadmap further emphasizes scaling up these homogeneous rack-scale GPU systems to deliver even higher throughput and compute density, particularly targeting the demands of large-scale AI workloads [\[23\]](#page-23-6).

However, the uniformity of such homogeneous systems implies a uniform, high cost per unit of resource—whether it be computational FLOPs, memory bandwidth, network bandwidth, or memory capacity—regardless of the specific task or operation. Consequently, deployments incur substantial costs even for tasks that may not require the maximum available specifications in every dimension. For example, an asynchronous AI agent that relies on long-lived external requests may not benefit from a 10% speedup in LLM inference, especially if it requires doubling infrastructure costs.

As highlighted previously, agentic AI workloads comprise diverse, dynamically orchestrated components with varying resource sensitivities. For instance, the decoding phase of large language models (LLMs) benefits significantly from greater memory capacity and bandwidth but does not demand as many computational FLOPs as the prefill stage. Similarly, data-intensive tasks might prioritize high memory bandwidth, while compute-bound operations require maximum computational power. Thus, using a homogeneous system forces a scenario where every task, irrespective of its specific resource requirements, incurs the same high uniform cost per unit of resource utilized.

In contrast, heterogeneous systems offer a compelling alternative by integrating compute nodes (both accelerators and CPUs) with varied specifications across different resource dimensions such as TFLOPS, memory bandwidth and capacity, network bandwidth, disk capacity, and CPU resources. Figure. [4](#page-9-0) shows an analysis of the cost/unit for different resources for a sample of AI HW: memory BW, TFLOPS and memory capacity for different classes and generations of NVIDIA, Intel and AMD HW. As expected, different HW offer a diverse array of cost optimization points for different resources. By adopting such heterogeneous systems, individual components of an agentic AI graph could be optimally mapped onto specific hardware nodes tailored to their unique performance requirements. Consequently, instead of incurring a uniform cost across all operations, each component could leverage the most cost-effective hardware tailored to its needs, intuitively reducing the overall cost associated with executing agentic AI workloads.

The primary challenge in exploiting this insight lies in the lack of infrastructure for systemically representing, scheduling, and executing agentic AI workloads across heterogeneous hardware. In particular, doing so requires:

- 1. A fine-grained workload model that describes agent behavior as low-level execution steps (e.g., model inference, tool use, memory lookup).
- 2. A scheduling system that can reason about cost-performance trade-offs and assign each step to the most appropriate hardware, accounting for both compute and data transfer overhead.
- 3. A heterogeneous compiler and orchestration layer that abstracts away the complexities of generating and placing code on heterogeneous nodes, seamlessly integrating tasks using appropriate networking primitives, and ensuring compliance with SLAs.

When agent workloads are represented as task graphs, the assignment process becomes a constrained optimization problem over that structure. Under empirically grounded assumptions based on hardware benchmarks, vendor specifications, and analytical models for resource usage, this reduces to a convex optimization problem—one that enables efficient and globally optimal planning across heterogeneous compute targets.

The next part formalizes this optimization framework. Subsequent parts in this section describe the systems-level mechanisms required to implement such a planner in practice — including profiling, cost modeling, and runtime execution across heterogeneous infrastructure.

#### <span id="page-8-0"></span>3.1 Formal Optimization Framework

With a structured representation of agent workloads as dataflow graphs, we can now formulate a principled optimization problem to assign tasks to heterogeneous hardware while minimizing cost and meeting performance constraints. These workloads include model execution, memory operations, retrieval and tool calls, as well as control logic — each with distinct compute and communication characteristics. In practice, this is often a multi-objective problem, where Pareto-optimal solutions must balance tradeoffs between cost, latency, energy, or other constraints.

We represent workloads as a directed graph G = (V, E), where each node i ∈ V is a task and each edge (i, k) ∈ E denotes a directed dependency. While many workloads are acyclic, our formulation supports general *directed* graphs, including those with cycles (e.g., recurrent subgraphs, feedback loops), so long as execution constraints (like bounded unrolling or check-pointing) are satisfied in runtime planning.

Each task i must be assigned to a hardware class j ∈ H (e.g., GPU tier, CPU, accelerator). The challenge is to optimize this assignment across all tasks to minimize total system cost, subject to latency, capacity, and throughput constraints. The optimization framework is flexible and supports varying objective functions and constraint formulations, enabling adaptation to different system goals and deployment scenarios.

### 3.1.1 Execution and Cost Model

Each task consumes resources θ (r) ij when executed on hardware j, where r ∈ {compute, memory, bandwidth}. Each device class j offers performance perf(r) j and has resource capacity cap (r) j . Execution time is assumed to be bottlenecked

> **[图片提取文字 (无描述)]:**
> AMD NVIDIA Intel ●H100 40 A40 35 6 30 ●B200 Cost Per Gbps Cost Per Fp16 20 15 A100 A40 A100 MI300x Gaudi3 MI300x ●H100 10 ●B200 Gaudi3 1 o<sup>C</sup> o<sup>C</sup> 1000 2000 3000 4000 5000 6000 7000 8000 1000 2000 3000 4000 Memory Bw Fp16 (a) Memory Bandwidth vs Cost per GB/s (b) FP16 Compute vs Cost per TFLOP 40 A40 300 35 250 30 ●B200 Cost Per Fp8 Gost Per 1200 1200 A100 MI300x 100 Gaudi3 10 A40 Gaudi3 H100 50 B200 5 0 0 L 75 100 125 150 2000 4000 6000 8000 25 50 175 Fp8 Memory (c) FP8 Compute vs Cost per TFLOP (d) Memory Capacity vs Cost per GB
![](_page_9_Figure_2.jpeg)

<span id="page-9-0"></span>Figure 4: Marginal cost-efficiency analysis of contemporary AI accelerators, derived from publicly available hardware specifications [24, 25, 26, 27, 28, 29, 30]. Devices are color-coded by manufacturer: NVIDIA (blue), Intel (green), and AMD (red). (a) Memory bandwidth versus cost per GB/s: Gaudi3 and MI300x exhibit the highest bandwidth efficiency. (b) FP16 compute throughput versus cost per TFLOP: H100, Gaudi3, and MI300x provide strong cost-efficiency. (c) FP8 compute throughput versus cost per TFLOP: B200 offers leading efficiency at low precision. (d) Total memory capacity versus cost per GB: MI300x and A40 deliver the most cost-effective memory provisioning.

by the slowest critical resource, and further affected by static overheads (e.g., network latency, kernel launch time) and communication costs:

$$t_{ij} = \max_{r} \left( \frac{\theta_{ij}^{(r)}}{\operatorname{perf}_{i}^{(r)}} \right) + l_{i} + d_{ij} + \delta_{ij}$$

Where:

- $l_i$ : static latency for task i,
- $d_{ij}$ : pipeline parallelism or inter-device communication cost,
- $\delta_{ij}$ : synchronization overhead from task-parallelism (e.g., all-reduce).
- $\max_r \left( \frac{\theta_{ij}^{(r)}}{\operatorname{perf}_j^{(r)}} \right)$ : latency for the slowest task in the graph.

In practice, these latency terms can be profiled from system traces, benchmarks, or prior executions, rather than analytically modeled.

The cost of executing task i on hardware j is modeled as:

$$Cost_{ij} = \sum_{r} \theta_{ij}^{(r)} \cdot c_j^{(r)} + \gamma \cdot d_{ij}$$

Where:

- c (r) j : cost per unit of resource r (e.g., per TFLOP or GB transferred),
- γ: weight on inter-device communication penalties.

### 3.1.2 Optimization Objective and Constraints

Decision Variables. Let xij ∈ [0, 1] be the fraction of task i assigned to hardware class j. In most systems, xij ∈ {0, 1}, but fractional assignment can represent workload splitting or soft allocation.

Objective. Minimize the total execution cost across all tasks:

$$\min \sum_{i \in V} \sum_{j \in H} x_{ij} \cdot \left( \sum_{r} \theta_{ij}^{(r)} \cdot c_{j}^{(r)} + \gamma \cdot d_{ij} \right) + \lambda \sum_{i \in V} s_{i}$$

The slack variable s<sup>i</sup> represents the amount by which task i's latency can exceed its SLA target. By incorporating s<sup>i</sup> in the objective function with a penalty weight λ, the optimizer is incentivized to minimize SLA violations unless they yield significant cost savings. Setting λ → ∞ enforces hard constraints.

### Constraints.

1. Assignment:

$$\sum_{j \in H} x_{ij} = 1 \quad \forall i \in V$$

2. Latency (with soft SLA):

$$t_i = \sum_j x_{ij} \cdot t_{ij}, \quad t_i - s_i \le T_{SLA}, \quad s_i \ge 0$$

3. Throughput (optional):

$$\sum_{i \in V} \frac{1}{t_i} \ge R$$

4. Hardware capacity:

$$\sum_{i \in V} x_{ij} \cdot \theta_{ij}^{(r)} \le \operatorname{cap}_{j}^{(r)} \quad \forall j, r$$

5. Feasibility:

$$x_{ij} \in [0,1] \quad \forall i, j$$

This convex formulation allows modeling heterogeneous system cost-performance tradeoffs, either through analytical resource scaling or profiled latency/cost estimates.

### Worked Example: Optimizing Prefill/Decode under SLA

As a concrete instantiation, we'll explore an example of optimizing a task node that is an LLM execution. We can further decompose the LLM execution task as a task graph G = (V, E) where:

- V = {prefill, decode},
- E = {(prefill → decode)}

We consider two device types available for running each task: HP (high performance) and CO (cost optimized). This example models a single inference request, consisting of one prefill followed by one decode phase, with no context reuse or multi-turn interaction.

We assume:

- Prefill processes 1000 input tokens,
- Decode generates 500 output tokens,
- Latency and cost per device-task pair are known from profiling. We assume these numbers are collected under optimal conditions for throughput and utilization, including maximized batching efficiency.
- There is no task splitting, i.e., each device exclusively executes its task.

| Task                  | Device Type | Latency (ms) | Cost (per token)             |
|-----------------------|-------------|--------------|------------------------------|
| Prefill               | HP          | 80           | \$0.00008                    |
| Prefill               | CO          | 130          | \$0.00005                    |
| Decode                | HP          | 25           | \$0.00006                    |
| Decode                | CO          | 30           | \$0.00002                    |
| KV Transfer (HP → CO) | —           | 10           | \$0.000005 per prefill token |

Table 3: Hypothetical latency and cost (profiled) for each task-device combination. HP represents a high-performance but expensive node, and CO represents a cost optimized option.

Let TSLA = 120 ms, which reflects a typical threshold for interactive user experiences (e.g., web search or chatbot responses). We evaluate valid assignments:

### Option A: Prefill and decode on HP

$$t = 80 + 25 = 105 \text{ ms}, \quad \text{SLA satisfied}$$
 
$$\text{Cost} = 1000 \cdot 0.00008 + 500 \cdot 0.00006 = \boxed{\$0.11}$$

#### Option B: Prefill on HP, decode on CO

$$t = 80 + 30 + 10 = 120 \text{ ms}, \quad \text{SLA satisfied}$$
 
$$\text{Cost} = 1000 \cdot 0.00008 + 500 \cdot 0.00002 + 1000 \cdot 0.000005 = \boxed{\$0.095}$$

Option C: All on CO

$$t = 130 + 30 = 160 \text{ ms}, \quad \text{SLA violated}$$
 
$$\text{Cost} = 1000 \cdot 0.00005 + 500 \cdot 0.00002 = \boxed{\$0.07}$$

Optimization Result: Given t<sup>i</sup> ≤ TSLA, Option C is infeasible. Option B achieves lower cost than Option A while satisfying latency constraints. Thus, the optimal assignment is:

$$x_{\text{prefill},HP} = 1, \quad x_{\text{decode},CO} = 1$$

This decision reflects a fundamental tradeoff: while high-performance devices may offer lower latency, they incur a higher cost per token. Strategic disaggregation enables more efficient resource allocation, reducing overall cost while meeting strict latency requirements.

Further, this example demonstrates how the optimization leverages device heterogeneity to minimize cost without violating SLAs. In larger graphs, this formulation generalizes to multi-node pipelines, branching agent flows, and cyclic controller-executor patterns — using profiled characteristics or analytical estimates for runtime planning. This extends beyond model execution tasks and includes workloads composed of tool calls, external APIs, and data processing.

# 4 System Design

Building upon the cost optimization insights derived from our earlier framework in Section [3.1,](#page-8-0) we propose a comprehensive system architecture designed to facilitate flexible, efficient, and dynamic inference across heterogeneous hardware. The system takes as input a dataflow graph, dynamically schedules its execution according to optimal cost configurations computed via our convex optimization approach, and supports diverse AI agent workloads, including dynamic control flows, chained models, and external tool invocations. Moreover, it remains adaptable to evolving hardware landscapes comprising both high and low-end accelerators from vendors such as NVIDIA, Intel, and AMD.

#### 4.1 Orchestration and Serving System

> **[图片提取文字 (无描述)]:**
> Inference Serving System Planner & Scheduler API Server Load Balancer / Cache Aware Request Router Compatible with OpenAl, etc. Runtime Runtime Runtime Memory Memory Memory. Model Model Model Management Management Management Execution Execution Execution Serving Metrics Metrics Metrics (Open Telemetry) Nodes Collector Collector Collector KV Cache KV Cache KV Cache Subgraph Subgraph Subgraph Execution Execution Execution High Performance Interconnect - Ethernet RoCE Object Storage Cache Management (Models & KV S)
![](_page_12_Figure_3.jpeg)

<span id="page-12-0"></span>Figure 5: High-level orchestration and serving system architecture

Figure [5](#page-12-0) illustrates the high-level design of our orchestration and serving system. The architecture prioritizes several key objectives:

- 1. Scalability: Automatically scales agentic workloads across heterogeneous hardware resources based on load and utilization.
- 2. Flexibility: Supports both synchronous workloads activated externally via APIs and asynchronous workloads operating autonomously.
- 3. Composability: Facilitates multi-turn interactions activated through repeated API calls or system state changes.

The core functionality of our orchestration system involves dynamically planning and placing fine-grained computational components onto a distributed fleet of hardware. It continuously monitors node availability, workload characteristics, and resource utilization to inform placement decisions. This dynamic allocation helps prevent resource contention and optimizes both throughput and cost efficiency.

At a system level, the orchestration is designed to be deployed within distributed cluster environments, such as Kubernetes, employing a separation between a slow-path responsible for planning and resource allocation, and a fast-path for immediate execution.

The orchestration architecture includes the following primary components:

- Planner & Scheduler (Slow Path): Continuously monitors hardware resources and workloads, dynamically allocating tasks based on the optimization strategies outlined in Section [3.1.](#page-8-0) This component handles workload migration, resource allocation, and planning.
- Load Balancer / Request Router (Fast Path): Routes requests based on cache locality and model availability, optimizing resource utilization and request aggregation for performance.
- Runtime: Deployed to each accelerator node, the runtime encapsulates the core execution environment responsible for handling workloads from the scheduler. It implements mechanisms for model and subgraph execution, KV cache and memory management, and metrics collection. It is designed to run across heterogeneous environments by providing an abstraction to device specific capabilities.
- RDMA Transport Layer: Utilizes a high-performance Ethernet fabric with RDMA over Converged Ethernet (RoCE) to facilitate efficient transfer of models and caches. Abstraction layers and open standards ensure interoperability and optimal performance.
- Cache Manager: Manages distributed key-value (KV) caches, graph databases, and agent memory storage, employing strategies for offloading less frequently accessed data to slower storage mediums such as secondary memory tiers, disks, or object storage.

For efficient execution across nodes, our system explicitly accounts for network topology and interconnect performance metrics, including latency, bandwidth, and potential contention. The orchestration system optimizes data communication patterns, particularly leveraging RoCE for workloads that require minimal latency between computational stages. Since KV cache placement significantly influences data movement, the cache management system actively coordinates distributed KV cache locations to minimize overheads associated with data transfers.

Running agentic graphs across heterogeneous environments presents unique challenges. These graphs are often not constructed with hardware diversity in mind, and naïvely executing them on heterogeneous systems can lead to inefficiencies or execution failures. To support intelligent scheduling and robust execution across varied accelerators, we must rethink how these workloads are represented and structured.

We propose the following components to address these requirements, as depicted in Figure [6:](#page-13-0)

- Dataflow representation using MLIR dialect: A MLIR dialect to represent the various components of an agentic AI graph to enable a systematic HW agnostic representation and enable planning.
- Dataflow orchestration system: A scheduler that can analyze the workload, available hardware, and dynamically decide how to allocate granular fragments across hardware nodes.
- Dataflow compiler: A flexible compiler capable of taking high level workloads, partitioning and translating nodes and operations into optimized kernels for different hardware backends.

#### 4.2 MLIR as a Foundation for Heterogeneous Agent Execution

To support fine-grained analysis and optimization of agentic workloads across heterogeneous systems, we represent each workload as a program graph encoded in an intermediate representation. Specifically, we adopt the *Multi-Level Intermediate Representation (MLIR)* framework [\[31\]](#page-23-14), which provides a rich, extensible infrastructure for expressing and transforming computations across abstraction levels.

> **[图片提取文字 (无描述)]:**
> Device Compiler Deployment Scheduler Specific Assets Task Planner / **Profile Information** Resource Optimizer (Performance & Usage) Information Compute MLIR Representation Graph
![](_page_13_Figure_10.jpeg)

<span id="page-13-0"></span>Figure 6: System design stack from MLIR representation through task planning and compilation to deployment

Figure [6](#page-13-0) shows how MLIR acts as a bridge between the raw compute graph and deployment. Workloads are transformed through dialect-based intermediate representations, optimized using both static analysis and runtime resource feedback, and ultimately compiled and scheduled for execution across a heterogeneous set of backends.

Each task in the agent workload graph, ranging from LLM execution and memory access to tool calls and control logic, is assigned to an MLIR operation or subgraph. These operations are structured into dialects, which can be tailored to specific semantics (e.g., LLM inference, key-value caching, external tool APIs), enabling both domain-specific reasoning and cross-layer optimization.

#### MLIR for Agentic Workload Planning

The use of MLIR brings several benefits to system design and execution planning:

- 1. Compositional Representation: Tasks can be hierarchically composed, allowing for encapsulation of complex behavior (e.g., an agent that delegates to sub-agents or models).
- 2. Fusion and Decomposition: Using MLIR transformations, adjacent or dependent operations can be fused to reduce communication overhead, or decomposed to enable distributed execution.
- 3. Static Analysis for Scheduling: MLIR supports passes for bufferization, shape inference, cost estimation, and dependency analysis. These passes enable extraction of resource usage vectors θ (r) ij and latency terms tij , which feed directly into the convex optimization framework and scheduler.
- 4. Extensibility Across Modalities: New dialects can capture novel task types (e.g., retrieval-augmented generation, multi-modal decoding), making the framework future-proof and adaptable to evolving agent designs.
- 5. Target-Aware Lowering: Once optimized, MLIR graphs can be lowered into device-specific IRs such as LLVM IR [\[32\]](#page-23-15), TensorRT [\[33\]](#page-23-16), or XLA HLO [\[34\]](#page-23-17), or alternate backends such as TVM [\[35\]](#page-23-18), IREE [\[36\]](#page-23-19), or Glow [\[37\]](#page-23-20), facilitating backend compilation for CPUs, GPUs, NPUs, and other accelerators.

#### Agent frameworks to MLIR

Modern agent frameworks such as LangChain allow users to express complex workflows using a high-level orchestration interface. These workflows typically compose memory, tool invocations, and language model calls in an imperative style. To optimize and schedule such workflows over heterogeneous hardware, we require a structured representation that exposes task boundaries, data dependencies, and operation semantics.

To optimize high-level agent workflows over heterogeneous infrastructure, the system must first lower these workflows into structured, semantically rich intermediate representations. Figure [7](#page-15-0) illustrates this process. At the top, a LangChainstyle orchestration defines an agent with memory and two tools—Search() and Calculator()—invoked in response to a user query. While simple to author, such imperative code lacks the structural annotations required for effective scheduling and device mapping.

The high-level MLIR representation (bottom left) introduces a typed dataflow graph over operations such as memory access, LLM invocation, and tool usage. On the bottom right, we show a decomposed variant that exposes internal parallelism within the model execution, enabling hardware-aware optimization. Specifically, we model a hybrid parallelism strategy that combines expert parallelism—where only a sparse subset of experts are activated per token—with tensor parallelism within each expert.

This is made explicit via a gate.select operation that routes input tokens to top-k experts. Each expert is then executed in parallel using expert.tp.prefill and expert.tp.decode, indicating a tensor-parallel subgraph per expert.

The MLIR representations below illustrate how such a workflow can be lowered into a structured, graph-based intermediate form. On the left, a high-level MLIR encoding captures typed operations such as memory load/store, LLM invocation, and tool usage. On the right, the same graph is decomposed into finer-grained operations: the LLM call is split into prefill and decode, and each tool invocation is separated into a lookup and a compute stage. This transformation reveals internal parallelism and resource requirements, enabling the compiler to reason about scheduling, placement, and pipelining across a heterogeneous system.

Each operation can be annotated with profiling metadata, resource usage estimates, or placement hints. A system pass then transforms this high-level IR into an annotated task graph ready for convex optimization.

#### Towards Heterogeneous-Aware Compilers

The MLIR-based representation serves as a bridge between high-level workload semantics and low-level scheduling decisions. Just as traditional compilers optimize instruction placement and register allocation, this system-level compiler optimizes task placement and inter-device orchestration.

Crucially, such transformations are not limited to static programs. With extensions to support dynamic execution, asynchronous control flow, and runtime cost feedback, MLIR provides a foundation for compiling not just neural networks but distributed agentic systems that interact with tools, memory, and other agents in real time.

> **[图片提取文字 (无描述)]:**
> (c) MLIR Representation (Decomposed + Hy-(a) LangChain Orchestration (High-Level) brid Parallelism) response = Agent( agent.graph { tools=[Search(), Calculator()], %0 = input.query("Summarize the memory=ConversationBuffer() compute needs of AI over the ).run("Summarize the compute needs of next 5 years.") AI over the next 5 years.") %1 = memory.load(%0)// Expert routing and tensorparallel prefill %2 = gate.select(%1) %3 = expert.tp.prefill(%1, %2) %4 = merge.expert\_outputs(%3) (b) MLIR Representation (High-Level) %5 = allreduce(%4)// Expert routing and tensoragent.graph { parallel decode %0 = input.query("Summarize the %6 = gate.select(%5)compute needs of AI over the %7 = expert.tp.decode(%5, %6) next 5 years.") %8 = merge.expert\_outputs(%7) %1 = memory.load(%0) %9 = allreduce(%8) %2 = 11m.call(%1)%3 = tool.invoke(search, %2) %10 = tool.lookup(search, %9) %4 = tool.invoke(calculator, %3) %11 = tool.compute(calculator, %10) %5 = memory.store(%4) %12 = memory.store(%11) return %5 return %12
![](_page_15_Figure_2.jpeg)

<span id="page-15-0"></span>Figure 7: Transformation of a LangChain-style agent program into progressively lower-level MLIR representations. Panel (a) shows the original orchestration logic, while panels (b) and (c) illustrate how a compiler can lower this workflow into high-level and decomposed MLIR forms, enabling hybrid parallelism and heterogeneous hardware scheduling.

In this section, we have shown how we can start with a high-level agent definition that a developer might interact with, transform that into MLIR, and then use that MLIR to perform high-level optimizations. This optmized graph can then be scheduled using resource and run-time information across distributed infrastructure. Further, there is a compiler system that can take these MLIR operators and lower them to hardware-specific frameworks. Overall, this system allows for optimized deployment of entire agent workloads across a diverse set of hardware.

# 5 Preliminary Results

To assess the effectiveness of our system, we conducted an evaluation using a representative conversational voice agent composed of modular components: speech-to-text, text-to-speech, web search, and a central LLM node, as depicted in Figure [2.](#page-3-1) These findings are preliminary, and comprehensive system validation is currently underway. Table [5](#page-16-0) summarizes the accelerator hardware included in our current evaluation.

Our optimization framework places the non-LLM components of the voice agent on CPUs given the task characteristic (relatively computationally light) and the relative cost of a CPU, hence the dominant factor impacting overall TCO is the LLM component which is the most computationally demanding part. As a result, the following focuses on exploring optimizations on the LLM component. For the LLM, we evaluated four configurations of the LLaMA 3 model: 8B and 70B parameter sizes, each in FP16 and FP8 precisions (see Table [4\)](#page-16-1). Computational and memory demands were profiled based on model size, sequence lengths, and architectural details as an input to the optimization framework. Device-specific performance metrics, such as latency and throughput, incorporate empirical measurements when available and are augmented by theoretical roofline modeling [\[38\]](#page-23-21) to represent realistic performance boundaries. All reported FLOP values assume dense computation, without accounting for sparsity.

To precisely isolate scheduling and hardware allocation benefits, we simulated a continuous workload scenario with unconstrained hardware availability. We evaluate which heterogeneous configuration leads to the maximum throughput (tokens/sec by maximizing batch size) under two different scenarios with SLAs that correspond to interactive and offline usage scenarios:

- Latency SLA (Interactive workloads): Time-to-First-Token (TTFT) 250 ms, Token-to-Token (TBT) 20 ms.
- Throughput SLA (Offline workloads): Maximize tokens/s/\$.

| Model                | Parameters (B) | Precision | Source       |
|----------------------|----------------|-----------|--------------|
| LLaMA 3 - 8B - FP16  | 8              | FP16      | Meta AI [39] |
| LLaMA 3 - 8B - FP8   | 8              | FP8       | Meta AI [39] |
| LLaMA 3 - 70B - FP16 | 70             | FP16      | Meta AI [39] |
| LLaMA 3 - 70B - FP8  | 70             | FP8       | Meta AI [39] |

<span id="page-16-1"></span>Table 4: Model configurations used in evaluation.

| Device | Manufacturer | Cost (\$) | Memory (GB) | Bandwidth (GB/s) | TFLOPs (FP16) | Operating Cost (\$/hr) |
|--------|--------------|-----------|-------------|------------------|---------------|------------------------|
| A40    | NVIDIA       | \$3,000   | 48          | 696              | 75            | \$0.15                 |
| A100   | NVIDIA       | \$8,000   | 80          | 2039             | 322           | \$0.25                 |
| Gaudi3 | Intel        | \$12,500  | 128         | 3700             | 1678          | \$0.49                 |
| MI300x | AMD          | \$20,000  | 192         | 5300             | 1307          | \$0.52                 |
| H100   | NVIDIA       | \$25,000  | 80          | 3350             | 1979          | \$0.60                 |
| B200   | NVIDIA       | \$40,000  | 192         | 8000             | 2250          | \$0.83                 |

<span id="page-16-0"></span>Table 5: Specifications of accelerator hardware used in the optimizer. Costs averaged across a representative sample of hardware resellers available in public listings as of June 2025.

> **[图片提取文字 (无描述)]:**
> Latency SLA Throughput SLA 4.0 3.5 3.0 2.5 2.0 2.0 1.5 H100 H100 1.0 0.5 0.0 8200:9audi3 8200:H100 H100:9audi3 8200:8200 Device Pair Device Pair Model Llama 3 - 8B - FP8 Llama 3 - 70B - FP8 Llama 3 - 8B - FP16 Llama 3 - 70B - FP16
![](_page_16_Figure_9.jpeg)

TCO Benefit for Heterogeneous Configs (input=512, output=4096)

<span id="page-16-2"></span>Figure 8: TCO Benefit for Heterogeneous Configs (input=512, output=4096). Comparison of cost efficiency across different Llama 3 models and device pairings. Dashed line at 1.0 indicates baseline TCO for H100::H100. Bars show top configurations that meet SLA constraints: Latency SLA (TTFT  $\leq$  250ms, TBT  $\leq$  20ms) and Throughput SLA (Maximize tokens/s/\$). Results are based on a performance model fit to real measurements and explore heterogeneous configurations that leverage both tensor parallelism and pipeline parallelism with disaggregated inference.

#### 5.1 TCO of Heterogeneous Systems

The evaluated accelerator hardware specifications are detailed in Table 5, covering GPUs and ASIC accelerators from multiple vendors to demonstrate the broad applicability of our framework. The operating cost model assumes that hardware is financed over a fixed amortization period of 4 years with an interest rate of 8%. For utility costs, we assume

Latency SLA

Throughput SLA

Throughput SLA

Throughput SLA

Throughput SLA

Throughput SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Latency SLA

Throughput SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

TCO Benefit for Heterogeneous Configs (input=4096, output=512)

<span id="page-17-0"></span>Figure 9: **TCO Benefit for Heterogeneous Configs (input=4096, output=512).** Comparison of cost efficiency across different Llama 3 models and device pairings. Dashed line at 1.0 indicates baseline TCO for H100::H100. Bars reflect top-performing configurations that satisfy SLA constraints: Latency SLA (TTFT  $\leq$  250ms, TBT  $\leq$  20ms) and Throughput SLA (Maximize tokens/s/\$.). These results are derived from a performance model calibrated to hardware measurements, incorporating both tensor parallelism and pipeline parallelism under disaggregated inference.

each node operates at its maximum rated TDP, with a cost of \$0.40/kWh. Other operational expenses, such as datacenter or colocation fees and nonrecurring engineering (NRE) costs, are excluded from the operating cost. Additionally, to describe the heterogeneous configurations, we leverage the operator "::" as a notation to denote disaggregated inference. The left and right operands correspond to the hardware configurations used during the prefill and decode stages, respectively.

Figures 8, and 9 demonstrate the TCO improvements achievable through heterogeneous hardware configurations compared against the homogeneous baseline configuration (H100::H100). We focus on 6 possible combinations of hardware pairs, as they best illustrate the variations between performance and cost. We evaluated two scenarios for input-output sequence lengths, corresponding to reasoning tasks (long intermediate/output token sizes) and summarization tasks (short output sequence length). For each configuration, the system automatically explores options and selects the best combination of tensor and pipeline parallelism based on the available network bandwidth (both scale up and scale out) for that configuration and the latency SLA. Initial increases in tensor parallelism substantially reduced latency; however, further increases introduced significant device-to-device communication overhead, negating the computational efficiency gains. Additionally, our framework automatically incorporates optimizations such as paged attention [12], further enhancing the efficiency of execution.

There are two interesting observations from the results.

- **B200::Gaudi 3** has the best overall TCO benefit, especially for FP8 model configurations, for both interactive as well as batch workloads. The benefits are present (albeit smaller) even compared to a B200::B200 baseline which is the latest generation system.
- H100::Gaudi 3 configuration is often comparable or slightly better than a B200::B200 configuration, implying that the Gaudi 3 can effectively complement the H100 and overall the heterogeneous configuration can deliver compelling performance, reducing the need to upgrade to Blackwell. The benefits are likely even higher if we incorporate the depreciation of the Hopper GPUs that have already been partially amortized, which is outside the scope of this paper.

#### 5.2 Deployment requirements and considerations

One of the central challenges in deploying workloads across distributed systems lies in managing the bandwidth and latency constraints imposed by the interconnect fabric linking accelerators. These interconnects are typically categorized as *scale-up* or *scale-out* fabrics. *Scale-up fabrics* aim to deliver high-bandwidth, low-latency connections with shared memory semantics across multiple accelerators within a single system, as exemplified by NVLink-based designs such as NVL72 [20]. In contrast, *scale-out fabrics* rely on commodity networking technologies such as Ethernet

and InfiniBand [\[22\]](#page-23-5), enabling the interconnection of large-scale clusters without shared memory, thereby requiring explicit software coordination for data movement.

In our system design, we assume that scale-up fabrics are confined to a single chassis, typically supporting up to 8 accelerators. Beyond this, we rely on high-speed *RDMA over Converged Ethernet (RoCE)* [\[40\]](#page-24-0), which is commonly deployed in modern large-scale AI datacenters [\[41\]](#page-24-1).

We utilize the underlying fabric for two primary purposes:

- 1. Inter-node parallelism: Distributing computation across multiple machines (for example tensor parallelism)
- 2. State transfer across pipeline stages: Moving shared runtime state between nodes (for example key-value (KV) caches, during prefill/decode disaggregation.)

Both inter-node parallelism and state transfer are incorporated into our total cost of ownership (TCO) model. The scalability of inter-node parallelism is constrained by the efficiency of data movement between accelerators, while state transfer primarily affects the end-to-end latency of the deployed agent.

Importantly, state transfer latency can often be partially amortized by overlapping communication with computation. For example, in prefill/decode disaggregation, key-value (KV) cache transfers contribute to the latency of the *second token*, as the cache must be transmitted from the prefill stage to the decode stage. Fortunately, the bandwidth demands of this transfer are typically well-supported by modern AI datacenter networks [\[42\]](#page-24-2). For completeness, we present the high-level bandwidth model that can be used to model the minimum bandwidth required to allow non-blocking pipelining of disaggregated inference:

$$BW_{\text{PeakEgress}} = \frac{\text{KV Cache Size}}{\text{TTFT} \cdot N_{PrefillGPU}} \tag{1}$$

$$BW_{\text{PeakIngress}} = \frac{\text{KV Cache Size}}{\text{TBT} \cdot N_{DeocodeGPU}}$$
 (2)

It is important to note that the above equations represent the *peak* bandwidth required to transfer a single KV cache instance. In practice, inference systems often operate on batched inputs, which linearly scales the effective KV cache size and, correspondingly, the peak bandwidth requirement.

However, if the primary concern is overall task completion time—as is common in batch-oriented workloads—then it is more appropriate to consider *amortized* bandwidth.

For practical workloads, we can estimate the peak bandwidth required based on the KV cache size and compute time. We compute the size of the key-value (KV) cache required for transformer-based models such as LLaMA using the following expression:

$$\text{KVCacheSize}_{\text{peak}} = 2 \cdot N_{\text{layers}} \cdot d_{\text{model}} \cdot \left(\frac{N_{\text{kv}}}{N_{\text{heads}}}\right) \cdot \text{ISL} \cdot BS \cdot \text{BPE} \tag{3}$$

### Legend:

• Nlayers: Number of transformer layers

• dmodel: Hidden dimension of the model

• Nkv: Number of key/value heads

• Nheads: Total number of attention heads

• ISL: Input sequence length (tokens)

• BS: Batch size

• BPE: Bytes per element (e.g., 2 for FP16)

Using the derived expressions, we observe that a 200–400 Gbps link is sufficient to meet the SLA requirements for transferring KV caches for input sequence lengths up to 32K tokens, depending on the specific LLaMA model variant employed. Such high-bandwidth interconnects are commonly available in modern high-performance AI datacenters.

While our TCO model incorporates a detailed treatment of networking latency and cost, we find that practical provisioning of interconnect bandwidth is generally sufficient to mitigate performance bottlenecks. Moreover, as noted by [\[42\]](#page-24-2), increases in model and context size can actually reduce bandwidth requirements in practice. For instance, total time for first token (TTFT) tends to grow superlinearly with input sequence length (ISL), whereas the KV cache size grows only linearly. Similarly, while decode latency depends on the number of decoding GPUs, the corresponding ingress bandwidth requirement decreases inversely. Additionally, recent models with more efficient attention mechanisms—such as Multi-Linear Attention (MLA)—require smaller KV cache sizes [\[43\]](#page-24-3), further reducing pressure on interconnect bandwidth.

#### 5.3 Analysis

To understand the above results, we explored how our optimization framework is making decisions on which parts of the voice agent workload are placed on which hardware. For example, the hardware allocations of different LLM inference stages (prefill and decode) are quite distinct given their different computational needs (prefill is computationally intensive whereas decode is more memory capacity intensive). Our framework inherently accommodates such optimizations by decomposing the LLM workload into granular components, enabling hardware resources to be matched precisely with operational demands.

Optimal hardware configurations varied significantly depending on input sequence length and decode tokens. For longer input sequences (Figure [9\)](#page-17-0), Intel Gaudi 3 accelerators emerged as the most cost-effective choice for prefill tasks due to their superior cost-performance ratio relative to NVIDIA B200. Conversely, when latency or FP8 performance is the primary concern, the higher computational power of the B200 justified its selection despite higher associated costs.

In decode-intensive scenarios (Figure [8\)](#page-16-2), Gaudi3 accelerators were selected for decode tasks due to their lowest marginal cost, as indicated in Figure [4,](#page-9-0) assuming the workload can accommodate slightly longer token-to-token latency. Conversely, the B200 provides the best overall performance at an increased cost but remains relatively efficient compared to previous-generation systems such as the H100.

In conclusion, our optimization framework effectively leverages the diverse performance characteristics of heterogeneous hardware resources, dynamically allocating workloads based on specific SLA requirements. This adaptability enables optimal utilization of hardware capabilities, ensuring both cost efficiency and performance responsiveness tailored to individual requests.

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

# 7 Future Work

This work opens numerous promising avenues for future research, particularly focused on expanding the capabilities and robustness of agentic workloads executed across heterogeneous computing environments. We identify two key directions below:

#### 7.1 Distributed Datacenter Scheduling and Optimization

A crucial future research area involves developing sophisticated scheduling frameworks tailored for distributed datacenter environments. These environments typically span a range of capabilities, from smaller, geographically dispersed datacenters—such as enterprise-level facilities offering proximity and lower latency—to large-scale hyperscaler datacenters that provide significant computational power. Efficiently orchestrating agentic workloads across these heterogeneous datacenters requires advanced scheduling techniques that dynamically balance latency, resource availability, compute intensity, and cost considerations. Methods including hierarchical scheduling, intelligent workload migration, and dynamic load balancing [\[54,](#page-24-14) [55\]](#page-24-15) could be explored to ensure optimal resource utilization and robust performance under varying conditions.

### 7.2 Cross-Device Agent Planning (Cloud and Edge)

Another important direction is extending our optimization framework to support seamless execution and planning of agentic tasks across hybrid cloud-edge deployments. Such cross-device planning must account for unique constraints like varying network latencies, data locality considerations, bandwidth variability, and heterogeneous compute capabilities between edge devices and cloud infrastructure. Recent protocols like Minion and MinionS [\[56\]](#page-25-0) demonstrate practical benefits of decomposing and parallelizing tasks between local and cloud language models, significantly reducing costs while preserving accuracy. Formalizing and generalizing these approaches into comprehensive optimization frameworks will enable rigorous guarantees and broader applicability. Developing efficient algorithms that can dynamically adapt task distribution in response to changing conditions (such as network disruptions, fluctuating workload characteristics, and evolving hardware availability) presents a compelling and non-trivial challenge. Research into these adaptive cross-device strategies will greatly enhance the applicability and resilience of distributed agent workloads in real-world deployments.

#### 7.3 Enabling AI Cloud Marketplace

The orchestration system described in this paper can be built upon to enable the creation of a comprehensive AI cloud Platform-as-a-Service (PaaS) marketplace. Unlike proprietary AI marketplace offerings such as NVIDIA Lepton, our approach democratizes AI infrastructure access, allowing entities with spare AI compute capacity of whichever HW to participate actively in the marketplace. Such a marketplace would allow third-party developers and infrastructure providers to offer optimized, modular AI components and workflows, each precisely mapped onto cost-effective hardware configurations. By abstracting hardware complexities, the orchestration system facilitates seamless integration and dynamic scaling, enabling enterprises and individuals to easily discover, purchase, and deploy specialized AI services tailored to their specific requirements, thereby accelerating innovation and broadening AI adoption across diverse market segments.

# 8 Conclusion

This paper describes the motivation for as well as the design and implementation of a system for delivering efficient and scalable agentic AI over heterogeneous infrastructure. We believe that the AI revolution is still in its infancy, and systems innovation that can enable a distributed, scale-out AI infrastructure is critical to enable cost-effective scaling of AI. Our work is a first step in that direction.

# Acknowledgments

We thank James Bartlett for his contributions to early brainstorming and technical discussions. We are also grateful to Taras Sereda, Natalie Serrino, and Omid Azizi for their valuable feedback during the review process and for their thoughtful discussions. This work was supported in part by compute resources provided by Gimlet Labs, Inc.

# References

- <span id="page-22-0"></span>[1] Mike Vizard. Survey surfaces rapid adoption of agentic AI. [https://techstrong.ai/agentic-ai/survey](https://techstrong.ai/agentic-ai/survey-surfaces-rapid-adoption-of-agentic-ai/) [-surfaces-rapid-adoption-of-agentic-ai/](https://techstrong.ai/agentic-ai/survey-surfaces-rapid-adoption-of-agentic-ai/), May 2025. Accessed 2025-06-15.
- <span id="page-22-1"></span>[2] Cloudera Press Release. 96% of enterprises are expanding use of AI agents. [https://www.cloudera.com](https://www.cloudera.com/about/news-and-blogs/press-releases/2025-04-16-96-percent-of-enterprises-are-expanding-use-of-ai-agents-according-to-latest-data-from-cloudera.html) [/about/news-and-blogs/press-releases/2025-04-16-96-percent-of-enterprises-are-expan](https://www.cloudera.com/about/news-and-blogs/press-releases/2025-04-16-96-percent-of-enterprises-are-expanding-use-of-ai-agents-according-to-latest-data-from-cloudera.html) [ding-use-of-ai-agents-according-to-latest-data-from-cloudera.html](https://www.cloudera.com/about/news-and-blogs/press-releases/2025-04-16-96-percent-of-enterprises-are-expanding-use-of-ai-agents-according-to-latest-data-from-cloudera.html), April 2025. Accessed 2025-06-15.
- <span id="page-22-2"></span>[3] PwC. PwC's AI agent survey. [https://www.pwc.com/us/en/tech-effect/ai-analytics/ai-agent-s](https://www.pwc.com/us/en/tech-effect/ai-analytics/ai-agent-survey.html) [urvey.html](https://www.pwc.com/us/en/tech-effect/ai-analytics/ai-agent-survey.html), May 2025. Accessed 2025-06-15.
- <span id="page-22-3"></span>[4] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models, 2023.
- <span id="page-22-4"></span>[5] Hui Yang, Sifu Yue, and Yunzhong He. Auto-GPT for online decision making: Benchmarks and additional opinions, 2023.
- <span id="page-22-5"></span>[6] Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. CAMEL: Communicative agents for "mind" exploration of large language model society, 2023.
- <span id="page-22-6"></span>[7] Zehang Deng, Yongjian Guo, Changzhou Han, Wanlun Ma, Junwu Xiong, Sheng Wen, and Yang Xiang. AI agents under threat: A survey of key security challenges and future pathways, 2024.
- <span id="page-22-7"></span>[8] Yufan Dang, Chen Qian, Xueheng Luo, Jingru Fan, Zihao Xie, Ruijie Shi, Weize Chen, Cheng Yang, Xiaoyin Che, Ye Tian, Xuantang Xiong, Lei Han, Zhiyuan Liu, and Maosong Sun. Multi-agent collaboration via evolving orchestration, 2025.
- <span id="page-22-8"></span>[9] Yingxuan Yang, Huacan Chai, Yuanyi Song, Siyuan Qi, Muning Wen, Ning Li, Junwei Liao, Haoyi Hu, Jianghao Lin, Gaowei Chang, Weiwen Liu, Ying Wen, Yong Yu, and Weinan Zhang. A survey of AI agent protocols, 2025.
- <span id="page-22-9"></span>[10] Haizhong Zheng, Xiaoyan Bai, Xueshen Liu, Z. Morley Mao, Beidi Chen, Fan Lai, and Atul Prakash. Learn to be efficient: Build structured sparsity in large language models, 2024.
- <span id="page-22-10"></span>[11] Shiyao Li, Xuefei Ning, Luning Wang, Tengxuan Liu, Xiangsheng Shi, Shengen Yan, Guohao Dai, Huazhong Yang, and Yu Wang. Evaluating quantized large language models, 2024.
- <span id="page-22-11"></span>[12] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention, 2023.
- <span id="page-22-12"></span>[13] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. Efficient large-scale language model training on GPU clusters. *CoRR*, abs/2104.04473, 2021.
- <span id="page-22-13"></span>[14] Sayash Kapoor, Benedikt Stroebl, Zachary S. Siegel, Nitya Nadgir, and Arvind Narayanan. AI agents that matter, 2024.
- <span id="page-22-14"></span>[15] S.J. Russell and P. Norvig. *Artificial Intelligence: A Modern Approach*. Prentice Hall international editions. Prentice Hall, 1995.
- <span id="page-22-15"></span>[16] LangGraph Contributors. Multi-agent architectures in LangGraph. [https://langchain-ai.github.io/lang](https://langchain-ai.github.io/langgraph/concepts/multi_agent/) [graph/concepts/multi\\_agent/](https://langchain-ai.github.io/langgraph/concepts/multi_agent/), 2024. Accessed: 2025-06-14.

- <span id="page-23-0"></span>[17] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: Open and efficient foundation language models, 2023.
- <span id="page-23-1"></span>[18] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018. Technical report.
- <span id="page-23-2"></span>[19] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding, 2019.
- <span id="page-23-3"></span>[20] NVIDIA. NVIDIA Blackwell Datasheet. Datasheet 3384703, NVIDIA, 2024. Dec 2024.
- <span id="page-23-4"></span>[21] NVIDIA. NVLink & NVSwitch: Fastest HPC Data Center Platform. [https://www.nvidia.com/en-us/data](https://www.nvidia.com/en-us/data-center/nvlink/) [-center/nvlink/](https://www.nvidia.com/en-us/data-center/nvlink/), 2025. Accessed: 2025-06-15.
- <span id="page-23-5"></span>[22] NVIDIA. NVIDIA InfiniBand (Ethernet & InfiniBand Networking Products). [https://www.nvidia.com/e](https://www.nvidia.com/en-us/networking/products/infiniband/) [n-us/networking/products/infiniband/](https://www.nvidia.com/en-us/networking/products/infiniband/), 2025. Accessed: 2025-06-15.
- <span id="page-23-6"></span>[23] NVIDIA. NVIDIA 800V HVDC Architecture Will Power the Next Generation of AI Factories. [https:](https://developer.nvidia.com/blog/nvidia-800-v-hvdc-architecture-will-power-the-next-generation-of-ai-factories/) [//developer.nvidia.com/blog/nvidia-800-v-hvdc-architecture-will-power-the-next-gener](https://developer.nvidia.com/blog/nvidia-800-v-hvdc-architecture-will-power-the-next-generation-of-ai-factories/) [ation-of-ai-factories/](https://developer.nvidia.com/blog/nvidia-800-v-hvdc-architecture-will-power-the-next-generation-of-ai-factories/), May 2025. Accessed: 2025-06-15.
- <span id="page-23-7"></span>[24] NVIDIA. NVIDIA A40 datasheet. [https://images.nvidia.com/content/Solutions/data-center/a4](https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf) [0/nvidia-a40-datasheet.pdf](https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf). Accessed: 2025-06-14.
- <span id="page-23-8"></span>[25] NVIDIA. NVIDIA A100 datasheet. [https://www.nvidia.com/content/dam/en-zz/Solutions/Data-C](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) [enter/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf). Accessed: 2025-06-14.
- <span id="page-23-9"></span>[26] NVIDIA. NVIDIA H100 datasheet. [https://resources.nvidia.com/en-us-gpu-resources/h100-dat](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306) [asheet-24306](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306). Accessed: 2025-06-14.
- <span id="page-23-10"></span>[27] NVIDIA. NVIDIA B200 datasheet. [https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-d](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet) [atasheet](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet). Accessed: 2025-06-14.
- <span id="page-23-11"></span>[28] Intel. Intel Gaudi 3 AI accelerator white paper. [https://cdrdv2-public.intel.com/817486/gaudi-3-a](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf) [i-accelerator-white-paper.pdf](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf). Accessed: 2025-06-14.
- <span id="page-23-12"></span>[29] AMD Instinct MI300X accelerator. [https://www.amd.com/content/dam/amd/en/documents/instinc](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf) [t-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf). Accessed: 2025-07-23.
- <span id="page-23-13"></span>[30] NVIDIA RTX 6000 Ada Generation. [https://resources.nvidia.com/en-us-briefcase-for-datashe](https://resources.nvidia.com/en-us-briefcase-for-datasheets/proviz-print-rtx6000-1) [ets/proviz-print-rtx6000-1](https://resources.nvidia.com/en-us-briefcase-for-datasheets/proviz-print-rtx6000-1). Accessed: 2025-07-23.
- <span id="page-23-14"></span>[31] Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, and Oleksandr Zinenko. MLIR: Scaling compiler infrastructure for domain specific computation. In *2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)*, pages 2–14. IEEE, 2021.
- <span id="page-23-15"></span>[32] LLVM Project. LLVM: A Compilation Framework for Lifelong Program Analysis and Transformation. [https:](https://llvm.org) [//llvm.org](https://llvm.org), 2023. https://llvm.org.
- <span id="page-23-16"></span>[33] NVIDIA Corporation. TensorRT. [h t t p s : / / d e v e l o p e r . n v i d i a . c o m / t e n s o r r t](https://developer.nvidia.com/tensorrt), 2023. https://developer.nvidia.com/tensorrt.
- <span id="page-23-17"></span>[34] Google Research. XLA: Optimizing compiler for machine learning. <https://www.tensorflow.org/xla>, 2023. https://www.tensorflow.org/xla.
- <span id="page-23-18"></span>[35] TVM Contributors. Apache TVM: An open deep learning compiler stack. <https://tvm.apache.org>, 2023. https://tvm.apache.org.
- <span id="page-23-19"></span>[36] OpenXLA Contributors. IREE: Intermediate representation execution environment. [https://openxla.org/pr](https://openxla.org/projects/iree) [ojects/iree](https://openxla.org/projects/iree), 2023. https://openxla.org/projects/iree.
- <span id="page-23-20"></span>[37] Facebook AI Research. Glow: Graph lowering neural network compiler. <https://github.com/pytorch/glow>, 2023. https://github.com/pytorch/glow.
- <span id="page-23-21"></span>[38] Samuel Williams, Andrew Waterman, and David Patterson. Roofline: an insightful visual performance model for multicore architectures. *Communications of the ACM*, 52(4):65–76, 2009.
- <span id="page-23-22"></span>[39] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.

- <span id="page-24-0"></span>[40] InfiniBand Trade Association. RoCE Initiative. <https://www.roceinitiative.org/about-overview/>, 2025. https://www.roceinitiative.org/about-overview/.
- <span id="page-24-1"></span>[41] Adithya Gangidi, Rui Miao, Shengbao Zheng, Sai Jayesh Bondu, Guilherme Goes, Hany Morsy, Rohit Puri, Mohammad Riftadi, Ashmitha Jeevaraj Shetty, Jingyi Yang, Shuqiang Zhang, Mikel Jimenez Fernandez, Shashidhar Gandham, and Hongyi Zeng. Rdma over ethernet for distributed training at meta scale. In *Proceedings of the ACM SIGCOMM 2024 Conference*, ACM SIGCOMM '24, page 57–70, New York, NY, USA, 2024. Association for Computing Machinery.
- <span id="page-24-2"></span>[42] Tiyasa Mitra, Ritika Borkar, Nidhi Bhatia, Ramon Matas, Shivam Raj, Dheevatsa Mudigere, Ritchie Zhao, Maximilian Golub, Arpan Dutta, Sailaja Madduri, et al. Beyond the buzz: A pragmatic take on inference disaggregation. *arXiv preprint arXiv:2506.05508*, 2025.
- <span id="page-24-3"></span>[43] DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang Yuan, Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe Pan, Runxin Xu, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Size Zheng, T. Wang, Tian Pei, Tian Yuan, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Liu, Xin Xie, Xingkai Yu, Xinnan Song, Xinyi Zhou, Xinyu Yang, Xuan Lu, Xuecheng Su, Y. Wu, Y. K. Li, Y. X. Wei, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Zheng, Yichao Zhang, Yiliang Xiong, Yilong Zhao, Ying He, Ying Tang, Yishi Piao, Yixin Dong, Yixuan Tan, Yiyuan Liu, Yongji Wang, Yongqiang Guo, Yuchen Zhu, Yuduan Wang, Yuheng Zou, Yukun Zha, Yunxian Ma, Yuting Yan, Yuxiang You, Yuxuan Liu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhewen Hao, Zhihong Shao, Zhiniu Wen, Zhipeng Xu, Zhongyu Zhang, Zhuoshu Li, Zihan Wang, Zihui Gu, Zilin Li, and Ziwei Xie. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024.
- <span id="page-24-4"></span>[44] NVIDIA. TensorRT-LLM. <https://github.com/NVIDIA/TensorRT-LLM>, 2023. Accessed: 2025-06-15.
- <span id="page-24-5"></span>[45] SGLang Contributors. SGLang: Serving LLMs with programmatic prompts and high throughput. [https:](https://github.com/InternLM/sglang) [//github.com/InternLM/sglang](https://github.com/InternLM/sglang), 2024. Accessed: 2025-06-15.
- <span id="page-24-6"></span>[46] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini. Splitwise: Efficient generative llm inference using phase splitting. In *2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)*, pages 118–132. IEEE, 2024.
- <span id="page-24-7"></span>[47] NVIDIA. Dynamo: A datacenter-scale distributed inference serving framework. [https://github.com/ai-d](https://github.com/ai-dynamo/dynamo) [ynamo/dynamo](https://github.com/ai-dynamo/dynamo), 2025. Accessed: 2025-06-15.
- <span id="page-24-8"></span>[48] llm-d community. llm-d: A kubernetes-native high-performance distributed LLM inference framework. [https:](https://llm-d.ai/) [//llm-d.ai/](https://llm-d.ai/), 2025. Accessed: 2025-06-15.
- <span id="page-24-9"></span>[49] IREE Team. IREE: Intermediate representation execution environment. <https://github.com/openxla/iree>, 2022. Accessed: 2025-06-15.
- <span id="page-24-10"></span>[50] MHLO Contributors. MHLO: MLIR HLO dialect for tensorflow and JAX. [https://github.com/openxla/m](https://github.com/openxla/mlir-hlo) [lir-hlo](https://github.com/openxla/mlir-hlo), 2021. Accessed: 2025-06-15.
- <span id="page-24-11"></span>[51] Philippe Tillet, Hyeontaek Johnson, and Christos Kozyrakis. Triton: An intermediate language and compiler for tiled neural network computations. In *Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)*, page 111–125. ACM, 2022.
- <span id="page-24-12"></span>[52] CrewAI Team. CrewAI: Orchestrate role-playing AI agents with memory and tools. [https://github.com/joa](https://github.com/joaomdmoura/crewAI) [omdmoura/crewAI](https://github.com/joaomdmoura/crewAI), 2024. Accessed: 2025-06-15.
- <span id="page-24-13"></span>[53] Yizhou Wu, Ziniu Song, Haotian Li, et al. AutoGen: Enabling next-gen LLM applications via multi-agent conversations. <https://microsoft.github.io/autogen>, 2023. Microsoft Research, Accessed: 2025-06-15.
- <span id="page-24-14"></span>[54] Jeffrey Dean and Luiz André Barroso. The tail at scale. *Communications of the ACM*, 56(2):74–80, 2013.
- <span id="page-24-15"></span>[55] Kai Chen, Hong Hu, Yongqiang Chen, and Jianfei Bai. Distributed scheduling and optimization in datacenters. *IEEE Communications Magazine*, 56(5):84–89, 2018.

<span id="page-25-0"></span>[56] Avanika Narayan, Dan Biderman, Sabri Eyuboglu, Avner May, Scott Linderman, James Zou, and Christopher Re. Minions: Cost-efficient collaboration between on-device and cloud language models, 2025.