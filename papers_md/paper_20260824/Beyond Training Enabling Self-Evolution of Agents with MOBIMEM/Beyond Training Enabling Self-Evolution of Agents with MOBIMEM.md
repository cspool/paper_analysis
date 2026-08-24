# Beyond Training: Enabling Self-Evolution of Agents with MOBIMEM

Zibin Liu<sup>1†</sup>, Cheng Zhang<sup>1†</sup>, Xi Zhao<sup>1†</sup>, Yunfei Feng<sup>†</sup>, Bingyu Bai<sup>†</sup>, Dahu Feng<sup>†</sup>, Erhu Feng<sup>®†</sup>, Yubin Xia<sup>†</sup>, Haibo Chen <sup>†</sup>

<sup>†</sup>Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University

### **Abstract**

Large Language Model (LLM) agents are increasingly deployed to automate complex workflows in mobile and desktop environments. However, current *model-centric* agent architectures struggle to self-evolve post-deployment: improving personalization, capability, and efficiency typically requires continuous model retraining/fine-tuning, which incurs prohibitive computational overheads and suffers from an inherent trade-off between model accuracy and inference efficiency.

To enable iterative self-evolution without model retraining, we propose MOBIMEM, a memory-centric agent system. MOBIMEM first introduces three specialized memory primitives to decouple agent evolution from model weights: (1) Profile Memory uses a lightweight distance-graph (Dis-Graph) structure to align with user preferences, resolving the accuracy-latency trade-off in user profile retrieval; (2) Experience Memory employs multi-level templates to instantiate execution logic for new tasks, ensuring capability generalization; and (3) Action Memory records fine-grained interaction sequences, reducing the reliance on expensive model inference. Building upon this memory architecture, MOBIMEM further integrates a suite of OS-inspired services to orchestrate execution: a scheduler that coordinates parallel sub-task execution and memory operations; an agent record-and-replay (AgentRR) mechanism that enables safe and efficient action reuse; and a context-aware exception handling that ensures graceful recovery from user interruptions and runtime errors.

Evaluation on AndroidWorld and top-50 apps shows that MOBIMEM achieves 83.1% profile alignment with 23.83 ms retrieval time (280× faster than GraphRAG baselines), improves task success rates by up to 50.3%, and reduces end-to-end latency by up to  $9\times$  on mobile devices, demonstrating the efficiency and practicality of memory-centric evolution in real-world deployments.

> **[图片提取文字 (无描述)]:**
> Ours DroidRun GBOX (Gemini 2.5 MobileUse-(Claude Pro) GUI-Owl V2 30B Sonnet 4.5) **7B** Accuracy Gemini 2.5 GUI-Owl UI-TARS-2 Qwen3 VL Computer 32B 72B 30B A3B Gemini Seed-1.5-Use 2.5 Flash **VL 20B UI-TARS UI-Venus-**-1.5 7B Navi 72B End-to-End Latency Small Medium Extra-Large Large Scale Scale Scale Scale
![](_page_0_Figure_9.jpeg)

<span id="page-0-0"></span>Figure 1: MOBIMEM tames the trade-off between AI agents' latency and accuracy by a memory-centric design.

#### 1 Introduction

The rapid advancement of large language models (LLMs) has catalyzed the emergence and proliferation of AI agents, which are capable of autonomously executing complex tasks through natural language understanding and tool manipulation. As AI agents [18, 55, 67] evolve, they are increasingly deployed in mobile and desktop environments to automate user workflows [26,42,43,70], spanning from simple information retrieval to multi-step, cross-application tasks that require coordinating operations across diverse apps and services. For example, agents may help users book hotels by comparing prices across multiple travel apps, schedule meetings by coordinating calendar and communication apps, etc.

Although contemporary AI agents demonstrate autonomous planning, decision-making, and execution capabilities, they often lack the capacity for continual evolution during deployment. A self-evolving agent should aim to achieve three objectives: (1) continually learn user preferences, (2) continually expand its capabilities, and (3) continually improve its execution efficiency. Given that current agents are predominantly built with **model-centric** architectures, such evolution is achieved through per-user model fine-tuning [9, 23, 25, 37, 63, 74] or reinforcement learning [11, 14, 20, 39, 51, 65], which incurs prohibitive computational costs for continual training both on-device and in the

<sup>&</sup>lt;sup>1</sup>The three authors contributed equally to this work and should be considered co-first authors.

<sup>&</sup>lt;sup>™</sup>Erhu Feng is the corresponding author: fengerhu1@sjtu.edu.cn

cloud. Moreover, relying solely on model-based approaches makes it difficult to balance the agent's performance and accuracy, as shown in Figure [1.](#page-0-0) For example, a powerful agent system commonly requires multimodal models with tens to hundreds of billions of parameters. Such models are not only difficult to deploy on edge devices but also introduce substantial latency in cloud deployment, thereby limiting their practical utility in real-world scenarios.

To avoid these post-deployment training overheads, we propose a memory-centric agent system: MOBIMEM, which supports agent self-evolution without the requirement for additional model training. MOBIMEM draws inspiration from classical OS mechanisms, including memory management, scheduling, record-and-replay, etc., and adapts them to agentic scenarios. Through fine-grained memory updating, caching, and orchestration, MOBIMEM supports continuous improvement in personalization, capability, and efficiency.

MOBIMEM organizes agent memory into three types: Profile Memory, Experience Memory, and Action Memory.

- *Profile Memory:* used to learn user preferences, enabling the agent to continually personalize its behavior. Although existing studies [\[10,](#page-12-5) [45,](#page-14-5) [56\]](#page-15-5) have attempted to capture user preferences through techniques such as RAG [\[32\]](#page-13-4), GraphRAG [\[13\]](#page-12-6), and hierarchical memory [\[45\]](#page-14-5), they still face a trade-off between retrieval accuracy and efficiency. RAG enables efficient information extraction through vector search but lacks explicit relationships between entities. GraphRAG provides richer relational information, but retrieval and update require model inference, introducing non-negligible latency. To address this, we design the Dis-Graph, which shifts semantic information from edges to nodes, while edges solely encode relational distances between nodes. It preserves profile accuracy while reducing the latency of retrieval and update operations.
- *Experience Memory:* storing experience templates, enabling the agent to continuously improve its capability. Prior work [\[58,](#page-15-6) [59\]](#page-15-7) only records the agent's execution trajectories as experience, but this approach compromises the agent's generalization. To address this limitation, MO-BIMEM introduces a multi-level, multi-step experience template. When a new task arrives, MOBIMEM inherits relevant templates and instantiates the execution steps required for the current task, ensuring that the experience remains applicable across diverse scenarios.
- *Action Memory:* recording the interactions between the agent and the system, allowing the agent to continually improve its execution efficiency. Similar to the procedural or "muscle" memory that humans develop through repeated practice, the agent can directly replay the action sequences stored in action memory for common tasks, without relying on model-based reasoning. To address issues analogous to cache staleness, MOBIMEM introduces a lightweight mechanism to determine whether an action memory entry is a valid hit (reusable) or to handle cases where the action

memory has become stale.

With the support of agent memory, MOBIMEM further provides system-level agent services, including the fine-grained task scheduler, agent-level record-and-replay mechanism, agent interrupt and exception handler, etc. Leveraging the experience memory, MOBIMEM establishes fine-grained dependencies at the subtask level, enabling independent subtasks to execute in parallel. Through the action memory, MOBIMEM offers an agent-level record-and-replay capability (AgentRR). Unlike system-level record-and-replay services [\[2,](#page-12-7) [12,](#page-12-8) [21,](#page-13-5) [30\]](#page-13-6), AgentRR does not rigidly reproduce previous actions; instead, it preserves the agent's generalization ability during replay. For interruption and exception handling, MOBIMEM maintains the context of the agent's execution process and integrates the corresponding exception handlers into its experience memory, ensuring that subsequent executions are not disrupted by the same exception.

We evaluate our system through extensive experiments on AndroidWorld [\[49\]](#page-14-6) benchmark and real-world workloads (including top 50 mobile Apps) across diverse hardware platforms. Profile Memory achieves 83.1% profile alignment with 23.83 ms retrieval latency, outperforming baseline approaches by up to 25% and achieving over 280× speedup compared to GraphRAG methods. Experience Memory improves task success rates by up to 50.3% across four agent models, with near-zero human effort through automated abstraction. Action Memory achieves 77.3% average action reuse rate with human-crafted templates, reducing end-to-end latency by up to 9× on resource-constrained mobile devices. In multi-task scenarios, our fine-grained task scheduling achieves up to 1.98× speedup by exploiting parallelism across independent sub-tasks. Moreover, Experience Memory and AgentRR technologies have already been deployed in a flagship smartphone.

## 2 Background

## 2.1 Mobile Agent Frameworks

Recent advances in reasoning paradigms [\[66,](#page-15-8) [73\]](#page-15-9) and GUI parsing tools [\[69\]](#page-15-10) enable GUI agent frameworks to adopt iterative execution paradigms where agents perceive screenshots, generate reasoning and actions autonomously. UI-TARS [\[4\]](#page-12-9) implements an online trace bootstrapping framework where the model dynamically learns from past task executions through continual training. Mobile-Agent-v3(MA3) [\[67\]](#page-15-0) introduces GUI-Owl as a foundational model. The framework maintains a compressed history storing interaction traces as the execution memory of the ongoing task. It also leverages an RAG module storing external or user-specific knowledge and a Reflector Agent for error diagnosis and recovery. AutoDroid [\[59\]](#page-15-7) employs an App Memory module built in an offline stage to store simulated task traces which will be used to synthesize execution guidelines, or be fully reused

<span id="page-2-0"></span>

| Table 1: Comparison of GUI agent frameworks and memory systems across memory architecture and system capabilities |
|-------------------------------------------------------------------------------------------------------------------|
|-------------------------------------------------------------------------------------------------------------------|

|                | Memory Architecture          |                         |                      | System Capabilities         |                         |                       |
|----------------|------------------------------|-------------------------|----------------------|-----------------------------|-------------------------|-----------------------|
| System         | User Profile<br>Memory       | Execution<br>Memory     | Action<br>Memory     | Task<br>Scheduling          | Error<br>Recovery       | User<br>Interrupt     |
| UI-TARS [4]    | Х                            | Training-time data      | X                    | Sequential                  | X                       | Х                     |
| MA3 [67]       | RAG module                   | Compressed histories    | X                    | Sequential                  | LLM Reflector           | X                     |
| AutoDroid [59] | X                            | Simulated traces        | Task-level           | Sequential                  | ×                       | X                     |
| MemGPT [45]    | Profile blocks (Key-value)   | Х                       | Х                    | -                           | -                       | -                     |
| Mem0 [10]      | User facts (Graph)           | X                       | X                    | -                           | -                       | -                     |
| A-MEM [64]     | General notes (Graph)        | ×                       | ×                    | -                           | -                       | -                     |
| Our Work       | Concept-Entity<br>(DisGraph) | Experience<br>Templates | Tree/Chain<br>Memory | Fine-grained<br>Parallelism | Iterative<br>Refinement | Exception<br>Handling |

for similar tasks during online deployment. Related frameworks explore variants with hierarchical planning [57], short-cut learning [27], lifelong learning [54], or broader OS-level scope [62,71].

While these frameworks demonstrate advances in execution memory and error handling (Table 1), they share fundamental limitations in memory architecture and system capabilities. On the memory side, they lack structured user profile memory for personalization and fine-grained action memory to fully exploit shared actions, while their execution memories store raw traces rather than distilled templates. On the system side, they employ sequential task scheduling without exploiting parallelism opportunities, and they do not support user interventions to correct agent errors during execution. Our work addresses these gaps through a comprehensive memory architecture that captures user profiles, execution experiences, and reusable action patterns, coupled with system capabilities including fine-grained parallelism, iterative refinement for error recovery, and exception handling mechanisms that enable user intervention and correction.

### 2.2 Memory Systems in AI Agents

Memory in agent architectures can be categorized into shortterm memory, which maintains immediate context within the model's input window, and long-term memory, which uses external storage to persist information across sessions [50,61]. Recent research focuses on enhancing long-term memory capabilities through mechanisms including indexing and retrieval [28, 41], dynamic updating and forgetting [7, 77], and memory-enhanced generation [46]. Building on these foundations, several memory systems enhance agent capabilities beyond single-task execution. MemGPT [45] introduces a hierarchical memory management system inspired by OS virtual memory, organizing information into main context, external context, and archival storage with self-editing capabilities through key-value structures. Mem0 [10] focuses on automatic fact extraction and deduplication, merging conversational statements into a consolidated graph structure to maintain user-specific facts across interactions. A-MEM [64]

emphasizes agentic self-organization of knowledge, where agents autonomously generate comprehensive memory notes with rich metadata and dynamically establish inter-memory links to form an evolving knowledge network. Other systems explore diverse architectures including human-like memory [24], memo-based mechanisms [40], self-controlled frameworks [53], unified memory architectures [34], and hybrid multimodal memory [33].

However, as shown in Table 1, these memory systems are designed for general conversational contexts rather than GUI automation domains. They excel at storing declarative knowledge (user facts, general notes) but lack the specialized memory structures needed for GUI agents. Specifically, they do not maintain execution memory to capture procedural knowledge from past interactions or provide action memory to cache reusable interaction patterns and templates. Our work optimizes memory architecture for GUI agents through Concept-Entity graphs that enable efficient multi-dimensional user profile retrieval, experience templates that distill execution traces, and Action Memory structures that cache reusable action sequences.

### 3 System Overview

Existing AI-agent systems still face challenges in continually evolving during deployment, such as achieving agent personalization and improving agent capability and efficiency without continual model training. To address these issues, we design MOBIMEM (Figure 2) with three hierarchical layers: a multi-agent layer, an agent memory layer, and an OS integration layer that provides agent services and coordinates system components.

Multi-agent Layer. Our system collaborates with four specialized agents that work together to execute user tasks. The *Profile Updater* processes OS metadata to extract and update user profiles into Profile Memory. The *Experience Generator* distills reusable task templates from execution histories and stores them in Experience Memory. The *Task Rewriter* matches incoming user requests to existing templates and fills template parameters for execution. The *Operator* gen-

> **[图片提取文字 (无描述)]:**
> Agent Layer Profile Updater Experience Generator Task Rewriter Operator § 4.2 Experience Memory § 4.3 Action Memory § 4.1 Profile Memory Profile Retrieval Experience Retrieval ActChain ActTree Evolution Engine Engine Update Template 1: Hotel App N Memory App 1 Layer Step 1 Step 2 Booking App Profile DisGraph Storage Experience Storage 2.1 Ξ Flight Ticketing Reused Concepts Actions 1. Enter "Flight" section Template 2: New 2. Select {departure} Entities Actions 3. Select (destination) OS Integration Agent Service Metadata Execution Perception Layer Provision Service Service § 5.1 § 5.2 § 5.3 Agent Exception **Agent Record** Agent XWF 0 Handler Scheduler and Replay
![](_page_3_Figure_0.jpeg)

<span id="page-3-0"></span>Figure 2: Three-layer architecture of MOBIMEM: specialized multi-agent layer (top), agent-tailored memory layer (middle), and OS integration layer (bottom).

erates actions when action reuse is not applicable or when encountering unrecorded situations. These agents leverage our memory system to access persistent context and reusable actions, reducing redundant reasoning.

Memory Layer. Our memory system offers three core modules (§4.1-4.3) to provide personalization and improve task success rates and execution efficiency. Profile Memory Module (§4.1) addresses the lack of personalization by organizing user preferences, facts, and behavior patterns in a DisGraph structure, which requires only one LLM call for updates and zero-LLM retrievals through embedding-based search and graph traversal. Experience Memory Module (§4.2) distills shared execution patterns by decomposing task execution into invariant control logic and variable parameters and storing them as experience templates that the Task Rewriter Agent can utilize. Action Memory Module (§4.3) reduces LLM inference overhead by caching historical actions in two structures: ActTree for prefix reuse and ActChain for prefix-suffix reuse. MOBIMEM also designs a fallback mechanism for Action Memory to handle cache-miss scenarios.

OS Integration Layer. Our system is built on an OS with first-class agent support, providing three categories of agent service: Agent Scheduler (§5.1) that orchestrates parallel subtasks according to the fine-grained experience memory, AgentRR (§5.2), a record-and-replay mechanism that captures execution traces for cache population and enables safe action reuse, and an Agent Exception Handler (§5.3) that enables graceful recovery from user interruptions, and records each exception handler in the experience memory. In addition to these agent services, MOBIMEM also provides agents with system-level perception and execution runtime services, including: metadata provision (e.g., screenshots, voice recordings) enables agents to analyze and learn user habits; percep-

tion service (e.g., UI hierarchy, element visibility and semantics) helps agents comprehend system and application state; execution service provides programmatic OS-level APIs for agents to perform UI interactions and event monitoring.

### <span id="page-3-2"></span>4 Agent Memory

This section presents the detailed designs of three types of agent memory: Profile Memory, Experience Memory, and Action Memory. We first describe how Profile Memory organizes a user's personalized information, including facts, preferences, and behavior patterns (§4.1). Next, we introduce how Experience Memory extracts and applies generalizable task-execution templates at multiple levels (§4.2). Finally, we explain how Action Memory accelerates task execution through action reuse (§4.3).

### <span id="page-3-1"></span>4.1 Profile Memory

Challenges. As AI agents interact with users over time, they gradually accumulate rich personalized information, including factual details, preferences, and habitual behaviors. To deliver truly personalized services, an agent must continuously interpret and retain the user's evolving preferences. Current agent systems attempt to support long-term memory through RAG-based techniques, but they still face inherent trade-offs between performance and accuracy. Simple RAG approaches offer fast retrieval but suffer from low accuracy, as embedding-based search often returns semantically similar but irrelevant information. Graph-based RAG systems improve accuracy but incur prohibitive latency, requiring expensive LLM calls for both graph updates and query traversal.

> **[图片提取文字 (无描述)]:**
> General Knowledge / Retrieval Executed Tasks Tasks Task Graph Structure Profile Profile **□**F9 ፚፚፚ OS Rewriter Updater Graph Retrieved Metadata Agent Storage Agent Updates **Entities** DisGraph Storage **Entity Updates** Profile Retrieval Let's go on a vacation. Object: Hotel Hotel .... Lodging ... Campsite Also buy some foods? price range: 150-200\$ a b Embedding Search Travel Announcement I just got back from a Retrieval Nodes Update trip. This one cost me Food Travel Memory 250\$ and Super 8 Food Communication Graph Walk Object Hotel was terrible. Collection Breakfast Shopping .. Suitcase Retrieval Entities Object: Hotel Breakfast price\_range: 150-250\$ Campsite blacklist brands: ["Super 8"] Hotel Dress Smartphone Fragrance User Message Object KV Storage Profile Update Concept Profile Retrieval Entity Retrieval Task | Retrieved Entities
![](_page_4_Figure_0.jpeg)

<span id="page-4-1"></span>Figure 3: Architecture of the Profile Memory module, showing profile updating, storage, and retrieval workflows.

**Key Insight.** Our analysis reveals that this trade-off stems from how structural relationships are represented. Simple RAG's flat structure lacks relational context, causing embedding-based retrieval to fetch semantically similar but contextually inappropriate information. GraphRAG improves accuracy by encoding rich relational semantics in edge weights, but this design necessitates expensive LLM calls to generate and traverse these semantic edges. Our key insight is to use a DisGraph architecture that shifts semantic information from edges to nodes. The graph comprises two types of nodes: abstract concepts and concrete entities, where entities connect only to concepts and concepts interconnect with each other. All edges in the graph are semantic-free, only indicating membership or relevance. Under this design, the relationship between any two nodes is correlated with the distance between them, with shorter paths generally indicating higher relevance. During retrieval, we only leverage embedding similarity to identify the most relevant starting nodes, then perform breadth-first search to expand to nodes within short path distances, gathering related profile information. Therefore, the entire retrieval process does not require any LLM involvement.

**Our Design.** As shown in Figure 3, the Profile Memory module operates through two workflows: an update workflow where Profile Updater Agent processes OS metadata and task traces to extract the profile information into the DisGraph, and a retrieval workflow where incoming tasks trigger embedding-based search and BFS traversal to gather relevant context. The storage organizes profile knowledge in the DisGraph architecture, with entities storing structured key-value pairs capturing user information. Operations on DisGraph mainly include three primitives: updating, retrieval, and splitting.

*Update logic.* When new observations arrive, the system first performs embedding-based retrieval to identify relevant nodes in the DisGraph. It then sends these retrieved nodes along with their outgoing edges and the new observations to the Profile Updater in a single LLM call. The agent outputs

> **[图片提取文字 (无描述)]:**
> Final task Experience Task Rewriter description Generator History Refined / New Exp. Retrieve Embedding Experience Vector Database Model Store Indexing Human Experience Template High-level Experiences Low-level Experiences Key: Flight Ticketing Key: Food ordering 1.1. click search bar 1.2. input {shop} Steps: Steps: 1.3. click "search" 1. Enter "Flight" section Search for {shop} Select {departure} Search for {food} 3.1. click (food) Select {destination} 3. Add (food) to cart 3.2. click "add to cart" 4. Select {date} 4. Pay for order 3.3. click "confirm"
![](_page_4_Figure_5.jpeg)

<span id="page-4-2"></span>Figure 4: Experience template example showing template structure and experience generation/retrieval workflows.

structured modifications (updates to existing entities or insertions of new entities), applied atomically to evolve the profile. Figure 3 illustrates how DisGraph updates the attributes of entities after obtaining a user's hotel reviews.

Retrieval logic. When a new task arrives, the system uses an embedding model to score the task description against all entities and concepts in the graph, selecting the top-k most relevant nodes as starting points. The system then performs breadth-first search from these starting points to expand the context. To balance coverage and relevance, the system partitions the profile context window into k equal shares, and uses round-robin scheduling to fill each share with content discovered from each starting point. Figure 3 illustrates an example in which the task "Let's go on a vacation this weekend. Also buy some foods?" triggers an embedding search that identifies Travel and Food as the starting nodes and subsequently collects content related to these entities.

Dynamic splitting. As the user profile expands, the number of entity nodes linked to a popular concept node increases, which in turn reduces retrieval precision. The BFS expansion retrieves an excessive number of entities under the same concept, which dilutes the effective context window. To address this, when a concept's entity count exceeds a threshold, the system invokes the Profile Updater to analyze entity attributes and create specialized subconcepts (e.g., splitting Travel into Business Travel and Leisure Travel), then redistributes entities to their most relevant subconcepts.

#### <span id="page-4-0"></span>4.2 Experience Memory

Challenges. Current approaches enhance agent capabilities primarily through training on domain-specific data. However, such training-based approaches face inherent scalability challenges in real-world deployment. They require large volumes of high-quality execution traces, which are costly to collect, leaving most long-tail tasks insufficiently covered. In addition, model training demands substantial computational resources. On-device training on endpoints such as smartphones remains

impractical, while performing cloud-based training for each user's long-tail tasks also incurs significant cost.

Key Insight. Rather than training the model to understand complex pages, learn page-transition relationships, and perform task planning, it is more effective to directly provide the model with correct execution experiences, thereby reducing its reasoning burden. However, due to the complexity of task descriptions and environments, providing experiences for every task-environment combination is also not scalable. To address this, we propose the *multi-level experience* and *experience template*. For experience templates, we observe that similar tasks share invariant control flow but have variant data flow. This enables us to abstract execution patterns into templates with parameter slots. For any concrete task, the agent inherits the corresponding template and instantiates parameter slots with task-specific values to produce an executable experience. Moreover, multi-level experience strikes a balance between generality and specificity. High-level experience generalizes to a wider range of scenarios but demands greater capability from the model, whereas low-level experience presents the opposite trade-off.

Our Design. Based on this insight, we design an Experience Memory module that enables agents to accumulate, abstract, and reapply execution knowledge. It consists of two core components: (1) an *Experience Store* that maintains multi-level experience templates, and (2) a *Vector Database* that enables fast semantic retrieval. As shown in Figure [4,](#page-4-2) higher-level experiences describe task-level control flow, while lower-level experiences provide concrete execution steps for precise navigation and interaction (e.g., click, swipe, etc.). More specifically, the Experience Memory module provides two core mechanisms:

*Experience generation and storage.* Experience templates are automatically synthesized when encountering new task classes, or manually authored by developers to bootstrap common tasks. For automatic synthesis, when a task cannot find a matching template during retrieval, the Experience Generator synthesizes a new template by referencing similar past experiences. For manual authoring, developers can directly author parameterized workflows. All templates are keyed by their core descriptions, indexed using an embedding model and stored in the vector database for fast retrieval.

*Template retrieval and parameter filling.* When a new task arrives, the system queries the vector database to retrieve the best-matching template based on the semantic similarity between the task description and existing template keys. The retrieved template is passed to the Task Rewriter to fill the parameter slots using information extracted from the current task. If no suitable experience template is found, the agent autonomously decides how to execute the task. Once the task is completed successfully, the Experience Generator will attempt to create a new experience template for future use.

*Cross-app task support.* In real-world scenarios, a single task may span multiple applications (e.g., compare prices

> **[图片提取文字 (无描述)]:**
> Help me book a hotel in BJ / SH ActTree ActChain Template 2 existing node reuseable Replay step Template 1 Predictor, new node click unreuseable "Prepare for search" step click "hotels" section existing edge click click city selector cache hit cache hit action edge "Input city" input cache miss input BJ input SH cache miss action edge "Confirm search" click search button **UI Diff** Operator Module replay record replay record (cache hit) predicted changed? action , Agent Service prev/curr model call UI repr. (cache miss)
![](_page_5_Figure_6.jpeg)

<span id="page-5-1"></span>Figure 5: Two structures of Action Memory: ActTree (left) for prefix reuse and ActChain (right) for prefix-suffix reuse.

across shopping apps). The Experience Memory module handles such cases via a DAG-based orchestration: experiences are modeled as a DAG of subtasks executed in topological order. Each subtask can declare parameter slots whose concrete values are determined by the outputs of its preceding subtasks. More details will be introduced in [§5.1.](#page-6-0)

## <span id="page-5-0"></span>4.3 Action Memory

Challenges. Pure LLM-based agents lack the inherent ability to self-improve execution efficiency after deployment. Even when executing previously completed tasks, these agents repeat the same reasoning process due to their stateless nature, resulting in consistently high latency, particularly for long-horizon tasks. Existing systems such as AutoDroid have attempted to address this through task-level action caching, which reuses entire execution traces for similar tasks. However, this approach fails to generalize across tasks with partially shared sub-procedures, and lacks mechanisms to detect environment changes, causing low cache hit rates and incorrect trace reuse in non-deterministic environments.

Key Insight. Our key insight is that, when humans perform similar or repetitive tasks, they often operate subconsciously and do not engage in deliberate, complex reasoning. In agentic scenarios, we also observe that tasks within the same application typically share common prefixes; consequently, these actions can be directly reused without invoking the model. By organizing execution traces into a tree structure, we can identify shared prefixes at runtime using a lightweight embedding model, analogous to human procedural memory. For tasks bound to the same experience, we can achieve even more aggressive optimization. Drawing from the analysis in Section [4.2,](#page-4-0) we can decompose a task into a sequence of invariant and variable steps. Invariant steps remain unchanged across executions and can be directly reused, while variable steps are dependent on specific parameters and can only be reused if their parameters exactly match those of previously executed steps. This decomposition allows us to cache and reuse the stable parts of an agent workflow without model reasoning.

Our Design. Based on the above idea, the Action Memory

module organizes interactions between the agent and system in two structures to support fine-grained action-level reuse: *ActTree* for prefix reuse and *ActChain* for prefix-suffix reuse, as shown in Figure [5.](#page-5-1)

*Prefix reuse.* When the incoming task is not bound to any experience template, the Action Memory module works in the prefix reuse mode. This assumes that tasks within the same app can typically reuse prefix actions and share prefix pages. However, once an action branch occurs, the consistency of the subsequent environment can no longer be guaranteed. For each application, it maintains an app-level memory with an ActTree structure where nodes represent UI states and edges represent state transitions with corresponding actions, aggregating all action sequences from previously completed tasks. We design a lightweight embedding model as a Replay Predictor to determine whether the current task can reuse actions completed by the historical task at the *n*-th layer of ActTree. The Action Memory module will adaptively select the reuse threshold to enhance reuse accuracy.

*Prefix-suffix reuse.* When an incoming task is associated with an experience template, the Action Memory module switches to a prefix-suffix reuse mode. Using experience templates makes it possible to recognize that suffix actions can also be merged into the same state (e.g., an invariant step), thereby enabling suffix reuse. For each experience template, the Action Memory module maintains an ActChain structure that stores detailed action sequences corresponding to the specific tasks mapped to that template. When a new task executes invariant steps, the Action Memory module directly reuses the action. For variable steps, if the associated parameter values match those of a historical task, the corresponding action can also be reused; otherwise, the LLM performs fresh reasoning and action generation.

*Correctness check and rollback mechanism.* To guarantee that cached actions remain valid despite potential app updates or UI changes, we employ a verification mechanism for detecting stale action memory before execution. The Action Memory module examines the UI hierarchy (e.g., XML) to locate an element with matching properties such as resource ID, class name, and text content, and uses fuzzy matching to tolerate minor variations. When the check fails, we infer that the page layout has likely changed due to app updates or dynamic content, triggering a rollback mechanism that discards the failed action, falls back to the LLM execution and updates the Action Memory with the new execution trace.

## 5 Agent Service: System Integration

By leveraging three types of agent memory, MOBIMEM enables improvements in personalization, capability and execution efficiency during agent deployment without requiring additional model training. As shown in Figure [6,](#page-6-1) inspired by task and memory management mechanisms in traditional operating systems, MOBIMEM further incorporates agent-

> **[图片提取文字 (无描述)]:**
> Agent Scheduler DAG Orchestrator Update Queues Async. Issue task Updates High Prio. Mid Prio. end start Low Prio. intermediate Parallel Resume Submitted Subtasks, Plan Updates Interrupt Agent Signal System Memory S Exception C AgentRR Modules Agents Handler Interrupt **UI State** Action Context Target Device (e.g., Phone, Desktop, ...) Takeover Exception Handling Record & Replay Memory Update Interactions Interactions Interactions
![](_page_6_Figure_6.jpeg)

<span id="page-6-1"></span>Figure 6: Agent Service integration architecture, showing how the Agent Scheduler, AgentRR, and Agent Exception Handler coordinate with memory modules, system agents, and the target device.

oriented scheduling, record-and-replay, interruption handling and exception recovery mechanisms. These enhancements improve the agent system's efficiency and robustness when handling multitasking scenarios and abnormal cases.

## <span id="page-6-0"></span>5.1 Agent Scheduler

During task execution, both agent and memory operations incur latency, and a strictly serial workflow yields higher endto-end latency. For example, memory retrievals can be completed within tens of milliseconds using embedding models and graph traversal, whereas memory updates require LLM inference, taking 1–2 seconds for profile construction and several seconds for template distillation. For operator agents, each task requires reading information from the screen, which takes about 1–10 seconds on average, depending on hardware capabilities and model size. However, these operations are not strictly interdependent, creating opportunities for parallel execution. In particular, this parallelism can be exploited during the planning stage, the execution stage, as well as enabling frontend-backend concurrency.

*Parallel execution coordination.* The scheduler coordinates parallel execution at multiple granularities during task processing. In the planning phase, MOBIMEM concurrently retrieves user preferences and experience templates from the profile memory and experience memory. It then invokes the Task Rewriter to generate concrete task descriptions. As for the execution phase, one task may involve multiple applications, MOBIMEM supports both coarse-grained and finegrained parallelism. Coarse-grained parallelism enables parallel execution at the application level. When sub-tasks from different applications are independent, such as performing price comparisons across multiple shopping apps, MOBIMEM schedules them to run concurrently. Fine-grained parallelism

enables concurrency at the step level. For example, experience templates may indicate that app B only depends on a particular step of app A where a specific parameter value is required. In such cases, MOBIMEM constructs a step-level DAG to capture these dependencies and maximize the concurrency of independent steps.

*Background update prioritization.* The scheduler manages continuous background updates through three separate queues with different priorities. Profile updates are assigned the lowest priority because they do not directly affect the accuracy or efficiency of subsequent task execution. In contrast, updates to the Experience Memory module and Action Memory module are given higher priority, as they are triggered upon task completion and have immediate impacts on the performance of future tasks. This prioritization ensures that high-priority updates are processed promptly and are not delayed by the continuous stream of profile updates.

## <span id="page-7-0"></span>5.2 AgentRR: Agent Record and Replay

The Action Memory mechanism relies on execution traces to support action reuse, but naïve action logging is insufficient for agent replay. The Action Memory module employs two structures, ActTree and ActChain; however, the ActTree must be dynamically constructed during execution, while the ActChain requires explicit mappings between experience templates and action sequences. We design Agent Record and Replay (AgentRR) to capture UI states, decision contexts, and execution steps necessary for both replay structures. Unlike traditional system-level record-and-replay techniques that merely reproduce exact action sequences, AgentRR offers agent-level replay capabilities that not only regenerate action trajectories but also preserve an agent's generalization ability through template-based abstraction and adaptive action verification.

*Recording mechanism.* We design AgentRR as a lightweight instrumentation layer that intercepts all interactions between the agent and the mobile device. At each decision point during execution, AgentRR records the current UI state (including the screenshot and UI hierarchy) and the corresponding action. For the ActTree, each recording adds a new path. If the nodes (UI states) and edges (actions) on this path match existing nodes and edges in the ActTree, AgentRR merges them and updates the associated task list. For the ActChain, AgentRR summarizes the action sequence into an experience template and generates the corresponding step-toaction mappings. If an experience template already exists, the current action trajectory is used to validate it.

*Replay mechanism.* When replaying cached actions, AgentRR verifies each action before execution as described in [§4.3.](#page-5-0) Upon successful verification, AgentRR translates the cached action into concrete operations. Each cached action already contains the action type, target UI element, and parameters, enabling AgentRR to map it directly to the corresponding

UI element in the current UI hierarchy. If verification fails, AgentRR falls back to the Operator Agent for re-planning, allowing the agent to complete the remaining steps. After task execution, AgentRR will record the newly generated actions as an alternative path or update stale action memory caused by UI changes or application updates.

# <span id="page-7-1"></span>5.3 Agent Exception Handler

During task execution, users may need to interrupt the agent when they observe incorrect actions or want to make manual corrections. Without proper exception handling mechanisms, such interruptions would break the automated execution flow. Moreover, if exception handlers are not preserved, the same errors may recur in subsequent executions, making it impossible to leverage prior handling experience.

*Exception detection and execution suspension.* We design an exception-aware execution model that treats user interventions as recoverable events. The system continuously monitors for interruption signals during execution, including explicit pause commands and manual UI interactions that conflict with planned agent actions. When an interruption is detected, the scheduler immediately suspends the current execution thread and retains the context of the agent's execution process. This includes preserving the full execution context: the current UI state, the partially completed task plan, and the action history. The system then yields control to the user, allowing them to inspect the current state, review planned actions, and make corrections through natural language instructions or direct UI manipulations.

*Recovery and continuation.* Once the user completes the corrections, the system integrates the exception handling actions with the execution context to proceed with subsequent planning and execution. Rather than directly incorporating user corrections into experience templates, the *Experience Generator* analyzes the entire execution trace including both the agent's actions and user's corrections. It compares the original plan with the corrections, identifies where and why the agent deviated from user intent, and distills these insights into improved templates. This analysis-driven approach ensures that the system learns what corrections the user made as well as why the corrections were necessary, enabling robust template evolution that prevents similar issues in future tasks.

## 6 Evaluation

We evaluate MOBIMEM along three dimensions that mirror our system design in Section [4:](#page-3-2) (1) whether the DisGraphbased User Profile module can efficiently capture and serve user preferences, (2) how experience templates in the Experience Memory module affect task success rates and cost, and (3) how the Action Memory module improves end-toend latency across different tasks. We deploy MOBIMEM on

<span id="page-8-1"></span>Table 2: Profile Memory Module evaluation results (500 historical tasks, 30 test tasks).

| System       | Write (ms) | Retrieval (ms) | Alignment (%) |
|--------------|------------|----------------|---------------|
| Vanilla RAG  | 1.76       | 19.58          | 66.4          |
| GraphRAG     | 37688.73   | 6675.82        | 81.1          |
| Ours (Graph) | 6138.87    | 23.83          | 83.1          |

<span id="page-8-2"></span>Table 3: Scalability of Profile Memory Module across different node counts.

| Nodes   | Update  | Retrieval (ms) |          | Storage |
|---------|---------|----------------|----------|---------|
|         | (ms)    | Vector DB      | DisGraph | (MB)    |
| 100     | 6630.60 | 8.77           | 0.15     | 1.41    |
| 1,000   | 7051.38 | 21.11          | 0.17     | 13.49   |
| 10,000  | 8017.17 | 178.40         | 0.26     | 134.79  |
| 100,000 | 9193.84 | 1291.28        | 0.35     | 1346.48 |

both edge and cloud environments and conduct evaluations in real-world mobile usage scenarios, covering the 50 most commonly used applications. Due to the strict latency requirements in edge scenarios, real deployments still rely on cloud compute resources (Intel Xeon Platinum 8378A CPUs and NVIDIA A100-SXM4-80GB GPUs) for model inference. For on-device evaluation, we use the Qualcomm Snapdragon 8 Elite SoC platform and deploy the model using llama.cpp [16]. We evaluate multiple SOTA GUI agent models for the operator agent to ensure generality, including MobiMind-4B [72], UI-TARS-1.5-7B [47], GUI-Owl-7B [67], Qwen3-VL-30B-A3B [8] and Gemini-2.5-Flash [19], covering different scales and architectures. We do not evaluate industrial flagship reasoning models such as Gemini-2.5-Pro or open-source models with more than 10B activated parameters, as their high inference latency cannot meet the latency SLOs of on-device agents. Other agents, such as Profile Updater, Experience Generator, and Task Rewriter, are not on the critical path, and we adopt Qwen3-VL-30B-A3B as the base model. The Experience Memory and AgentRR features proposed in MOBIMEM have already been deployed on a flagship smartphone.

<span id="page-8-5"></span>Table 4: Efforts required to enable new agent capabilities when employing different approaches.

| Approach    | Data           | Person-hrs | GPU hrs | Acc.  |
|-------------|----------------|------------|---------|-------|
| Fine-tuning | $\sim$ 100 ex. | 4.0        | 0.25    | 58.5% |
| Exp.(Man.)  | $\sim$ 5 ex.   | 0.2        | 0       | 63.5% |
| Exp.(Syn.)  | $\sim$ 5 ex.   | 0          | 0.0027  | 60.1% |

### 6.1 Agent's Personalization

To evaluate whether agents can accurately learn and model user profiles from historical task executions, we design a benchmark<sup>1</sup> with synthetic user profiles. Each profile contains descriptions of a person's characteristics across various aspects, including facts (e.g., home address, frequently used

> **[图片提取文字 (无描述)]:**
> GUI-OwI-7B Gemini-2.5-Flash GUI-OwI-7B w/ experience Gemini-2.5-Flash w/ experience UI-TARS-1.5-7B Qwen3-VL-30B-A3B UI-TARS-1.5-7B w/ experience Qwen3-VL-30B-A3B w/ experience Average success rate (%) Task index
![](_page_8_Figure_10.jpeg)

<span id="page-8-3"></span>Figure 7: Average success rates of different agents on AndroidWorld with and without experience templates.

> **[图片提取文字 (无描述)]:**
> without Experience with Experience In Distribution Out of Distribution Success Rate (%) 100 80 60 40 20 Shop Delivery Social Video Travel Shop Delivery Social Video Travel
![](_page_8_Figure_12.jpeg)

<span id="page-8-4"></span>Figure 8: Task success rates with and without experience templates in ID/OOD scenarios.

accounts), preferences (e.g., shopping habits, hotel booking preferences), etc. For each user, we generate 500 historical tasks that explicitly reflect the user's profile information, and 30 test tasks with ambiguous descriptions that require the agent to infer missing details based on learned profile memory. For each test task, we define which profile information should be retrieved based on the task context, serving as the ground truth for evaluating retrieval quality. During evaluation, each system learns from the historical tasks by updating its profile store, then retrieves relevant profile information for the test tasks. We measure three metrics: write latency (time to update profile), retrieval latency (time to retrieve profile information), and profile alignment (the ratio of the retrieved profile information to the ground truth). We compare our DisGraph-based Profile Memory against two baselines: Vanilla RAG, which stores preferences as unstructured text in a vector database with embedding-based retrieval, and GraphRAG, which constructs a knowledge graph with entity extraction and hierarchical summarization following the Mem0 approach. All approaches use a context window of 2000 tokens for retrieval.

As shown in Table 2, Vanilla RAG achieves low latency (1.76ms write, 19.58ms retrieval) but suffers from poor alignment (66.4%) due to its flat structure lacking relational context. GraphRAG improves alignment to 81.1% by encoding relational semantics in edge weights, but incurs high latency (37.69s write, 6.68s retrieval) from expensive LLM calls for entity extraction, edge management, and query traversal. Our system achieves the best of both compared to the GraphRAG:

<span id="page-8-0"></span><sup>&</sup>lt;sup>1</sup>Details in the Appendix.

> **[图片提取文字 (无描述)]:**
> Human-crafted Experience LLM Generated Experience ActTree Rate (%) 83 54.4 50.3 44.8 38.8 38.0 CV Reuse 38 40 23.8 6.2 20 Ticketing Delivery Hotel E-mail Shopping Browser Video Мар Task Type
![](_page_9_Figure_0.jpeg)

<span id="page-9-1"></span>Figure 9: Action reuse rates of ActTree and ActChain with LLM-generated/human-crafted experience templates.

83.1% alignment (25% higher than Vanilla RAG) with 6.14s write and 23.83ms retrieval latency (6.1× and 280× faster than GraphRAG respectively). This stems from our DisGraph architecture that decouples relational semantics from edges to nodes: writes require only one LLM call for entity classification and concept attachment, while retrieval operates through embedding-based starting point selection and BFS traversal with zero LLM calls. High accuracy is maintained because multi-branch BFS gathers contextually relevant information from multiple conceptual dimensions.

Scalability analysis. To evaluate how Profile Memory scales with user profile size, we test update and retrieval performance as the number of profile nodes grows from 100 to 100,000. As shown in Table 3, update latency increases moderately from 6.6 seconds to 9.2 seconds as profile size grows. Because the update process is dominated by a single LLM inference for entity classification and concept attachment, its runtime remains nearly unchanged. For retrieval, DisGraph traversal remains nearly constant at about 0.15ms–0.35ms across all profile sizes, confirming the zero-LLM overhead of our graph walking mechanism. Total retrieval latency is dominated by the initial vector database search used to select starting nodes, which increases from 8.8 ms (100 nodes) to 1.29 s (100,000 nodes, with 135 MB storage), a delay that remains acceptable in real-world scenarios.

### 6.2 Agent's Capability

**Task success rate in AndroidWorld.** We first evaluate how experience templates improve task success rates in AndroidWorld [49], which provides 116 diverse tasks across 20 Android apps. Our baseline implementation references DroidRun [17], the SOTA open-source work on Android-World, and we augment it with our experience mechanisms to measure the improvements. We test four agent models with and without experience templates: GUI-Owl-7B, UI-TARS-1.5-7B<sup>2</sup>, Gemini-2.5-Flash, and Qwen3-VL-30B-A3B.

As shown in Figure 7, experience templates consistently improve success rates across all agents. Weaker models benefit more: UI-TARS-1.5-7B achieves a 50.3% relative improvement, while stronger models such as Gemini and Qwen gain

> **[图片提取文字 (无描述)]:**
> MobiMind-4B (Before/After) UI-TARS-1.5-7B (Before/After) GUI-Owl-7B (Before/After) Reuse Rate 60 Latency (s) use Rate (%) 50 30 Ticketing Delivery Hotel E-mail Shopping Browser Video Map Task Type
![](_page_9_Figure_8.jpeg)

<span id="page-9-2"></span>Figure 10: End-to-end latency and reuse rates for three agent models with and without Action Memory.

21%–22%, and GUI-Owl gains 10.5%. Beyond this overall trend, we observe that different template levels contribute differently across model types. For general-purpose models such as Gemini and Qwen, lower-level templates are more effective because the logic of UI interactions is difficult to infer through reasoning alone. In contrast, for domain-specific models like GUI-Owl, which already possess basic UI interaction skills, higher-level templates reduce reliance on autonomous planning, thereby yielding greater benefits for smaller models. Each experience template consumes 7.8KB on average, resulting in 900KB total storage for the 116 templates.

Task success rate in real-world applications. We evaluate success rates when applying experience memory across various real-world tasks, covering both in-distribution (ID) and out-of-distribution (OOD; i.e., long-tail tasks beyond the agent's training data). UI-TARS-1.5-7B is used as the agent model for this evaluation. For tasks that are only partially completed, the completion rate is taken as the success rate. As shown in Figure 8, applying experience memory leads to substantial improvements in task success rates for both OOD and ID settings. Because OOD tasks often involve complex reasoning, introducing experience templates significantly reduces the reasoning burden and yields greater accuracy gains (44.1% for OOD tasks and 22.0% for ID tasks).

Cost-effectiveness. To evaluate the cost-effectiveness of experience memory, we compare three approaches for enabling new agent capabilities: model fine-tuning, manually authored experience templates, and automatically synthesized experience templates in our system. Table 4 compares these approaches across four key metrics: data volume (number of task execution traces needed), person-hours (human effort for data collection, annotation, or template authoring), GPU hours (computational resources for model training or LLM-based template generation), and accuracy (task success rate).

Fine-tuning requires  $\sim 100$  training examples, 4 personhours, and 0.25 GPU hours to achieve 58.5% accuracy, but must be repeated for each new task family. In contrast, both template methods require only  $\sim 5$  examples ( $20 \times$  less data) by explicitly separating invariant control logic from variable parameter slots. Manual authoring takes 0.2 person-hours for experts to identify control logic and mark variable slots, achieving the highest accuracy (63.5%) through more precise separation of invariant and variant steps, while automatic

<span id="page-9-0"></span><sup>&</sup>lt;sup>2</sup>The 33.0% accuracy is reported in UI-TARS paper, but we cannot fully reproduce this result as the prompts used for testing are not open-sourced.

> **[图片提取文字 (无描述)]:**
> A100 (Before/After) Snapdragon 8 Elite (Before/After) 910B (Before/After) Hotel **Ticketing** Delivery E-mail Latency (s) 9.00x 5.76x 5.11x 2.17x 10<sup>2</sup> 102 10<sup>2</sup> 3.60x 3.00x 2.85x 1.78x 2.13x 1.94x 1.89x 1.47x 10<sup>1</sup> 10 10 Shopping Browser Video Map Latency (s) 10<sup>2</sup> 1.93x 10<sup>2</sup> 1.85x 1.80x 1.58x 10<sup>2</sup> 1.64x 1.62x 1.57x 1.44x 10 101 1.41x 1.41x 1.36x 1.29x 10
![](_page_10_Figure_0.jpeg)

<span id="page-10-0"></span>Figure 11: Performance boost of Action Memory across different hardware configurations.

synthesis automates this process through semantic clustering and action sequence alignment, requiring zero human effort and only 0.0027 GPU hours to achieve 60.1% accuracy. This demonstrates that experience memory provides a cost-effective alternative to fine-tuning by separating invariant control logic from runtime variables, enabling rapid task family coverage while maintaining competitive accuracy.

### 6.3 Agent's Efficiency

To evaluate the effectiveness of action reuse, we evaluate the different structures of Action Memory under realistic workload patterns, which include 8 diverse task categories: email, train ticketing, food delivery, hotel booking, shopping, web browser, media playback, and map navigation. For each category, we generate parameterized task instances using an LLM to create realistic variations. We submit a sequence of 454 tasks to the system across these categories.

Action reuse rate. We first evaluate how different implementations of Action Memory affect action reuse rates. Figure 9 compares the reuse rates of ActTree and ActChain across task categories. For ActChain, we further evaluate the impact of two template sources on reuse rates: LLM-generated templates and human-crafted templates. ActTree only exploits prefix reuse, achieving 37.5% average reuse rate. ActChain exploits both prefix and suffix reuse, reaching higher reuse rates compared to ActTree. Human-crafted templates strictly separate invariant and variable actions into adjacent steps to maximize the reuse of invariant steps, achieving the highest average reuse rate of 77.3%. LLM-generated templates do not enforce the above constraint, resulting in lower average reuse rate of 59.7% but still outperform ActTree, demonstrating the effectiveness of prefix-suffix reuse.

**End-to-end latency.** We evaluate how Action Memory affects end-to-end task execution time across different agents. We employ ActChain with human-crafted templates for this evaluation. As shown in Figure 10, without Action Memory, latency varies across agents due to differences in inference efficiency: MobiMind-4B averages 14.1s, UI-TARS-1.5-7B 14.7s, and GUI-Owl-7B 38.0s. With Action Memory enabled, all three models achieve substantial latency reduc-

> **[图片提取文字 (无描述)]:**
> Serial Execution Mode Coarse-grained Parallel Mode Fine-Grained Parallel Mode Execution Time (s) 50.50 44.35 50. 60 33.6 23.83 83 40 8 31 3 26 œί 3 20 search+shop multi-video multi-shop single-shop search search +social +social +social +social +social +shop Task Combination
![](_page_10_Figure_7.jpeg)

<span id="page-10-1"></span>Figure 12: End-to-end performance of different execution modes across six task categories.

tions: MobiMind-4B reduces to 8.6s (up to  $2.1\times$  speedup), UI-TARS-1.5-7B to 8.8s (up to  $2.2\times$  speedup), and GUI-Owl-7B to 16.2s (up to  $4.5\times$  speedup). The latency reduction correlates strongly with reuse rates: tasks such as hotel query and train ticketing with over 92% reuse rates achieve up to  $4.5\times$  speedup, demonstrating that Action Memory effectively eliminates the bottleneck of LLM inference. Tasks such as shopping and browser with lower reuse rates ( $\sim$ 70%) still require longer execution time due to more frequent LLM inference calls for cache misses. For memory overhead, Action Memory uses only 1.54MB ( $\sim$ 6,000 cached actions), negligible on modern devices.

Performance across different hardware. Finally, we evaluate Action Memory across different hardware configurations representing diverse deployment scenarios: A100 GPU, Ascend 910B NPU [35] (cloud server), and Qualcomm Snapdragon 8 Elite SoC [48] (mobile device). We use MobiMind-4B as the agent model for this evaluation. As shown in Figure 11, without Action Memory, task latency varies dramatically across hardware: A100 averages 14.1s, 910B 27.4s, while Snapdragon suffers from 153.2s due to limited computational resources (CPU-only). After enabling Action Memory, latency becomes much more uniform across hardware configurations: most tasks complete within 10s-50s across different underlying compute platforms. The speedup is most pronounced on mobile device where Action Memory achieves  $1.6 \times -9 \times$  speedup across tasks, as action reuse effectively eliminates expensive on-device inference. Even on highperformance hardware like A100, Action Memory provides consistent  $1.3 \times -2.1 \times$  speedup by avoiding redundant inference. This demonstrates that Action Memory effectively shifts the bottleneck from LLM inference to lightweight action execution and UI interaction, enabling practical deployment on mobile and edge devices.

#### 6.4 Performance on Multi-task Scheduling

To evaluate MOBIMEM's performance in realistic scenarios, we test MOBIMEM across complex multi-app tasks that require cross-app coordination and data transfer. These tasks represent real-world use cases where information is retrieved from one or more apps and consumed in other apps. The benchmark involves three application categories: search, shop,

> **[图片提取文字 (无描述)]:**
> A. Serial Execution Mode Planning Phase Query price in APP1 Query price in APP2 Send Message in APP3 Total Time 48.27 s B. Coarse-grained Parallel mode Planning Phase Query price in APP1 1.28× speedup Query price in APP2 Total Time 37.55 s Send Message in APP3 C. Fine-Grained Parallel Mode Planning Phase Query price in APP1 1.62× speedup Query price in APP2 Total Time 29.84 s Prepare in APP3 Suspended Send Sync Point: All data ready T=0
![](_page_11_Figure_0.jpeg)

<span id="page-11-1"></span>Figure 13: Detailed execution timeline comparison between serial and parallel modes for a multi-shop+social task.

and social networking, using applications such as Chrome, Amazon, WeChat, etc. These application categories are combined to form six task categories as shown in Figure [12,](#page-10-1) where the "multi-" prefix indicates that multiple apps of the same type are involved. Each category contains multiple task instances with different parameter initializations.[3](#page-11-0) We compare the end-to-end latency of three execution modes: *Serial* executes all sub-tasks sequentially. *Coarse-grained* parallelizes independent sub-tasks at the application level. *Fine-grained* further enables step-level parallelism within each sub-task.

Coarse-grained parallelism achieves up to 1.41× speedup over serial execution in the multi-shop+social task. Finegrained parallelism achieves more substantial improvements, with up to 1.98× speedup over serial execution in the same task category. For tasks with sequential app dependencies (e.g., search+shop+social), coarse-grained parallelism provides no benefit, while fine-grained mode still achieves 1.50× and 1.53× speedups through step-level parallelism.

To understand the performance improvements, we analyze a representative task from the multi-shop+social category, as illustrated in Figure [13.](#page-11-1) This task queries prices for the same item across two shopping apps and then shares the results via a social app. In serial mode, the system sequentially queries prices in two shopping apps and then sends a message in a social app, taking 48.27 seconds total. In coarse-grained mode, the two price queries execute in parallel, reducing execution time to 37.65 seconds (1.28× speedup). In fine-grained mode, the scheduler proactively executes steps in the social app that do not depend on price data (e.g., navigating to the chatting page) while price queries run in parallel, blocking only at message composition when price data is required. This further reduces execution time to 29.84 seconds (1.62× speedup), demonstrating that MOBIMEM's fine-grained parallelization effectively exploits step-level dependencies and achieves superior performance.

## 7 Discussion

Platform portability. Our system is implemented on mobile devices with structured GUI introspection, OS-level execution APIs, and execution trace logging. The evaluation demonstrates hardware portability across cloud GPUs, NPUs, and mobile SoCs, validating that our memory mechanisms maintain effectiveness across diverse computational resources. The core memory mechanisms (DisGraph for user profiles, multi-level templates for experiences, and ActTree/ActChain for action reuse) are platform-agnostic abstractions that can be adapted to different environments. Desktop systems provide equivalent capabilities through Microsoft UI Automation [\[44\]](#page-14-15), Linux AT-SPI [\[38\]](#page-13-13), and Apple Accessibility API [\[3\]](#page-12-15), web browsers offer comprehensive DOM introspection with tools like Selenium [\[6\]](#page-12-16) and Playwright [\[5\]](#page-12-17), and commandline interfaces can leverage process monitoring and output parsing. Through appropriate abstraction layers, our memory mechanisms can generalize across diverse platforms.

Comparison with general agent systems. General agent frameworks such as LangChain [\[31\]](#page-13-14), AutoGen [\[60\]](#page-15-17), and MetaGPT [\[22\]](#page-13-15) orchestrate multi-agent workflows atop LLM serving systems [\[1,](#page-12-18) [15,](#page-12-19) [29,](#page-13-16) [36,](#page-13-17) [52,](#page-14-16) [68,](#page-15-18) [75,](#page-16-1) [76,](#page-16-2) [78\]](#page-16-3) that optimize inference efficiency. While these frameworks enable flexible agent composition, their memory mechanisms target conversational agents with linear interaction patterns rather than GUI agents operating in stateful visual environments. The higher cost and complexity of GUI automation necessitate specialized memory: costly physical UI operations require fine-grained action caching, trial-and-error in long-horizon tasks demands abstraction of reusable experience templates from execution traces, and personalization requires capturing multi-app user preference patterns. Existing frameworks rely on conversation buffers designed for dialogue history rather than memory supporting state-indexed action retrieval, hierarchical experience abstraction, and evolving user profiles. MOBIMEM introduces three specialized memory types that transform GUI agents from static executors into self-evolving systems that improve through deployment.

## 8 Conclusion

In this paper, we present MOBIMEM, a memory-centric system that enables agent self-evolution without continual model training. MOBIMEM introduces three specialized memory types with system-level integration, demonstrating significant improvements in preference alignment, execution accuracy and latency. As AI agents are increasingly deployed in edge environments, we believe the memory-centric paradigm offers a practical path toward continually evolving agents that learn from agent memories rather than expensive model updates.

<span id="page-11-0"></span><sup>3</sup>Details in the Appendix.

## References

- <span id="page-12-18"></span>[1] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ramjee. Taming throughputlatency tradeoff in llm inference with sarathi-serve. In *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)*, pages 117–134, 2024.
- <span id="page-12-7"></span>[2] Gautam Altekar and Ion Stoica. Odr: Outputdeterministic replay for multicore debugging. In *Proceedings of the ACM SIGOPS 22nd symposium on Operating systems principles*, pages 193–206, 2009.
- <span id="page-12-15"></span>[3] Apple Inc. Accessibility API. [https:](https://developer.apple.com/documentation/accessibility/accessibility-api) [//developer.apple.com/documentation/](https://developer.apple.com/documentation/accessibility/accessibility-api) [accessibility/accessibility-api](https://developer.apple.com/documentation/accessibility/accessibility-api), 2025. Accessed: 2025-12-04.
- <span id="page-12-9"></span>[4] Authors. Ui-tars: A unified framework for gui automation with large language models. *arXiv preprint*, 2024. To be updated with actual citation.
- <span id="page-12-17"></span>[5] Playwright Authors. Playwright. [https://](https://playwright.dev/) [playwright.dev/](https://playwright.dev/), 2025. Accessed: 2025-12-04.
- <span id="page-12-16"></span>[6] Selenium Authors. Selenium. [https://www.](https://www.selenium.dev/) [selenium.dev/](https://www.selenium.dev/), 2025. Accessed: 2025-12-04.
- <span id="page-12-10"></span>[7] Sanghwan Bae, Donghyun Kwak, Soyoung Kang, Min Young Lee, Sungdong Kim, Yuin Jeong, Hyeri Kim, Sang-Woo Lee, Woomyoung Park, and Nako Sung. Keep me updated! memory management in long-term conversations. *arXiv preprint arXiv:2210.08750*, 2022.
- <span id="page-12-12"></span>[8] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025.
- <span id="page-12-1"></span>[9] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. In *Annual Meeting of the Association for Computational Linguistics*, 2024.

- <span id="page-12-5"></span>[10] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building productionready ai agents with scalable long-term memory, 2025.
- <span id="page-12-2"></span>[11] Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. Agentic reinforced policy optimization, 2025.
- <span id="page-12-8"></span>[12] George W Dunlap, Samuel T King, Sukru Cinar, Murtaza A Basrai, and Peter M Chen. Revirt: Enabling intrusion analysis through virtual-machine logging and replay. *ACM SIGOPS Operating Systems Review*, 36(SI):211–224, 2002.
- <span id="page-12-6"></span>[13] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization, 2025.
- <span id="page-12-3"></span>[14] Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for llm agent training. In *Advances in Neural Information Processing Systems*, 2025.
- <span id="page-12-19"></span>[15] Yao Fu, Leyang Xue, Yeqi Huang, Andrei-Octavian Brabete, Dmitrii Ustiugov, Yuvraj Patel, and Luo Mai. Serverlessllm: Low-latency serverless inference for large language models. In *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)*, pages 135–153, 2024.
- <span id="page-12-11"></span>[16] Ggerganov. llama.cpp - LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware. [https://github.com/ggerganov/](https://github.com/ggerganov/llama.cpp) [llama.cpp](https://github.com/ggerganov/llama.cpp), 2025. Accessed: 2025-12-04.
- <span id="page-12-14"></span>[17] Droidrun GmbH. Droidrun: A framework for controlling android and ios devices through llm agents. <https://github.com/droidrun/droidrun>, 2025. Accessed: 2025-12-09.
- <span id="page-12-0"></span>[18] Gonzalo Gonzalez-Pumariega, Vincent Tu, Chih-Lun Lee, Jiachen Yang, Ang Li, and Xin Eric Wang. The unreasonable effectiveness of scaling agents for computer use, 2025.
- <span id="page-12-13"></span>[19] Google. Gemini 2.5 flash. [https://deepmind.](https://deepmind.google/models/gemini/flash/) [google/models/gemini/flash/](https://deepmind.google/models/gemini/flash/), 2025. Accessed: 2025-12-04.
- <span id="page-12-4"></span>[20] Zhangxuan Gu, Zhengwen Zeng, Zhenyu Xu, Xingran Zhou, Shuheng Shen, Yunfei Liu, Beitong Zhou, Changhua Meng, Tianyu Xia, Weizhi Chen, Yue Wen, Jingya Dou, Fei Tang, Jinzhen Lin, Yulin Liu, Zhenlin Guo, Yichen Gong, Heng Jia, Changlong Gao,

- Yuan Guo, Yong Deng, Zhenyu Guo, Liang Chen, and Weiqiang Wang. Ui-venus technical report: Building high-performance ui agents with rft, 2025.
- <span id="page-13-5"></span>[21] Zhenyu Guo, Xi Wang, Jian Tang, Xuezheng Liu, Zhilei Xu, Ming Wu, M. Frans Kaashoek, and Zheng Zhang. R2: an application-level kernel for record and replay. In *Proceedings of the 8th USENIX Conference on Operating Systems Design and Implementation*, OSDI'08, page 193–208, USA, 2008. USENIX Association.
- <span id="page-13-15"></span>[22] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al. Metagpt: Meta programming for a multi-agent collaborative framework. In *The Twelfth International Conference on Learning Representations*, 2023.
- <span id="page-13-1"></span>[23] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, and Jie Tang. Cogagent: A visual language model for GUI agents. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition, Seattle, WA, USA, June 16-22, 2024*, pages 14281–14290. IEEE, 2024.
- <span id="page-13-9"></span>[24] Yuki Hou, Haruki Tamoto, and Homei Miyashita. " my agent understands me better": Integrating dynamic human-like memory recall and consolidation in llmbased agents. In *Extended Abstracts of the CHI Conference on Human Factors in Computing Systems*, pages 1–7, 2024.
- <span id="page-13-2"></span>[25] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.
- <span id="page-13-0"></span>[26] IBM. What are agentic workflows? [https://www.](https://www.ibm.com/think/topics/agentic-workflows) [ibm.com/think/topics/agentic-workflows](https://www.ibm.com/think/topics/agentic-workflows), 2025. Accessed: 2025-12-09.
- <span id="page-13-7"></span>[27] Wenjia Jiang, Yangyang Zhuang, Chenxi Song, Xu Yang, Joey Tianyi Zhou, and Chi Zhang. Appagentx: Evolving gui agents as proficient smartphone users, 2025.
- <span id="page-13-8"></span>[28] Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiologically inspired long-term memory for large language models. *Advances in Neural Information Processing Systems*, 37:59532–59569, 2024.
- <span id="page-13-16"></span>[29] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the 29th symposium on operating systems principles*, pages 611–626, 2023.

- <span id="page-13-6"></span>[30] Oren Laadan, Nicolas Viennot, and Jason Nieh. Transparent, lightweight application execution replay on commodity multiprocessor operating systems. In *Proceedings of the ACM SIGMETRICS international conference on Measurement and modeling of computer systems*, pages 155–166, 2010.
- <span id="page-13-14"></span>[31] LangChain. Langchain. [https://github.com/](https://github.com/langchain-ai/langchain) [langchain-ai/langchain](https://github.com/langchain-ai/langchain), 2025.
- <span id="page-13-4"></span>[32] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In *Proceedings of the 34th International Conference on Neural Information Processing Systems*, NIPS '20, Red Hook, NY, USA, 2020. Curran Associates Inc.
- <span id="page-13-11"></span>[33] Zaijing Li, Yuquan Xie, Rui Shao, Gongwei Chen, Dongmei Jiang, and Liqiang Nie. Optimus-1: Hybrid multimodal memory empowered agents excel in long-horizon tasks. *Advances in neural information processing systems*, 37:49881–49913, 2024.
- <span id="page-13-10"></span>[34] Zhiyu Li, Shichao Song, Hanyu Wang, Simin Niu, Ding Chen, Jiawei Yang, Chenyang Xi, Huayi Lai, Jihao Zhao, Yezhaohui Wang, Junpeng Ren, Zehao Lin, Jiahao Huo, Tianyi Chen, Kai Chen, Kehang Li, Zhiqiang Yin, Qingchen Yu, Bo Tang, Hongkang Yang, Zhi-Qin John Xu, and Feiyu Xiong. Memos: An operating system for memory-augmented generation (mag) in large language models, 2025.
- <span id="page-13-12"></span>[35] Heng Liao, Jiajin Tu, Jing Xia, Hu Liu, Xiping Zhou, Honghui Yuan, and Yuxing Hu. Ascend: a scalable and unified architecture for ubiquitous deep neural network computing : Industry track paper. In *2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA)*, pages 789–801, 2021.
- <span id="page-13-17"></span>[36] Chaofan Lin, Zhenhua Han, Chengruidong Zhang, Yuqing Yang, Fan Yang, Chen Chen, and Lili Qiu. Parrot: Efficient serving of llm-based applications with semantic variable. In *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)*, pages 929–945, 2024.
- <span id="page-13-3"></span>[37] Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Shiwei Wu, Zechen Bai, Weixian Lei, Lijuan Wang, and Mike Zheng Shou. Showui: One visionlanguage-action model for gui visual agent, 2024.
- <span id="page-13-13"></span>[38] Linux From Scratch. AT SPI. [https:](https://www.linuxfromscratch.org/blfs/view/5.1/gnome/at-spi.html) [//www.linuxfromscratch.org/blfs/view/5.](https://www.linuxfromscratch.org/blfs/view/5.1/gnome/at-spi.html) [1/gnome/at-spi.html](https://www.linuxfromscratch.org/blfs/view/5.1/gnome/at-spi.html). Accessed: 2025-12-04.

- <span id="page-14-3"></span>[39] Fanbin Lu, Zhisheng Zhong, Shu Liu, Chi-Wing Fu, and Jiaya Jia. Arpo:end-to-end policy optimization for gui agents with experience replay, 2025.
- <span id="page-14-11"></span>[40] Junru Lu, Siyu An, Mingbao Lin, Gabriele Pergola, Yulan He, Di Yin, Xing Sun, and Yunsheng Wu. Memochat: Tuning llms to use memos for consistent longrange open-domain conversation. *arXiv preprint arXiv:2308.08239*, 2023.
- <span id="page-14-9"></span>[41] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of llm agents. *arXiv preprint arXiv:2402.17753*, 2024.
- <span id="page-14-1"></span>[42] Microsoft. Creating cross-system workflows with power automate. [https://kumopartners.com/](https://kumopartners.com/creating-cross-system-workflows/) [creating-cross-system-workflows/](https://kumopartners.com/creating-cross-system-workflows/), 2025. Accessed: 2025-12-09.
- <span id="page-14-2"></span>[43] Microsoft. Introducing agent flows: Transforming automation with ai-first workflows. Microsoft Copilot Blog, 2025.
- <span id="page-14-15"></span>[44] Microsoft. Ui automation overview. [https://](https://learn.microsoft.com/en-us/dotnet/framework/ui-automation/ui-automation-overview) [learn.microsoft.com/en-us/dotnet/framework/](https://learn.microsoft.com/en-us/dotnet/framework/ui-automation/ui-automation-overview) [ui-automation/ui-automation-overview](https://learn.microsoft.com/en-us/dotnet/framework/ui-automation/ui-automation-overview), 2025. Accessed: 2025-12-04.
- <span id="page-14-5"></span>[45] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. Memgpt: Towards llms as operating systems, 2024.
- <span id="page-14-10"></span>[46] Hongjin Qian, Peitian Zhang, Zheng Liu, Kelong Mao, and Zhicheng Dou. Memorag: Moving towards next-gen rag via memory-inspired knowledge discovery. *arXiv preprint arXiv:2409.05591*, 1, 2024.
- <span id="page-14-13"></span>[47] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haoli Chen, Zhaojian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, and Guang Shi. Ui-tars: Pioneering automated gui interaction with native agents, 2025.
- <span id="page-14-14"></span>[48] Qualcomm. Snapdragon 8 elite mobile platform. [https://www.qualcomm.](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-mobile-platform) [com/smartphones/products/8-series/](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-mobile-platform) [snapdragon-8-elite-mobile-platform](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-mobile-platform), 2025. Accessed: 2025-12-04.
- <span id="page-14-6"></span>[49] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala,

- Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lillicrap, and Oriana Riva. Androidworld: A dynamic benchmarking environment for autonomous agents, 2024.
- <span id="page-14-8"></span>[50] Lianlei Shan, Shixian Luo, Zezhou Zhu, Yu Yuan, and Yong Wu. Cognitive memory in large language models. *arXiv preprint arXiv:2504.02441*, 2025.
- <span id="page-14-4"></span>[51] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In *Proceedings of the Twentieth European Conference on Computer Systems*, EuroSys '25, page 1279–1297, New York, NY, USA, 2025. Association for Computing Machinery.
- <span id="page-14-16"></span>[52] Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, and Wei Lin. Llumnix: Dynamic scheduling for large language model serving. In *18th USENIX symposium on operating systems design and implementation (OSDI 24)*, pages 173–191, 2024.
- <span id="page-14-12"></span>[53] Bing Wang, Xinnian Liang, Jian Yang, Hui Huang, Shuangzhi Wu, Peihao Wu, Lu Lu, Zejun Ma, and Zhoujun Li. Enhancing large language model with self-controlled memory framework. *arXiv preprint arXiv:2304.13343*, 2023.
- <span id="page-14-7"></span>[54] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models, 2023.
- <span id="page-14-0"></span>[55] Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, Wanjun Zhong, Yining Ye, Yujia Qin, Yuwen Xiong, Yuxin Song, Zhiyong Wu, Aoyan Li, Bo Li, Chen Dun, Chong Liu, Daoguang Zan, Fuxing Leng, Hanbin Wang, Hao Yu, Haobin Chen, Hongyi Guo, Jing Su, Jingjia Huang, Kai Shen, Kaiyu Shi, Lin Yan, Peiyao Zhao, Pengfei Liu, Qinghao Ye, Renjie Zheng, Shulin Xin, Wayne Xin Zhao, Wen Heng, Wenhao Huang, Wenqian Wang, Xiaobo Qin, Yi Lin, Youbin Wu, Zehui Chen, Zihao Wang, Baoquan Zhong, Xinchun Zhang, Xujing Li, Yuanfan Li, Zhongkai Zhao, Chengquan Jiang, Faming Wu, Haotian Zhou, Jinlin Pang, Li Han, Qi Liu, Qianli Ma, Siyao Liu, Songhua Cai, Wenqi Fu, Xin Liu, Yaohui Wang, Zhi Zhang, Bo Zhou, Guoliang Li, Jiajun Shi, Jiale Yang, Jie Tang, Li Li, Qihua Han, Taoran Lu, Woyu Lin, Xiaokang Tong, Xinyao Li, Yichi Zhang, Yu Miao, Zhengxuan Jiang, Zili Li, Ziyuan Zhao, Chenxin Li, Dehua Ma, Feng Lin, Ge Zhang, Haihua Yang, Hangyu Guo, Hongda Zhu, Jiaheng Liu, Junda Du, Kai Cai, Kuanye Li, Lichen Yuan, Meilan Han, Minchao Wang, Shuyue Guo, Tianhao Cheng, Xiaobo Ma, Xiaojun Xiao, Xiaolong Huang,

- Xinjie Chen, Yidi Du, Yilin Chen, Yiwen Wang, Zhaojian Li, Zhenzhu Yang, Zhiyuan Zeng, Chaolin Jin, Chen Li, Hao Chen, Haoli Chen, Jian Chen, Qinghao Zhao, and Guang Shi. Ui-tars-2 technical report: Advancing gui agent with multi-turn reinforcement learning, 2025.
- <span id="page-15-5"></span>[56] Yu Wang and Xi Chen. Mirix: Multi-agent memory system for llm-based agents, 2025.
- <span id="page-15-12"></span>[57] Zhenhailong Wang, Haiyang Xu, Junyang Wang, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, and Heng Ji. Mobile-agent-e: Self-evolving mobile assistant for complex tasks, 2025.
- <span id="page-15-6"></span>[58] Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. In *Forty-second International Conference on Machine Learning*, 2025.
- <span id="page-15-7"></span>[59] Hao Wen, Yuanchun Li, Guohong Liu, Shanhui Zhao, Tao Yu, Toby Jia-Jun Li, Shiqi Jiang, Yunhao Liu, Yaqin Zhang, and Yunxin Liu. Autodroid: Llm-powered task automation in android. In *Proceedings of the 30th Annual International Conference on Mobile Computing and Networking*, ACM MobiCom '24, page 543–557, New York, NY, USA, 2024. Association for Computing Machinery.
- <span id="page-15-17"></span>[60] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversations. In *First Conference on Language Modeling*, 2024.
- <span id="page-15-15"></span>[61] Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang, Yongyue Zhang, Huifeng Guo, Ruiming Tang, and Yong Liu. From human memory to ai memory: A survey on memory mechanisms in the era of llms. *arXiv preprint arXiv:2504.15965*, 2025.
- <span id="page-15-13"></span>[62] Zhiyong Wu, Chengcheng Han, Zichen Ding, Zhenmin Weng, Zhoumianze Liu, Shunyu Yao, Tao Yu, and Lingpeng Kong. Os-copilot: Towards generalist computer agents with self-improvement, 2024.
- <span id="page-15-2"></span>[63] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, and Yu Qiao. Os-atlas: A foundation action model for generalist gui agents, 2024.
- <span id="page-15-11"></span>[64] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. In *Advances in Neural Information Processing Systems*, 2025.
- <span id="page-15-4"></span>[65] Yifan Xu, Xiao Liu, Xinghan Liu, Jiaqi Fu, Hanchen Zhang, Bohao Jing, Shudan Zhang, Yuting Wang, Wenyi Zhao, and Yuxiao Dong. Mobilerl: Online agentic reinforcement learning for mobile gui agents, 2025.

- <span id="page-15-8"></span>[66] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In *International Conference on Learning Representations (ICLR)*, 2023.
- <span id="page-15-0"></span>[67] Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, Jitong Liao, Qi Zheng, Fei Huang, Jingren Zhou, and Ming Yan. Mobile-agent-v3: Fundamental agents for gui automation, 2025.
- <span id="page-15-18"></span>[68] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. Orca: A distributed serving system for transformer-based generative models. In *16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)*, pages 521–538, 2022.
- <span id="page-15-10"></span>[69] Wenwen Yu, Zhibo Yang, Jianqiang Wan, Sibo Song, Jun Tang, Wenqing Cheng, Yuliang Liu, and Xiang Bai. Omniparser v2: Structured-points-of-thought for unified visual text parsing and its generality to multimodal large language models, 2025.
- <span id="page-15-1"></span>[70] Zapier. Zapier: Automate ai workflows, agents, and apps. <https://zapier.com/>, 2025. Accessed: 2025-12-09.
- <span id="page-15-14"></span>[71] Chaoyun Zhang, He Huang, Chiming Ni, Jian Mu, Si Qin, Shilin He, Lu Wang, Fangkai Yang, Pu Zhao, Chao Du, Liqun Li, Yu Kang, Zhao Jiang, Suzhen Zheng, Rujia Wang, Jiaxu Qian, Minghua Ma, Jian-Guang Lou, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. Ufo2: The desktop agentos, 2025.
- <span id="page-15-16"></span>[72] Cheng Zhang, Erhu Feng, Xi Zhao, Yisheng Zhao, Wangbo Gong, Jiahui Sun, Dong Du, Zhichao Hua, Yubin Xia, and Haibo Chen. Mobiagent: A systematic framework for customizable mobile agents, 2025.
- <span id="page-15-9"></span>[73] Jiwen Zhang, Jihao Wu, Teng Yihua, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. Android in the zoo: Chain-of-action-thought for GUI agents. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, *Findings of the Association for Computational Linguistics: EMNLP 2024*, pages 12016–12031, Miami, Florida, USA, November 2024. Association for Computational Linguistics.
- <span id="page-15-3"></span>[74] Zhong Zhang, Yaxi Lu, Yikun Fu, Yupeng Huo, Shenzhi Yang, Yesai Wu, Han Si, Xin Cong, Haotian Chen, Yankai Lin, Jie Xie, Wei Zhou, Wang Xu, Yuanheng Zhang, Zhou Su, Zhongwu Zhai, Xiaoming Liu, Yudong Mei, Jianming Xu, Hongyan Tian, Chongyi Wang, Chi Chen, Yuan Yao, Zhiyuan Liu, and Maosong Sun. Agentcpm-gui: Building mobile-use agents with reinforcement fine-tuning, 2025.

- <span id="page-16-1"></span>[75] Lianmin Zheng, Zhuohan Li, Hao Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping Huang, Yida Wang, Yuanzhong Xu, Danyang Zhuo, Eric P Xing, et al. Alpa: Automating inter-and intra-operator parallelism for distributed deep learning. In *16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)*, pages 559–578, 2022.
- <span id="page-16-2"></span>[76] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language model programs. *Advances in neural information processing systems*, 37:62557–62583, 2024.
- <span id="page-16-0"></span>[77] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, pages 19724–19731, 2024.
- <span id="page-16-3"></span>[78] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. Distserve: disaggregating prefill and decoding for goodputoptimized large language model serving. In *Proceedings of the 18th USENIX Conference on Operating Systems Design and Implementation*, OSDI'24, USA, 2024. USENIX Association.

# A User Profile Benchmark

We design a user profile benchmark to evaluate whether agents can accurately learn and model user profiles from historical task execution traces. The benchmark consists of 20 synthetic user profiles, each containing textual descriptions across various aspects of user behavior including facts (e.g., home address, work location), preferences (e.g., shopping habits, hotel booking preferences), and past experiences (e.g., previously visited places, completed activities). For each user, we generate 500 historical tasks that explicitly reflect the user's profile information, and 30 test tasks with ambiguous descriptions that require agents to infer missing details based on learned profile memory. User profiles span 5 task categories (shopping, hotel booking, travel, food delivery, entertainment) with over 30 profile dimensions in total. This section provides detailed information about benchmark design, task formats, LLM-based task rewriting, and evaluation methodology.

# A.1 Benchmark Design

### A.1.1 Design Rationale

The benchmark evaluates three core capabilities of agent systems. First, agents must extract structured profile information from task execution traces containing implicit signals about user facts, preferences, habits, and experiences. Second, agents must store this information in memory systems that enable efficient retrieval. Third, agents must apply learned profiles to complete ambiguous tasks by inferring missing task parameters.

### A.1.2 Data Structure

Each user profile contains structured information across five task categories: shopping, hotel booking, travel, food delivery, and entertainment. Within each category, profiles specify 3–7 dimensions covering various aspects such as preferences, habits, constraints, and contextual information. Historical tasks explicitly encode user profile information through natural language instructions. Test tasks are intentionally ambiguous, omitting specific profile details to require agents to retrieve and apply learned profile memory.

## A.2 Task Format and Examples

#### A.2.1 Historical Task Format

Historical tasks are natural language instructions that explicitly contain user profile information. Each task is a complete sentence describing a specific action, with profile information (preferences, habits, constraints) embedded naturally.

Example historical tasks for a budget-conscious user:

"I need to buy a keychain organizer on Taobao—looking for something affordable from a domestic brand, beige color preferred, under 20 yuan, and I'd like next-day delivery if possible."

"Can you help me find high-speed rail tickets from Chengdu to Luzhou on the 12306 app? I'm visiting relatives and prefer a direct train around 11 AM, second-class seat is fine to keep costs down. I usually book three days ahead."

"I'd like to order a light lunch on Eleme—Chinese food, but not too spicy. Something budget-friendly with beef brisket and lots of vegetables would be perfect. Please have it delivered during my lunch break."

These tasks contain explicit profile signals including price preferences ("affordable", "under 20 yuan", "keep costs down", "budget-friendly"), brand preferences ("domestic brand"), style preferences ("beige color", "not too spicy"), timing constraints ("visiting relatives", "lunch break"), and behavioral patterns ("I usually book three days ahead").

### A.2.2 Shared Test Task Format

Shared test tasks are intentionally ambiguous, containing only high-level task descriptions without specific profile details. Agents must infer and apply learned profile information to complete these tasks.

### Example shared test tasks:

"I'm heading to Guangzhou tomorrow to meet some friends. Can you book me a train ticket on 12306, find a hotel on Ctrip near good transit options for one night, and order some household items on Taobao that I can bring with me?"

"Planning to relax at home this weekend. Could you order some Chinese food for me on Meituan, find some interesting short videos on Bilibili, and maybe get me a comfy cushion from Taobao?"

These tasks are ambiguous in several ways. Price constraints are unspecified: "book me a train ticket" does not indicate seat class or budget. Brand and style information is missing: "order some household items" does not specify brand, quality level, or style. Timing and context are unclear: "relax at home this weekend" does not provide exact delivery windows, viewing schedules, or personal taste information.

## A.3 LLM-Based Task Rewriting

To improve the realism of historical tasks, we employ an LLMbased rewriting process that transforms template-generated tasks into natural user utterances. The rewriting process takes structured task templates and produces conversational language while preserving all profile signals embedded in the original tasks.

### A.3.1 Rewriting Process

The rewriting process takes original tasks generated from templates along with user profile summaries as input. The LLM is then prompted to rewrite these tasks into more natural language while preserving the core task intent and all embedded profile information. The process produces rewritten tasks that maintain all profile signals but express them through conversational language that resembles authentic user requests.

#### Example rewriting:

Original: "Search for a pillow on Taobao, choose domestic brand and lowest price".

Rewritten: "I'm looking for an affordable pillow on Taobao—preferably memory foam from a domestic brand. Something really comfortable in beige would be great, and my budget is around 100 yuan. If you can get next-day delivery, that'd be perfect!"

The rewritten version maintains core profile signals (price, brand, style, delivery, timing), uses more natural and conversational language, adds contextual details that make the request sound authentic, and preserves all required profile dimensions while sounding like a real user query.

## A.4 Evaluation Methodology

### A.4.1 Evaluation Workflow

The evaluation follows a three-phase workflow.

Phase 1: Learning. The agent processes historical tasks (500 per user), extracts profile information and stores it in memory, and learning is evaluated by checking if extracted profile information matches the ground-truth profile.

Phase 2: Task Rewriting. The agent receives shared test tasks (30 ambiguous tasks), retrieves relevant learned profile information for each task, uses LLM to rewrite the ambiguous task into a personalized complete task description, and the rewritten task should incorporate learned profile information naturally.

Phase 3: Profile Matching. For each rewritten task, an LLM judge evaluates whether required profile information is present, compares the rewritten task against the ground-truth profile, and computes metrics including matched information count, total information count, and alignment score.

#### A.4.2 Required Profile Information Generation

To determine what profile information should be present in each test task, we employ an LLM-based analysis process. For each user, the process takes three inputs: the user's groundtruth profile, the user's historical task history, and the shared test tasks. The LLM analyzes each test task and identifies

which specific profile elements are necessary to complete that task for the given user.

### Example analysis prompt template:

You are an expert in user profile analysis. For each task in the shared task list, determine what profile information is required to complete that task.

#### User Profile:

[Coarse-grained profile information across different categories]

#### User Task History:

[Sample historical tasks showing fine-grained behavior patterns]

#### Shared Tasks:

[Test tasks to analyze, numbered 1, 2, 3, ...]

#### Output:

For each task, identify the required profile information:

```
Task 1: { element_1: value, ... }
Task 2: { element_1: value, ... }
...
```

This analysis produces a structured mapping from each test task to its required profile information, represented as key-value pairs (e.g., "price\_constraint: budget-friendly", "brand\_preference: domestic brand"). This approach ensures that evaluation is context-aware: different test tasks may require different subsets of profile information, and the required information is determined based on both the task nature and the user's historical behavior patterns.

### A.4.3 Evaluation Metrics

We evaluate systems using three metrics: write latency (time to update profile from each historical task), retrieval latency (time to retrieve profile information for each test task), and profile alignment (percentage of ground-truth profile information successfully retrieved, evaluated by an LLM judge).

For profile alignment evaluation, the system first identifies which profile information should be present in each test task based on task type and ground-truth profile. An LLM judge then evaluates whether the agent's rewritten task contains these required profile elements.

#### Example evaluation prompt template:

You are an evaluator assessing whether a personalized task reflects required user profile information. Determine if the following profile elements are present in the rewritten task.

#### Required Profile Information:

[Profile information based on user profile and task context]

#### Personalized Rewritten Task:

[Rewritten task text]

#### Instructions:

Analyze the rewritten task and determine whether each profile element is clearly reflected.

The judge returns a structured JSON response:

```
{
  "profile_check": [
    {
      "profile_element": "...",
      "expected_value": "...",
      "matched": true/false,
      "evidence": "..."
    },
    ...
}
```

Each element includes the profile element name, expected value (from user profile), a boolean "matched" field indicating whether the profile information is reflected in the task, and an "evidence" field providing justification.

Computed metrics: We compute per-task alignment as the percentage of required profile elements matched in each rewritten task, overall alignment as the average alignment across all 30 test tasks, and per-dimension accuracy as the matching accuracy for each profile dimension. These metrics quantify both the completeness and accuracy of profile learning and retrieval.

# B Multi-Task Execution in Real-World Scenarios

To complement the performance evaluation presented in our main paper, this appendix provides detailed descriptions of the real-world testing scenarios used to evaluate the Agent Scheduler's multi-task execution capabilities. These scenarios involve complex workflows requiring data transfer and synchronization across multiple applications, representing typical user interactions in practical mobile usage. We measure endto-end execution latency across three execution modes: Serial, Coarse-grained Parallel, and Fine-grained Parallel.

## B.1 Execution Modes in Practice

We use a representative shopping and social networking scenario to illustrate how the three execution modes handle realworld tasks differently.

Scenario: Query the price of a specific item in two different shopping applications (App A and App B), and then send the gathered information to a contact via a social networking application (App C).

#### B.1.1 Serial Execution

The agent executes tasks strictly sequentially. It first completes the price query in App A, then performs the price query in App B, and finally launches App C to send the message.

*Timeline:* 
$$T_{total} = T_{AppA} + T_{AppB} + T_{AppC}$$

#### B.1.2 Coarse-grained Parallelism

The agent identifies independent sub-tasks at the application level. It executes the price queries in App A and App B simultaneously. The system blocks the execution of App C until both App A and App B have fully completed their tasks and returned the results.

*Timeline: Ttotal* = max(*TAppA*,*TAppB*) +*TAppC*

#### B.1.3 Fine-grained Parallelism

The agent exploits step-level parallelism by analyzing data dependencies. While the price query sub-tasks in shopping apps (App A and App B) are being executed, the agent simultaneously operates in App C (e.g., searching for the contact, entering the chat interface, and activating the input field). The execution in App C is suspended *only* at the specific step where the message content (the prices) is required. Once the query results from App A and App B become available, the execution in App C resumes immediately to send the message.

*Timeline: Ttotal* = max(*TAppA*,*TAppB*,*TAppC*\_*setup*) + *TAppC*\_*send*

# B.2 Real-World Application Coverage

To ensure comprehensive evaluation across diverse usage patterns, we tested the system with 25 mainstream applications covering Social, Shopping, Information, and Video domains. These applications represent the most commonly used mobile apps in real-world scenarios.

### B.2.1 Application Categories

The tested applications span four functional categories corresponding to typical user activities. Table [5](#page-19-0) lists the applications in each category.

#### B.2.2 Task Scenarios

We select 6 distinct task scenarios that represent common real-world multi-app workflows. These scenarios range from simple sequential operations to complex parallel coordination tasks. By varying the specific applications used within each

Table 5: Real-world applications tested across four functional categories.

<span id="page-19-0"></span>

| Category | Applications & Description                 |
|----------|--------------------------------------------|
| Social   | WeChat, QQ, Sina Weibo (instant messaging  |
|          | & social networking)                       |
| Shop     | Taobao, JD.com, Pinduoduo, Xianyu, Ele.me, |
|          | Meituan (e-commerce & services)            |
| Search   | Xiaohongshu, Zhihu, Toutiao, Dianping,     |
|          | Browser (info retrieval & tools)           |
| Video    | Bilibili, iQIYI, Tencent<br>Video, Youku,  |
|          | Douyin, Kuaishou (video streaming)         |

| Scenario           | Logic & Example Instruction                 |
|--------------------|---------------------------------------------|
| search+shop+social | Serial/Pipeline:<br>Find<br>recommended     |
|                    | 2025 Canon cameras on Xiaohong              |
|                    | shu, search on Taobao, send details via     |
|                    | WeChat.                                     |
| multi-video+social | Parallel Query: Check iQIYI, Tencent        |
|                    | Video for Joy of Life 3 updates, notify via |
|                    | WeChat if found.                            |
| multi-shop+social  | Parallel Comparison: Query DJI Ac           |
|                    | tion 5 price on Taobao and JD.com, send     |
|                    | comparison via WeChat.                      |
| single-shop+social | Simple Pipeline: Query DJI Action 5         |
|                    | price on Taobao, send result via WeChat.    |
| search+social      | Info Sharing: Search Disney Christmas       |
|                    | event dates on Xiaohongshu, send sched      |
|                    | ule via WeChat.                             |
| search+shop        | Decision & Action: Find best-rated          |
|                    | Sony headphones under 1000 CNY on           |
|                    | Zhihu, order on Taobao.                     |

<span id="page-20-0"></span>Table 6: Task scenarios representing real-world usage patterns.

category and adjusting task parameters, we create a total of 50 test instances covering diverse usage patterns.

Table [6](#page-20-0) describes the task scenarios and provides example instructions.

#### B.2.3 Test Instance Construction

To ensure the evaluation reflects realistic usage diversity, we construct the 50 test instances according to the following principles:

Application Variation: For each task scenario, we randomly selected compatible applications within each category to ensure the agent handles different UI designs and interaction patterns. For example, shopping tasks alternated between Taobao, JD.com, and Pinduoduo.

Parameter Diversity: We varied search keywords, product names, target prices, and contact names across different instances to prevent result caching and ensure genuine task execution.

Cross-Category Coordination: We included tasks linking Video and Social apps (e.g., Douyin + QQ), Shopping and Search apps (e.g., Taobao + Zhihu), and Information Retrieval and Shopping apps (e.g., Xiaohongshu + Taobao) to test the scheduler's ability to handle data passing between diverse application architectures.

## B.3 Evaluation Methodology

For each task instance, we execute the workflow using all three execution modes. The primary metric is the end-toend latency (seconds), measured from the moment the user issues the command until the final action (e.g., message sent or order placed) is confirmed by the system. We calculate the speedup of the parallel modes relative to the serial baseline to quantify the efficiency improvements provided by the Agent

Scheduler. Each task instance is executed multiple times to ensure measurement reliability, and we report the average latency across runs.