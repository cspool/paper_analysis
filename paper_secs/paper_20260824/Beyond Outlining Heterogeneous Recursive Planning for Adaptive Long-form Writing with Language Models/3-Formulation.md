# 3 Formulation

In this section, we formulate the fundamental components of a long-form writing agent system, focusing on three heterogeneous task types essential for writing: retrieval (information gathering), reasoning (content planning), and composition (text generation). We further formalize the writing planning problem with a conceptual framework inspired by the hierarchical task network planning.

#### 3.1 Writing Agent System

We first introduce the notion of the writing agent system.

Definition 3.1 (Writing Agent System). A *writing agent system* is a tuple

$$\Sigma_{\mathcal{A}} = (\mathcal{A}, \mathcal{M}, D, W),$$

where A is the *agent kernel* responsible for processing writing instructions, solving writing tasks, and selecting actions. M is the *internal memory* maintaining writing-related information like outlines, drafted content, and retrieved references. D is the *database* (e.g., search engine, reference documents) and W is the writing *workspace*.

### <span id="page-2-0"></span>3.2 Task Types

The writing process naturally involves three types of heterogeneous cognitive tasks: retrieval for information gathering, reasoning for content planning, and composition for content generation. This categorization aligns with cognitive models of agents [\(Sumers et al.,](#page-10-11) [2024\)](#page-10-11) and reflects the distinct operational patterns in writing tasks.

Definition 3.2 (Retrieval Task). Let i be the information needs during writing (e.g., factual queries, reference searches). A *retrieval task* ta(i) for aims

<span id="page-3-0"></span>> **[图片提取文字 (无描述)]:**
> Reasoning  $t_r(p, K)$ Retrieval Composition  $t_c(s, g, K)$  $t_a(i)$  $\mathcal{M} \ni K$ W = sDatabase Memory Workspace
![](_page_3_Figure_0.jpeg)

Figure 2: The abstract flow of tasks. The arrow indicates the information flow of a task: the system state at the arrowhead is modified by the labeled task, while the hollow circle end signifies that the associated system state remains unchanged.

to acquire relevant information from the environment and update it into the agent's memory M.

Definition 3.3 (Reasoning Task). Let p represent a writing-related problem requiring logical inference (e.g., outline planning, content organization). A *reasoning task* tr(p, K) aims to derive new knowledge or make decisions based on available information K in agent's internal memory M.

Definition 3.4 (Composition Task). Let g represent the text generation objective specifying target states of the written content. A *composition task* tc(s, g, K) aims to generate text that meets specified requirements (e.g., style, length, structure) through a sequence of writing actions, given current workspace state s and knowledge K ∈ M.

We illustrate the abstract flow of the three tasks in Figure [2.](#page-3-0) Retrieval Task functions as contextindependent operations that enhance working memory without modifying the workspace; Reasoning Task performs memory-to-memory transformation contingent upon satisfaction of logical preconditions; and Composition Task executes workspacealtering operations and then updates related information to the memory.

#### 3.3 Planning for Writing

Planning for writing is based on the assumption that the writing process as complex tasks composed by simpler, executable subtasks. This perspective follows HTN planning, where the objective is not to achieve a set of goals but instead to perform some set of primitive tasks.

In the context of writing, primitive tasks are the basic actions that can be executed directly by the agent. Breaking down complex tasks into these primitives improves accuracy [\(Chen et al.,](#page-9-5) [2024a\)](#page-9-5) and allows flexible action interleaving. By assuming a theoretical set T<sup>p</sup> of primitive tasks (without explicitly specifying its composition), we formulate the writing planning problem as follows.

Definition 3.5 (Writing Planning Problem). A *writing planning problem* is a tuple

$$\langle t_c(g, s_0, K_0), T_p \rangle$$

where tc(g, s0, K0) is the top-level composition task, with a writing goal g, the initial state of the writing workspace s0, and the initial content of the agent's memory K0. T<sup>p</sup> is the set of executable primitive retrieval, reasoning and composition tasks. A solution π = ⟨t1, t2, . . . , tk⟩ to this planning problem is a sequence of primitive tasks that achieves the writing objective while maintaining coherence and satisfying constraints.

### 4 Heterogeneous Recursive Planning

Based on the formulation of the writing task planning problem, we propose a heterogeneous recursive planning method (HRP) inspired by the HTN planning and the heterogeneity of the three cognitive tasks. In this section, we introduce the key components of our approach.

#### 4.1 Recursive Planning

The classical HTN planning paradigm solves problems through hierarchical decomposition until reaching primitive executable operations. Following our formulation of the writing planning problem, we adopt a recursive planning strategy, in alignment with classical HTN approaches.

The core of this planning process is task decomposition: each task is broken down into subtasks, and the same decomposition logic is recursively applied to those subtasks. Unlike traditional asneeded decomposition methods that rely on execution failure to stop further planning, our approach introduces a different termination criterion. We only continue planning if certain subtask types necessitate further decomposition, ensuring that the final operations are always executable without redundant decomposition.

#### <span id="page-3-1"></span>4.2 Typed Task Integration

Building upon our formal characterization of cognitive task types in Section [3.2,](#page-2-0) we extend the recursive planning framework with type-aware decomposition mechanisms.

Our integration addresses the cognitive heterogeneity inherent in writing processes. While complex tasks may involve blended operations, their decomposition should respect the dominant cognitive type based on primary objectives. We formalize this as:

**Hypothesis** (Type Specification in Decomposition). During hierarchical decomposition of writing tasks, all generated subtasks can be specified as exactly one cognitive type.

This hypothesis suggests that the writing planning problem can be decomposed into sub-planning problems of three distinct task types. For example, assume task  $t_c(g, s_0, K_0)$  can be decomposed into a sequential combination of subtasks  $t_a(i)$ ,  $t_r(p, K')$ , and  $t_c(g, s_0, K'')$ , where K' and K'' denote the modified knowledge in  $\mathcal M$  after executing the preceding tasks. The solution of  $\langle t_c(g, s_0, K_0), T_p \rangle$  is then the combination of solutions of planning problems  $\langle t_a(i), T_p \rangle$ ,  $\langle t_r(p,K'),T_p\rangle$ , and  $\langle t_c(g,s_0,K''),T_p\rangle$ . These solutions must satisfy their corresponding executability conditions and goal achievement criteria. For instance, subtasks of composition may include retrieval or reasoning tasks to modify the internal memory. They must have a composition-type subtask to reach the goal.

Motivated by the above analysis, we integrate task types into the planning procedure. Our method features the following key design elements:

- Dynamic type annotation: Each subtask generated in a planning step is assigned a specific type. It facilitates the function call of heterogeneous agents, for example, a search agent to conduct a retrieval task.
- **Type-aware decomposition**: This provides targeted guidance for potential subtask breakdowns based on the type of the current task.

#### 5 WriteHERE Framework

We propose WriteHERE, an adaptive writing framework that integrates HRP with state-based hierarchical task scheduling, implemented using structural memory and graph-based context control. We summarize its core logic in Algorithm 1 and introduce the key concepts below. A detailed walkthrough with a specific example is provided in Appendix D.

### <span id="page-4-0"></span>Algorithm 1 WriteHERE framework

**Require:** Memory  $\mathcal{M} = (G, W)$ : Task Graph

```
G = (V, E) with root V_{\text{init}} = \{v_{\text{root}}\};
     Workspace W; Initial state S(v_{\text{root}}) \leftarrow
     ACTIVE
Ensure: S(v) = SILENT, \forall v \in V
 1: while \exists v \in V \mid S(v) \neq SILENT do
        Select v^* \leftarrow \arg\min_{v \in V} \{ BFS\text{-depth}(v) \mid
        S(v) = ACTIVE
        Get knowledge K \leftarrow \text{GETINFO}(\mathcal{M}, v^*)
 3:
        v^* \leftarrow \mathsf{Update}(v^*, K)
        if IsAtomic(v^*, K) then
 4:
 5:
           M \leftarrow \mathsf{Execute}(v^*, K) /\!\!/ \mathsf{Differs depend}
           ing on task type
           S(v^*) \leftarrow \text{SILENT}
 6:
 7:
        else
           \{v_1,\ldots,v_k\}\leftarrow \mathsf{TypedPlan}(v^*,K)
 8:
           ADDCHILDREN(G, \{v_1, ..., v_k\}, v^*)
 9:
           S(v^*) \leftarrow SUSPENDED
10:
        end if
11:
        Update S(v) for all v in V to {SILENT, SUS-
12:
        PENDED or ACTIVE}
13: end while
```

**Task graph.** Tasks and their dependencies are modeled as a directed acyclic graph G=(V,E). Each node is denoted with the type, goal, dependencies information and execution result of it. The graph G starts with a single root node with  $g_{root}$  describing the user input request and  $t_{root}$  defined as composition. G is dynamically expanded and updated throughout the process.

State-based hierarchical task scheduling. Our approach interleaves task execution with planning, enabling adaptive planning that responds to action outcomes through a hierarchical task scheduling algorithm. The algorithm manages dynamic task decomposition through assigning one of the three states to each task node v, denoted as S(v): AC-TIVE, SUSPENDED, or SILENT. A task is SUS-PENDED while its prerequisites are incomplete or after it has been decomposed into subtasks. It becomes ACTIVE only when all prerequisites are met, marking it ready for processing. Upon completion, a task transitions to the SILENT state. Starting from the root, the algorithm iteratively selects ACTIVE task nearest to the root with BFS-based topological sorting. The selected task is either executed directly (if primitive) or decomposed into subtasks which are then integrated into the graph. This process continues until all tasks reach the SILENT state, ensuring the systematic completion of the entire task hierarchy.

Memory and context control. The memory M of our agent system consists of task graph G and the workspace W. This memory does not serve as the complete context for planning or subtask execution; instead, relevant knowledge is retrieved through a context control module. As introduced in Section [4.2,](#page-3-1) the knowledge context of a decomposed subtask is determined by the knowledge context of its parent task and the execution results of its preceding tasks. Our context control strategy adheres to this principle. For each task node, the framework constructs task-specific knowledge comprising the current workspace state and relevant task graph information, including node information from parent nodes up to a specified depth and precedent nodes on which it depends. Additionally, the planning modules (IsAtomic and TypedPlan) receives global structural information about G, including the goals, types, and dependencies of all nodes. We abstract this logic as GETINFO(M, v) in Algorithm [1.](#page-4-0)

LLM operations. The framework prompting LLMs for the following core operations: updates the task goals, determines the primitivity of the task, execute the primitive task, and generate the typed plan. Specifically, the Update module refines the goal of the selected task node based on the related knowledge. The IsAtomic module then employs an LLM to determine if a task is atomic (i.e. primitive, directly executable) or complex (requiring decomposition). If a task is complex, the TypedPlan module decomposes the goal into a structured list of subtasks. To ensure validity, this process employ structured prompting to constrain the LLM's output format and apply programmatic validation rules to detect and correct dependency errors, guaranteeing robust execution. The Execute module invokes specialized executors for different primitive task types. Specifically, the composition executor generates text segments, while the reasoning executor produces structured analyses or decisions. The retrieval executor returns a summary of the retrieved information.

### 6 Experiments

We evaluate our approach through experiments on two challenging long-form writing tasks: narrative generation and report generation. Our investigation addresses three key aspects: (1) the comparative performance of our method against state-of-the-art baselines, (2) the impact of the recursive planning and task-type module, and (3) the generalization capability across diverse task domains.

## 6.1 Narrative Generation

Narrative generation involves complex reasoning and composition tasks. We use the TELL ME A STORY fiction writing dataset proposed in the paper of Agent's Room [\(Huot et al.,](#page-9-0) [2024\)](#page-9-0).

Datasets. TELL ME A STORY offers a collection of complex, well-structured narratives paired with detailed narrative generation prompts. The dataset consists of 230 samples, with each prompt averaging 113 tokens and corresponding narrative responses averaging 1,498 tokens.

Baselines. We implement two primary baselines: (1) End-to-End (E2E): where we directly provide the story prompt to the base LLM without any additional guidance or planning steps; and (2) Agents' Room [\(Huot et al.,](#page-9-0) [2024\)](#page-9-0): a collaborative writing framework with multiple agents that decomposes the story generation process into planning and writing phases. In the planning phase, specialized agents outline key story elements including plot structure, character development, and setting details. Writing agents then generate the full narrative following this structured plan.

Evaluation metrics. We adopt the LLM-based evaluator for story assessment proposed by [Huot](#page-9-0) [et al.](#page-9-0) [\(2024\)](#page-9-0), which demonstrates strong correlation with human judgments (Spearman's rank correlation ρ = 0.62, p < 0.01). For each story pair, the evaluator determines which is superior or equivalent across these dimensions and overall, producing win-tie-loss judgments. To convert these pairwise comparisons into quantitative scores, we employ the Davidson model [\(Davidson,](#page-9-6) [1970\)](#page-9-6), which effectively handles cases with ties. Following the practice of [Huot et al.](#page-9-0) [\(2024\)](#page-9-0), we implement the evaluator using Gemini (2.0-Flash) as the base LLM. To mitigate position bias, we conduct 7 evaluations in each ordering (14 total trials) and determine the final outcome through majority voting.

Configurations. For Agent's Room baseline, we implement the plan+write version according to the paper, which includes 4 planning agents (conflict,

<span id="page-6-0"></span>

| Backbones         | Methods       | Dimensions |            |             |              |         |  |  |
|-------------------|---------------|------------|------------|-------------|--------------|---------|--|--|
|                   |               | Plot       | Creativity | Development | Language Use | Overall |  |  |
|                   | E2E           | 0.337      | 0.218      | 0.288       | 0.202        | 0.270   |  |  |
|                   | Agent's Room  | 1.035      | 0.712      | 0.948       | 0.680        | 0.869   |  |  |
| GPT-4o            | WriteHERE     | 1.470      | 2.005      | 1.967       | 2.233        | 2.143   |  |  |
|                   | w/o Recursive | 1.307      | 1.327      | 1.041       | 1.192        | 1.100   |  |  |
|                   | w/o Type      | 0.852      | 0.733      | 0.756       | 0.693        | 0.717   |  |  |
|                   | E2E           | 0.036      | 0.016      | 0.032       | 0.017        | 0.025   |  |  |
|                   | Agent's Room  | 1.029      | 0.480      | 0.778       | 0.484        | 0.694   |  |  |
| Claude-3.5-Sonnet | WriteHERE     | 2.016      | 2.634      | 2.959       | 2.264        | 2.852   |  |  |
|                   | w/o Recursive | 1.145      | 1.396      | 0.707       | 1.517        | 0.918   |  |  |
|                   | w/o Type      | 0.774      | 0.475      | 0.525       | 0.518        | 0.512   |  |  |

Table 1: Quantitative strength scores of methods on the TELL ME A STORY dataset. The scores are derived from pairwise comparisons of all generated stories, with the final relative strength calculated using the Davidson model. This score is non-linear; improvements at the higher end of the scale are progressively more challenging. Ablations of our method are highlighted in grey. The highest value in each column is in bold.

<span id="page-6-1"></span>> **[图片提取文字 (无描述)]:**
> 2K 8K 4K 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0 creativ. develop. language overall creativ. develop. language overall creativ. develop. language overall plot plot plot win tie loss
![](_page_6_Figure_2.jpeg)

Figure 3: The evaluation results of WriteHERE v.s. Agent's Room at different generation lengths.

character, setting, plot) and 5 writing agents (exposition, rising action, climax, falling action, resolution). We use a length estimator along with the writing agents to enable the length control. For our method, two task types are included: reasoning (Design) and composition (Writing). We implement a Design agent and a Writing agent as the primitive task executors.

### 6.1.1 Results

As shown in Table [1,](#page-6-0) Agent's Room significantly outperforms the E2E baseline, aligning with results reported in their original paper. Our proposed method demonstrates superior performance across all five key evaluation metrics compared to baseline approaches. This consistent improvement holds across two different backbone LLMs, validating the robustness of our approach across base models.

Ablation study. To analyze the contributions of individual components, we conducted an ablation study with two key variations: 1) Non-recursive

generation ("w/o Recursive"): This variant removes the recursive decomposition process, instead generating the entire plan in a single step similar to baseline methods. 2) Task-type removal ("w/o Type"): This variant omits explicit task-type information during decomposition. While still employing recursive breakdown, the model no longer utilizes type-specific decomposition logic.

Extended lengths. We also evaluated how different methods scale with increasing generation length. From our dataset, we selected 60 samples that an LLM identified as suitable for generating texts over 8,000 words. We then conducted experiments by prompting models to generate articles of three different lengths: 2K, 4K, and 8K words, operating under the assumption that task complexity increases with required text length. Figure [3](#page-6-1) presents pairwise comparisons of the overall metric between our method and Agents Room with GPT-4o as the base LLM. We excluded the E2E baseline from this comparison as it is unable to

<span id="page-7-1"></span>

| Backbones         | Methods           | Report Quality |         |       |         |  |
|-------------------|-------------------|----------------|---------|-------|---------|--|
|                   |                   | Relevance      | Breadth | Depth | Novelty |  |
|                   | STORM             | 4.76           | 4.58    | 4.30  | 4.32    |  |
|                   | Co-STORM          | 4.36           | 4.22    | 4.02  | 4.17    |  |
| GPT-4o            | WriteHERE         | 4.93           | 4.86    | 4.79  | 4.51    |  |
|                   | w/o HRP           | 4.83           | 4.18    | 3.74  | 4.17    |  |
|                   | STORM             | 4.66           | 4.63    | 4.40  | 4.41    |  |
|                   | Co-STORM          | 3.87           | 3.56    | 3.46  | 3.82    |  |
| Claude-3.5-Sonnet | WriteHERE         | 4.96           | 4.92    | 4.93  | 4.82    |  |
|                   | w/o HRP           | 4.84           | 4.51    | 4.24  | 4.46    |  |
|                   | WriteHERE         | 4.97           | 4.94    | 4.95  | 4.88    |  |
| DeepSeek-R1       | w/o HRP           | 4.94           | 4.81    | 4.83  | 4.80    |  |
| Commercial        | PPL-Deep Research | 4.93           | 4.73    | 4.75  | 4.45    |  |

Table 2: Comparison of method performance on WildSeek, evaluated by o1-preview. The scores represent absolute grades on a 1-5 scale based on a detailed rubric. Our method and its ablations are highlighted with a grey background.

generate texts of 4K or 8K words. For 2,000-word stories, our method and Agents Room performed comparably on more than 50% of samples. However, our method demonstrates increasingly significant advantages over the baseline as task length increases, highlighting its effectiveness in handling more complex long-form content generation.

#### 6.2 Report Generation

Compared with story generation, report generation task further need the integration of complex retrieval tasks with reasoning and composition. We employed a hybrid evaluation strategy to balance rigor, scale, and alignment with existing benchmarks. Specifically, we used LLM-based evaluation to enable large-scale pairwise comparisons and human evaluation for the most challenging and complex reports over 10,000 words.

Datasets. We use the WildSeek dataset proposed by [\(Jiang et al.,](#page-9-3) [2024\)](#page-9-3). WildSeek offers a collection of real-world information-seeking tasks paired with user goals for evaluating complex information retrieval capabilities. The dataset consists of 100 samples across 24 domains, collected from users of the STORM web application. Each data point comprises a Topic-Intent sentence pair.

Baselines. We compare our method with STORM [\(Shao et al.,](#page-10-1) [2024\)](#page-10-1) and Co-STORM [\(Jiang](#page-9-3) [et al.,](#page-9-3) [2024\)](#page-9-3). STORM is a writing system that uses perspective-guided question asking from retrieval and constructs Wikipedia-like articles through generating outlines and section-by-section writing. Co-STORM extends STORM by introducing a user-participated roundtable discussion to enhance the diversity of retrieved information and improve coverage of unknown unknowns. Both baseline methods rely on retrieval-augmented generation and use similar outline-driven approaches for long-form text generation.

Evaluation metrics. We utilize the evaluation framework established by Co-STORM, which examines the final report across four dimensions: Relevance, Broad Coverage (Breadth), Depth, and Novelty. A LLM-based evaluator assesses each dimension on a 5-point scale, with the original Topic and Intent provided. We employ the latest OpenAI o1-preview as our primary evaluator model.

Configurations. We use Bing Search API for retrieval. We use the latest official implementation of STORM[1](#page-7-0) with their default configurations. For Co-STORM, we follow the official implementation with its user-simulator. We design a search agent, an analyzing agent, and a writing agent as the primitive task executors for retrieval, reasoning and composition respectively. For the search agent, we implement a multi-agent framework comprising a retrieval agent, a reranking agent, and a summarization agent. See Appendix [C](#page-19-0) for more details.

## 6.2.1 Results

Our primary experiment on the WildSeek dataset is presented in Table [2.](#page-7-1) The results demonstrate

<span id="page-7-0"></span><sup>1</sup> https://github.com/stanford-oval/storm

that our method consistently outperforms the current state-of-the-art approaches across four distinct automatic evaluation metrics. This further validates the effectiveness and generalizability of our approach. We observe a significant improvement in writing depth with our method. Additionally, our approach consistently outperforms existing methods in terms of relevance, engagement, and breadth of the generated content.

Ablation study. To further validate the effectiveness of our approach, we implemented an ablation version, where we retained the same search agent setup but removed the recursive planning strategy (denoted as "w/o HRP" in Table [2\)](#page-7-1). This modification required the planner to generate subtasks as a linear workflow all at once rather than in a hierarchical manner. By isolating this variable, we could quantify the performance gains specifically attributable to recursive planning. We observe a significant drop in depth metrics in the ablation version, demonstrating the benefits of HRP. Additionally, removing recursive planning results in a notable decline in novelty and breadth, further highlighting its contribution to the generation quality.

Reasoning model compatibility. We further experimented using the reasoning model DeepSeek-R1 [\(DeepSeek,](#page-9-7) [2024\)](#page-9-7) as the base LLM. Results demonstrate that our approach maintains significant performance advantages. Particularly notable improvements were observed in reasoning depth and breadth metrics. This demonstrates our method's consistent ability to enhance reasoning capabilities. Our analysis included Perplexity's Deep Research[2](#page-8-0) (Feb. 2025), a commercial reasoning model based agent, tested on the same dataset. The results demonstrate that our methodology, when implemented with either Claude or DeepSeek-R1 as the base model, delivers significantly superior performance across all measured metrics compared to this commercial alternative.

#### 6.2.2 Long Reports and Human Evaluation

To assess our framework's ability in generating extended long-form reports (over 10,000 words), we conducted a dedicated human evaluation study, detailed in Appendix [B.](#page-13-0)

Dataset. Existing datasets like WildSeek provide prompts that are too concise and lack necessary details to specify requirements for complex, longform reports. To tackle that, we created a new benchmark dataset, LongReport, specifically designed with 12 complex prompts intended to elicit comprehensive reports. Our topic selection prioritizes time-sensitive subjects that require the model to access current knowledge, with topics systematically categorized based on varying assessment emphases.

Evaluation metrics and baseline. We adopted the four dimension as on WildSeek with one additional dimension *Clarity, Cohesion, and Language* to assess organization and language use. We recruited five volunteer annotators with qualified technical backgrounds to compare reports generated by WriteHERE against a state-of-the-art commercial baseline, Gemini Deep-Research (2.5 Pro)[3](#page-8-1) . Each annotator provided absolute scores on a 1-5 scale for all five dimensions and indicated their overall preference.

Results. The results demonstrate that our method exhibits performance comparable to Gemini, with a slight advantage reflected in a 7:5 vote score across 12 topics, which further validates the capability of our approach for long-form writing.

