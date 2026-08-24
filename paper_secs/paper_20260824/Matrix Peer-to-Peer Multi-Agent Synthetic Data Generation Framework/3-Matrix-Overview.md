# 3 Matrix Overview

This section provides an overview of the Matrix framework, describing its system architecture and the core algorithm that enables scalable, asynchronous multi-agent synthetic data generation.

## 3.1 System Architecture

Figure [1](#page-3-0) illustrates the architecture of Matrix, a distributed runtime for multi-agent synthetic data generation. The system is designed to be modular and configurable, separating lightweight peer-to-peer agents from scalable backend services (e.g., LLM inference and containerized tool execution) so that components can be adapted and scaled independently across different workflows.

Cluster Management. The framework is deployed atop SLURM [Yoo et al.](#page-18-11) [\(2003\)](#page-18-11), a widely adopted distributed computing environment, with a Ray [Moritz et al.](#page-17-2) [\(2018\)](#page-17-2) cluster serving as the execution substrate. Ray Serve provides high-throughput LLM inference services, backed by vLLM [Kwon et al.](#page-17-6) [\(2023\)](#page-17-6), SGLang [Zheng et al.](#page-18-12) [\(2024\)](#page-18-12), and FastGen [FAIR](#page-15-9) [\(2025\)](#page-15-9). Containerized execution is supported through Apptainer [Kurtzer et al.](#page-17-7) [\(2017\)](#page-17-7), enabling stateful environments to be launched on demand. Each agent is implemented as a Ray Actor, allowing scalable parallelization and fine-grained resource placement across worker nodes.

<span id="page-3-0"></span>> **[图片提取文字 (无描述)]:**
> Configuration Agent Environments Agents prompts Orchestrator Data Loader Multi-Agents Data input Yes output Metrics success rate rewards Sink Continue Orchestrator Orchestrator sequential hierarchical performance logging request response Trajectory Metrics Data Store Monitoring Cluster Management LlaMa bash Seed data Agent Apptainer Experiments vLLM Actor Generated Container (system/gpu Data metrics, queue RAY length, query latency) load Assets SLURM Data
![](_page_3_Figure_0.jpeg)

Figure 1 Matrix Agentic Data Generation Architecture.

Configuration. System configurability is managed through Hydra [Yadan](#page-18-13) [\(2019\)](#page-18-13), which specifies agent roles, input–output schemas, generation metrics, and resource requirements (e.g., LLM engine selection). The configuration also defines the orchestrator responsible for control and data flow management. Users can define the use case specific orchestrator class in Hydra, then hand-write control flow as imperative if/else logic in the implemetation. Matrix also supports a graph-based specification. For users familiar with LangGraph [LangChain](#page-17-0) [\(2025\)](#page-17-0), we provide a LangGraphOrchestrator that performs state transitions based on a user-supplied LangGraph: users define nodes (agent roles) and edges (transition rules) in the standard LangGraph API, and the orchestrator updates its state by executing the graph's routing decision at each step. This makes non-sequential workflows easier to express and to visualize while preserving Matrix's decentralized, message-driven execution model.

Agents Environments. In the peer-to-peer generation process, each input datum is encapsulated into an orchestrator instance and passed to the initial agent. The agent processes the instance, updates the orchestrator state, and forwards control to the next designated agent. This process continues iteratively until completion. A detailed algorithmic description is provided in Section [3.2.](#page-3-1) We will use Matrix to build different agents environments in the experiments section.

Monitoring. Logging and observability are critical for debugging and performance analysis. Matrix integrates with Grafana [Labs](#page-17-8) [\(2025\)](#page-17-8) for real-time monitoring. In addition to standard performance metrics, it provides custom indicators such as distributed queue length and the number of pending asynchronous tasks. These metrics help identify throughput bottlenecks and evaluate overall system health.

## <span id="page-3-1"></span>3.2 Data Generation Algorithm

Algorithm [1](#page-4-0) illustrates the core workflow of Matrix's peer-to-peer agentic generation runtime. The system begins by reading the Hydra configuration cfg, which specifies all agent roles and their resource requirements (e.g., CPU, GPU, and memory). As shown in Lines 19–24, the function create\_team() instantiates a distributed team of Ray actors for each agent role, allowing heterogeneous resource allocation across agent types.

#### <span id="page-4-0"></span>Algorithm 1: Matrix P2P agentic generation pseudocode.

```
1 @ray.remote
2 class AgentActor: # agent base class with an event loop to process orchestration messages
3 async def _event_loop(self, team):
4 while True:
5 orchestrator = await self.queue.get()
6 result = self.process(orchestrator)
7 orchestrator.update(result) # update conversation history and determine next agent
8 next_agent = orchestrator.current_agent()
9 random.choice(team[next_agent]).send(orchestrator) # send updated orchestrator to next agent
11 class SequentialOrchestrator: # a typical orchestrator with a configurable order of execution
12 def update(self, result):
13 self.history.append(result)
14 self.index = (self.index + 1) % len(self.order) # take the next agent in the given order with loop around
16 def current_agent(self):
17 return "_sink" if self.is_done else self.order[self.index] # loop around until is_done flag is set
19 def create_team(cfg): # create a team of agents based on the configuration
20 return {
21 role: [ray_create_actor(role, role_cfg) # each agent instance become a Ray actor
22 for _ in range(role_cfg.num_instances)]
23 for role, role_cfg in cfg.items()
24 }
25 # main processing
26 team = create_team(cfg.agents)
27 for item in dataset: # process each dataum concurrently up to max_concurrency asyncio tasks
28 orchestrator = Orchestrator(item)
29 first = random.choice(team[orchestrator.current_agent])
30 first.send(orchestrator) # send the orchestrator to the start agent
```

Agent EventLoop. The main generation loop (Lines 26–30) iterates over input items in the dataset. For each item, an Orchestrator object is created to manage task-specific state and control flow. The orchestrator is initially dispatched to the first agent in the sequence, sampled randomly from the corresponding role group. Each agent runs as a persistent event-driven process (Lines 3–9) implemented by the AgentActor class. Within its asynchronous \_event\_loop, the agent dequeues orchestrators from its inbox, applies role-specific logic through process(), updates task state, and forwards it to the next designated agent.

Orchestration. An example orchestrator SequentialOrchestrator is in Lines 11–17. It maintains a structured history of intermediate results and a configurable order, which determines the sequence of participating agents. After each interaction, update() advances the internal index to the next agent. The process continues cyclically until the orchestrator's is\_done flag is set, at which point it is routed to a special terminal agent, \_sink, for result persistence and metric aggregation.

Concurrency Control. Advanced runtime features (not shown for brevity) include task-level concurrency control through a max\_concurrency parameter and semaphore-based scheduling. The semaphore is decremented when an orchestrator is dispatched to the first agent and incremented upon completion by the \_sink agent. This mechanism limits the number of active orchestrators, ensuring controlled resource utilization and stability during large-scale distributed execution. Note we rely on Ray to avoid race conditions in distributed RPC calls.

