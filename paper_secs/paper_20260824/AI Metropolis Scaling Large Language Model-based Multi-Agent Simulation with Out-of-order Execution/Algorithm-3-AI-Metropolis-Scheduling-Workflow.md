# Algorithm 3 AI Metropolis Scheduling Workflow

```
1: Input: target step, agents, world
```

- 2: Initialize: base step ← 0, ready agents ← agents
- 3: worker pool ← InitProcessPool(process routine)
- 4: ready queue ← PriorityQueue()
- 5: ack queue ← PriorityQueue()
- 6: dependency graph ← Graph(agents)

#### {Controller}

- 7: while base step < target step do
- 8: ready clusters ← *geo clustering*(ready agents)
- 9: ready queue.put(ready clusters)
- 10: ack cluster ← ack queue.get()
- 11: ready agents ← update agents(agents, ack cluster)
- 12: base step ← update base step(ack cluster)
- 13: end while

#### {Worker}

- 14: while base step < target step do
- 15: cluster ← ready queue.get()
- 16: actions ← [ ]
- 17: for all agent in cluster do
- 18: actions.append(agent.*proceed*(world))
- 19: end for
- 20: world.*resolve conflict and commit*(actions)
- 21: dependency graph.*update*(cluster)
- 22: end while

#### 3.1 Overview

Below we explain the essential terms in AI Metropolis:

• Blocked: An agent A becomes *blocked* if it has to wait for another agent B to finish the current step before it can proceed, ensuring temporal causality and simulation correctness. This is formally defined in §3.2.

- Coupled: Agents A and B, become *coupled* if they are sufficiently close, interact with each other, and thus must proceed together. This is formally defined in §3.2.
- Cluster: A cluster is a group of *coupled* agents at the same step. Each agent can independently issue requests to LLMs within its *proceed* function. However, the entire group needs to synchronize at the end of the step to resolve potential conflicts and avoid dependency violations as described in §3.2.
- Worker: A worker is a process that handles one cluster, to proceed one step at a time. Within the worker process, each agent in the cluster operates in its own thread to communicate with the LLM serving engine and process its tasks. Workers are independent processes without synchronization between them, and the number of workers can be adjusted based on available CPU resources.
- Controller: The controller is the main process of the simulation engine. After initializing the world, it periodically communicates with workers through two queues: it prepares tasks for workers via the *ready queue* and confirms the completion of clusters from workers through the *ack queue*.

During a simulation, workers continuously pull clusters from the *ready queue*. After finishing the step for a cluster, a worker updates the dependency graph stored in a database for all agents in the cluster and then places the completed cluster in the *ack queue* as a completion confirmation. Simultaneously, the controller continuously pulls notifications from the *ack queue*. Each time it processes a confirmation, it queries the dependency graph to filter out the agents that are not *blocked*, creating new ready clusters out of the ready agents and placing them into the *ready queue* promptly. This workflow allows agents to advance steps ahead of others as long as dependency permits. Notably, both the *ready queue* and *ack queue* are priority queues that automatically sort tasks based on their associated steps.

We discuss the the dependency tracking mechanism in §3.2, the spatiotemporal dependency graph realizing the mechanism in [§3.3,](#page-4-0) agent clustering in [§3.4,](#page-4-0) priority scheduling in [§3.5,](#page-5-0) and key implementation details in [§3.6.](#page-5-0)

#### 3.2 Dependency Tracking

As described in [§2.2](#page-2-0) and shown in Figure [2,](#page-2-0) temporal causality introduces the dependence of tasks in a simulation. Using the language of computer systems, temporal causality creates an order of a set of reads and writes on a shared memory. At the beginning of each step, agents read different parts

<span id="page-4-0"></span>of the environment and at the end, they commit writes to different parts. Therefore, the tasks of an agent across different steps must be serialized, as it reads what it wrote in the last step. The dependency between tasks of different agents can then also be formulated as a read-after-write data dependency. For two agents A and B, if A is about to observe a part of the world at step StepA, and B is about to write to that part of the world at step StepB. if Step<sup>B</sup> is smaller than StepA, meaning the write is designed to happen before the read, then A must wait for B to complete step Step<sup>B</sup> before start the tasks in step StepA.

In simulations, each agent typically perceives only a portion of the world, defined as the surrounding area within a specified size—denoted as radius p. We also assume a maximum speed limit, denoted as max vel, governing both agents' movement and information propagation within each step. For instance, in GenAgent, an agent perceives an area within a radius of 4 grid units and can modify the status of an adjacent grid by interacting with an object or agent on it or moving to it. This defines the region the agent can read from and write to. Consequently, to maintain read-afterwrite data dependency, the following condition must hold throughout the simulation:

$$\forall$$
 agents  $A,B,$  and their current steps  $Step_A$  and  $Step_B,$  if  $Step_A \neq Step_B,$  then  $dist(A,B) > radius\_p + (|Step_A - Step_B| - 1) \times max\_vel$ 

where dist(A, B) denotes the distance between A and B. This ensures that a read in a later step will never touch the region that a write in a prior step might influence. Intuitively, this means that agents never perceive other agents who exist at different times. To enforce this condition, we can derive the following rules to determine the relationships between the tasks of agents. The complete derivation can be found in Appendix [A](#page-12-0) and the following are the rules AI Metropolis uses: for any two agents A and B and their tasks in steps Step<sup>A</sup> and StepB,

- We define that A and B are *coupled* if Step<sup>A</sup> = Step<sup>B</sup> and dist(A, B) ≤ radius p+max vel, which means they must be grouped into the same cluster and proceed to the next step together.
- We define that A is *blocked* by B if dist(A, B) ≤ (Step<sup>A</sup> − Step<sup>B</sup> + 1) × max vel + radius p, which means A cannot start its tasks in step Step<sup>A</sup> until B has proceeded to the next step.
- A cluster can freely advance to the next step if none of its agents are blocked by any other agent.

Note that this set of rules are conservative, meaning the read in the later step will wait for all potential writes to a certain

> **[图片提取文字 (无描述)]:**
> F@z A@x C@y B@x D@y E@y
![](_page_4_Figure_9.jpeg)

Figure 3. An example of a spatiotemporal dependency graph. Each node, such as A@x, represents an agent (A) at a specific time step (x). Single arrows indicate dependencies, while double arrows represent coupled relationships between agents. Purple boxes denote clusters of agents, where green nodes indicate agents that are ready for execution and orange nodes represent blocked agents.

region to finish, even if the write does not occur eventually. While this may still preserve some false dependencies, we show in [§4](#page-5-0) that it achieves performance close to the optimal. Additionally, this design does not require a data race detector and correction mechanism, making it more scalable and easier to implement.

#### 3.3 Spatiotemporal Dependency Graph

The rules defined in [§3.2](#page-3-0) define a specialized dependency graph that tracks dependencies between agents. Each node in the graph represents an agent, containing its temporal (time step) and spatial (coordinates on the map) information. An edge A → B indicates that agent B is currently *blocked* by agent A, while A ↔ B signifies that agents A and B are coupled as shown in Figure 3.

Our system maintains this graph in an in-memory database, and whenever a worker advances a cluster to the next step, it applies the rules to re-examine the relationships of each agent in the cluster with any other relevant agents. Any changes are then written to the database. The examination and graph updates are encapsulated within a transaction to ensure data consistency. Afterward, the worker process places a completion confirmation into the *ack queue*.

The dependency graph will be utilized by the controller process for efficiently identifying the agents that are not *blocked*, allowing it to release maximum parallelism.

#### 3.4 Agent Clustering

When agents are close enough to each other, they perceive each other's actions committed in the last step. In other words, they collectively read what they wrote in the last step. They might also encounter write conflicts that must be resolved by developer-specified rules; for example, two agents might both want to use the bathroom, but only one can step in. These potential interactions couple them into a cluster that must proceed together. Whenever there are new ready agents who are not blocked, the controller process <span id="page-5-0"></span>runs *geo clustering* to group coupled agents into clusters based on the rule described in [§3.2.](#page-3-0) If none of the members of a cluster are blocked, the cluster is considered ready and will be placed into the *ready queue*. Using clusters as the minimal synchronized units, as opposed to synchronizing all agents as described in Algorithm [1,](#page-1-0) effectively reduces false dependencies and scheduling overhead.

#### 3.5 Priority Scheduling

As motivated in [§2.2,](#page-2-0) allowing agents to process tasks associated with different time steps simultaneously creates requests of varying priorities. We found that the time step associated with a request serves as a good measure of its priority. A write operation in a prior step can block many reads in subsequent steps; intuitively, the smaller the time step, the more future actions it can potentially block. To enhance parallelism, we maintain both the *ready queue* and *ack queue* as priority queues, prioritizing the execution of tasks from earlier steps. No preemption during LLM inference is applied as that might cause extra overhead [\(Sheng](#page-11-0) [et al.,](#page-11-0) [2023b\)](#page-11-0). We demonstrate the effectiveness of this priority scheduling in [§4.4.](#page-8-0)

#### 3.6 System Implementation

In addition to the design choices around dependency management, clustering, and priority scheduling that enable AI Metropolis to expose more parallelism from the simulation, it is also worth highlighting some of the design choices that make AI Metropolis scalable:

Proper Mapping of Parallelism. Choosing the right programming abstraction for different tasks is critical to scalability. AI Metropolis employs threads for agents, as the need for synchronization within clusters requires low-overhead communication. Processes are used for the controller and workers to avoid the Python GIL and to enable scaling beyond a single machine.

#### Light and Fast Critical Path on the Controller Process.

All tasks on the critical path are implemented in C++ to reduce overhead and bypass the limitations of the Python GIL. Furthermore, heavy lifting, including complex agent processing logic and dependency graph updates, is offloaded to concurrent workers. This approach lightens the critical path for the controller process, minimizing workers' waiting time for task allocation.

Scalable I/O. Except for *ready queue* and *ack queue*, all inter-process synchronization are handled through an inmemory [\(Redis\)](#page-11-0) database. This database also handles transactional updates for all simulation states and stores instrumentation data, supporting automatic scalability beyond a single node.

Decoupling Simulation Engine from LLM Serving Engine. In AI Metropolis, only workers communicate with the LLM serving engine through a thin shim layer, providing easy observability and scalability.

We implemented the core of AI Metropolis in about 1k lines of C++ and 2k lines of Python code. An additional 3k lines of Python code (excluding assets and prompts) were written to port the GenAgent simulation using our interfaces. This is about 50% of the code compared to the original implementation, achieving up to an order of magnitude speed-up and promising scalability, as demonstrated in §4.

