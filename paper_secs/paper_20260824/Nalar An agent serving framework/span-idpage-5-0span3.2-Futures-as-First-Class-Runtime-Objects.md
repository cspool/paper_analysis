# <span id="page-5-0"></span>3.2 Futures as First-Class Runtime Objects

NALAR's futures are inspired by prior systems such as Ray [\[32\]](#page-14-7), CIEL [\[33\]](#page-14-8), and Dask [\[34\]](#page-14-9). A future in NALAR represents a long-running, agent-driven computation and encapsulates its readiness, consumers, and workflow position. This *metadata* enables informed scheduling decisions.

NALAR 's futures are designed to be unobtrusive to workflow programmers. In contrast to systems like Ray, where programmers must explicitly manipulate futures via calls such as ray.get() or ray.wait(), NALAR allows most workflows to be written without any direct interaction with future objects. The runtime transparently manages their creation, propagation, and resolution. This not only simplifies programming but also enables developers to run the same unmodified code locally for testing, without needing to emulate distributed future-handling logic. We believe programmer experience is one of NALAR 's key contributions. Developers can build and evaluate their agentic workflows locally without any dependency on NALAR, and only integrate with the framework at runtime. This contrasts sharply with systems like Ray [\[32\]](#page-14-7), CIEL [\[33\]](#page-14-8), and Orleans [\[3\]](#page-13-3), which require developers to interact with the library before writing any code.

Futures API: In certain scenarios, programmers may want to interact with futures, as shown in Line 29 of Figure [4:](#page-4-1) the programmer could check whether multiple tasks have failed without blocking and immediately relaunch them, enabling greater parallelism and fault tolerance. To enable this, NALAR futures provide a simple API, with two methods: (i) future.available (): returns true if the value is ready, false otherwise; (ii) future.value (timeout=t), returns the future output, and blocks upto timeout *t*. [§4.3](#page-7-0) discusses run-time future creation and management.

## 3.3 Custom State Management

Agentic workflows often require maintaining state for longrunning, session-based requests. We analyzed several agentic applications on GitHub [\[6,](#page-13-4) [7,](#page-13-5) [25,](#page-14-10) [35\]](#page-14-11) and observed that developers typically use Python lists and dictionaries for maintaining custom state. Current frameworks force developers to manually manage state, including its lifetime and placement [\[12,](#page-13-1) [13,](#page-13-2) [40\]](#page-15-1), which is challenging because: (1) it is difficult for the programmer to anticipate runtime conditions and (2) it requires rewriting workflows whenever the application needs or logic changes. For efficiency, the serving framework should transparently and dynamically manage state, handling placement, consistency, and life-times without developer intervention. To simplify the management of custom state and give the framework visibility and control over it, NALAR provides *managedList* and *managedDict* abstractions. To utilize these in their workflows, the developers import these in their workflow and use them as standard Python lists and dictionaries. The framework transparently manages placement, consis-

Table 1: NALAR's hint interface

<span id="page-5-3"></span>

| Hint          | Values   | Descriptions                                                                                      |  |
|---------------|----------|---------------------------------------------------------------------------------------------------|--|
| stateful      | Boolean  | True indicates for a session successive calls to<br>the agent will be routed to the same instance |  |
| batchable     | Boolean  | True indicates that module can accept a batch of request                                          |  |
| preemptable   | function | A running request on this agent can be preempted,<br>by calling the given function name           |  |
| max_instances | Integer  | Indicates the max number of instances<br>to initialize                                            |  |
| min_instances | Integer  | Indicates the min number of instances<br>the framework should keep alive                          |  |
| resources     | Dict     | A dictionary of CPU, GPU and Memory to allocate                                                   |  |

tency, and life-cycle. Also, it automatically tracks the *session* associated with the current program instance. We discuss the design details in [§4.3.](#page-7-0)

## <span id="page-5-2"></span>3.4 Runtime *directives*

Agents and tools often have execution properties that the runtime can exploit for efficiency. For example, if an agent supports batching, a common pattern in ML workloads, NALAR can coalesce compatible futures and execute them together, as the throughput of LLM output generation greatly benefits from batching [\[2,](#page-13-6)[24\]](#page-14-12). Incorporating such agent-specific characteristics enables more informed scheduling and placement decisions than futures alone would allow.

To this end, NALAR provides a *directive* interface. For example, in Line 7 of Figure [4,](#page-4-1) the programmer indicates that the developer agent supports batching. Table [1](#page-5-3) lists the supported agent-level directives used by the runtime to guide execution. Most directives are straightforward, but we highlight the *stateful* directive. For agents marked stateful, NALAR guarantees that all requests to the agent are associated with a single user request and a single session are scheduled in order and routed to the same agent instance ensuring consistent processing.

