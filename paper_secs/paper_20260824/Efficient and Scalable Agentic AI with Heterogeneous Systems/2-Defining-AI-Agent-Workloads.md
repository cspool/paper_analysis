# 2 Defining AI Agent Workloads

AI agents can broadly be defined as computational entities that perceive their environment, process information, and act toward a specified goal. Unlike traditional software governed by static control flow, agents operate based on data, models, and dynamic policies. Their design spans a wide spectrum, ranging from simple, rule-based systems to complex, autonomous frameworks capable of multi-step planning, memory management, and external tool integration.

#### 2.1 Autonomy and Complexity

AI agents can differ significantly in their level of autonomy and internal complexity. At one end of the spectrum are reflex based agents that follow simple, condition-based logic such as spam filters or thermostats that map input to output using predefined rules [\[15\]](#page-22-14). More advanced agents exhibit autonomy by reasoning over multiple steps and selecting actions through a closed loop of perception, decision-making, and tool use. For example, AI agents can be leveraged to answer open-ended questions with real data by dynamically querying external sources, such as Wikipedia or search APIs [\[4\]](#page-22-3).

These workflows demand richer architectural structure to effectively coordinate control flow and memory over time. As agent behaviors become more complex, systems have evolved from single agent pipelines to multi-agent architectures with distinct hierarchies. Figure [1](#page-3-0) presents a taxonomy of common agentic structures, illustrating how control can flow through single agents, coordinated peers, hierarchical layers, or fully custom graphs. For example, in a peer-to-peer agent network, multiple agents operate concurrently on different sub-tasks and collaborate by exchanging information to achieve a shared goal. In contrast, a hierarchical architecture introduces structured layers of control, where higher level agents handle planning and decision making, while lower level agents focus on specialized execution tasks.

#### 2.2 Interactivity

Another dimension of variation is interactivity, which refers to how agents engage with their environment or users throughout task execution. Interactive agents operate in a feedback loop, adjusting behavior in response to real-time inputs. These agents often expose APIs, hold state across turns, and issue clarifying queries to disambiguate user intent. This design is well-suited for customer service bots, coding copilots, or multi-step reasoning tasks. In contrast, noninteractive agents execute in a single pass, producing outputs without runtime adaptation. Such agents are commonly used for document summarization, batch inference, or fixed automation pipelines.

