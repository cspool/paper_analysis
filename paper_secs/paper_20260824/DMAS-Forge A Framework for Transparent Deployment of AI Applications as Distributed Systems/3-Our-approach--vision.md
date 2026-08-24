# 3 Our approach & vision

We envision a "write-once, deploy-everywhere" paradigm, where developers write multi-agent applications once and our framework flexibly ports them to different execution environments, compliant with communication protocols. This vision is summarized in Fig. [1.](#page-1-0)

#### 3.1 DMAS-Forge compiler

We propose a compiler-based approach that enables clean separation of core agent logic from the underlying communication protocols and deployment infrastructure. Our key insight is that the core computation model of an AI application is completely orthogonal to how and where the computation is performed. This clean separation allows application developers to provide their structural agentic workflow independently from the deployment choices, and allows developers to plug-and-play any deployment choice in the future.

Our compiler expects two inputs, highlighted in Fig. [2.](#page-2-0)

The structural agentic workflow (ln[.13\)](#page-2-1): a graph-like computation workflow that includes the various agent implementations with their corresponding tools and the connections between the different agent computations. We follow the computational graph model of an existing AI programming framework, LangGraph, that represents the nodes in the graphs as agents and the edges between the nodes as agent communications.

The deployment specification (ln[.8\)](#page-2-2): this is the deployment information for the computation graph including how each agent/tool will run (i.e., process, container, serverless), how different agents/tools will connect and communicate (i.e., choice of protocol), runtime constraints (e.g., hardware type, number of replicas, access policies), and the underlying LLM for each agent.

The compiler automatically generates the necessary glue code for deploying the application as described by the provided workflow and the deployment specification. It automatically bakes in the code to ensure that connected agents/tools comply with the user-specified communication protocol. Additionally, depending on the deployment targets, it automatically generates the necessary configuration files, including Dockerfiles and Kubernetes configurations to enforce access policies, ensure the binding to the desired LLM type, etc.

Our compiler-based approach is inspired by previous approaches to flexibly configure microservices [\[1\]](#page-3-8), and flexibly support distributed deployment of computation graphs in different environments [\[21\]](#page-4-13).

## 4 Prototype

We showcase an initial prototype[1](#page-2-3) in Go language, named DMAS-Forge. To build DMAS-Forge, we extend the Blueprint microservices compiler [\[1\]](#page-3-8) to support AI applications. We chose Blueprint as it is compatible with our computational model and it already provides a large array of deployment choices and infrastructure components that can be leveraged.

Programming interfaces. [Table 1](#page-2-4) shows extensions we add to Blueprint to support the new requirements. First, we extend Blueprint's workflow API to allow users to easily implement their applications as structured workflows, similar to LangGraph. Second, we offer a new wiring API, through which developers can input the deployment specifications to DMAS-Forge.

In our prototype, this is achieved by implementing several new Blueprint plugins: (1) An Agent plugin, to transparently create agents and connect them with any OpenAI-compatible model. (2) A vLLM plugin, to automate model deployment in vLLM [\[22\]](#page-4-14). (3) A kagent plugin, for supporting distributed

```
1 import DMASForge / http
2 import DMASForge / linuxcontainer
3 import DMASForge / vllm
4 import DMASForge / openai
6 spec = DMASForge.NewSpec()
8 // Deployment specification
9 def DeployAgent ( agent ) :
10 http . Deploy ( spec , agent )
11 linuxcontainer . Deploy (spec, agent )
13 // Structural agentic workflow
14 model = vllm . Model ("gpt -4o")
15 weather = openai . Agent ( model , prompt =" ...")
16 news = openai . Agent ( model , prompt ="... ")
17 weather . Connect ( news )
18
19 // Deployment
20 DeployAgent ( news )
21 DeployAgent ( weather )
```

<span id="page-2-1"></span>Figure 2: Two-agent application in DMAS-Forge.

<span id="page-2-4"></span>

| Workflow Type Extensions |                                        |
|--------------------------|----------------------------------------|
| Agent                    | Specifies an Agent                     |
| .Connect(agent)          | Connects two agents                    |
| .AddTool(tool)           | Adds a tool                            |
| Model                    | Specifies a Model                      |
| Tool                     | Specifies a Tool that an Agent can use |
| Wiring API               |                                        |
| NewSpec()                | Creates a new DMAS-Forge spec          |
| openai.Agent()           | New instance of an openai Agent        |
| vllm.Model(name)         | Launches a new model instance          |
| NewTool()                | New instance of a tool                 |
| a2a.Deploy()             | Deploy Agent with A2A                  |

Table 1: DMAS-Forge API overview.

mcp.Deploy() Deploy Tool using MCP kagent.Deploy() Deploy via kagent

deployments with kagent [\[20\]](#page-4-6), an emerging Kubernetesnative framework for AI agents that provides Custom Resource Definitions (CRDs) for agents, tools and models.

Lastly, we leverage the RPC-over-HTTP Blueprint plugin as an example of inter-agent communication protocol in our prototype. Fig. [2](#page-2-0) shows a complete example of the use of these plugins to deploy two agents as Linux containers.

<span id="page-2-3"></span><sup>1</sup>Available at [https://github.com/vaastav/DMAS\\_forge](https://github.com/vaastav/DMAS_forge)

<span id="page-3-9"></span>> **[图片提取文字 (无描述)]:**
> Dev d .. p Profiling & **DMASForge** Deployment Cost model MAS specs Platform-specific artifacts optimization loop
![](_page_3_Figure_1.jpeg)

Figure 3: DMAS-FORGE as enabler of a closed-loop optimization pipeline for MAS deployment.

#### 5 Discussion and future work

DMAS-FORGE is a compiler-based framework that transforms multi-agent applications into a distributed deployment with the effort of one click. It targets diverse runtime environments and aims at generating the necessary glue code for each of them. We presented an initial prototype that supports Linux containers. We plan to extend to other runtime environment, as well as showcasing its benefits for (at least) the following areas.

**Optimization pipelines**. A key area for improvement would be to streamline the creation of closed-loop optimization pipelines. Applications can be written, deployed, and profiled for communication patterns and resource usage, then automatically restructured for better performance. For example, LangGraph's communication structures (such as sequential, parallel, or conditional flows) can be optimized at runtime without requiring manual redeployment. This pipeline is illustrated in Fig. 3.

This process relies on measuring and modeling communication patterns, performance, and cost-efficiency across different infrastructures and deployment environments. It also involves decisions about whether agents should be colocated or separated into different containers. A key question is whether alternative communication patterns or protocols might better suit the application, given its deployment, latency, and performance constraints.

With these improvements, users no longer need to manually specify the number of replicas or communication protocols. Instead, the framework can make those choices automatically, based on the available infrastructure and performance requirements.

Automatic security boundaries. Another area of future work is to explore means of re-adjusting security policies and determining the least amount of authorization for a DMAS container (or other deployed instance), based on the agents and tools it is running. This is particularly useful when agents are co-located or disaggregated across containers by the optimization pipeline. Automatically determining

the security boundaries becomes necessary to minimize the attack surface of each deployed container.

Engineering challenges. Future work should align DMAS-FORGE with the complete features of current agentic frameworks and extend them. A key open question is whether to build comprehensive agentic capabilities directly into DMAS-FORGE or to serve as an abstraction layer over existing frameworks. The latter approach would require techniques (e.g., monkey patching) to redirect framework-specific communication primitives to protocol-compliant distributed channels and avoid substantial re-engineering effort.

#### References

- <span id="page-3-8"></span> Vaastav Anand, Deepak Garg, Antoine Kaufmann, and Jonathan Mace.
   Blueprint: A Toolchain for Highly-Reconfigurable Microservice Applications. In SOSP. Association for Computing Machinery.
- <span id="page-3-3"></span>[2] Anthropic. 2024. Model Context Protocol (MCP). https://docs. anthropic.com/en/docs/mcp.
- <span id="page-3-4"></span>[3] Muhammad Bilal, Marco Canini, Rodrigo Fonseca, and Rodrigo Rodrigues. 2023. With Great Freedom Comes Great Opportunity: Rethinking Resource Allocation for Serverless Functions. In *Proceedings* of EuroSys'23.
- <span id="page-3-1"></span>[4] Gohar Irfan Chaudhry, Esha Choukse, Íñigo Goiri, Rodrigo Fonseca, Adam Belay, and Ricardo Bianchini. 2025. Towards Resource-Efficient Compound AI Systems. In *HotOS*. Association for Computing Machinery.
- <span id="page-3-6"></span>[5] GitHub Community. 2025. Issue #227. https://github.com/i-ambee/acp/discussions/227.
- [6] Reddit community. 2025. Reddit thread: running each agent node in LangGraph workflow in its own docker container. https://www.reddit.com/r/LangChain/comments/1i2848r/running\_each\_agent\_node\_in\_langgraph\_workflow\_in/.
- <span id="page-3-7"></span> [7] StackOverflow community. 2024. Deploying Langgraph nodes in separate containers. https://stackoverflow.com/questions/79677336/ deploying-langgraph-nodes-in-separate-containers.
- <span id="page-3-2"></span>[8] Davis Cornelia. 2025. Production-ready agents with the OpenAI Agents SDK + Temporal. https://temporal.io/blog/announcing-openaiagents-sdk-integration.
- <span id="page-3-5"></span>[9] Microsoft Corporation. 2024. AutoGen. https://microsoft.github.io/autogen/stable//index.html.
- <span id="page-3-0"></span>[10] Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. 2023. Improving Factuality and Reasoning in Language Models through Multiagent Debate. arXiv:2305.14325

- <span id="page-4-12"></span>[11] Inc. FoundryLabs. 2025. E2B: AI Sandboxes for Automation Agents. [https://e2b.dev/docs.](https://e2b.dev/docs)
- <span id="page-4-9"></span>[12] CrewAI Inc. 2024. CrewAI. [https://www.crewai.com.](https://www.crewai.com)
- <span id="page-4-8"></span>[13] LangChain Inc. 2024. LangGraph. [https://langchain-ai.github.io/](https://langchain-ai.github.io/langgraph/) [langgraph/.](https://langchain-ai.github.io/langgraph/)
- <span id="page-4-0"></span>[14] LangChain Inc. 2025. LangGraph: Workflows and Agents. [https:](https://langchain-ai.github.io/langgraph/tutorials/workflows/#set-up) [//langchain-ai.github.io/langgraph/tutorials/workflows/#set-up.](https://langchain-ai.github.io/langgraph/tutorials/workflows/#set-up)
- <span id="page-4-2"></span>[15] Zhenkun Li, Lingyao Li, Shuhang Lin, and Yongfeng Zhang. 2025. Know the Ropes: A Heuristic Strategy for LLM-based Multi-Agent System Design. arXiv[:2505.16979](https://arxiv.org/abs/2505.16979) [cs.AI] [https://arxiv.org/abs/2505.](https://arxiv.org/abs/2505.16979) [16979](https://arxiv.org/abs/2505.16979)
- <span id="page-4-10"></span>[16] LlamaIndex. 2024. LlamaIndex. [https://www.llamaindex.ai.](https://www.llamaindex.ai)
- <span id="page-4-7"></span>[17] Google LLC. 2025. Agent2Agent (A2A) Protocol. [https://github.com/](https://github.com/a2aproject/A2A) [a2aproject/A2A.](https://github.com/a2aproject/A2A)
- <span id="page-4-5"></span>[18] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Sean Welleck, Bodhisattwa Prasad Majumder, Shashank Gupta, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. In NeurIPS. Curran Associates Inc.

- <span id="page-4-11"></span>[19] Agno maintainers. 2025. Agno Teams. [https://docs.agno.com/concepts/](https://docs.agno.com/concepts/teams/introduction) [teams/introduction.](https://docs.agno.com/concepts/teams/introduction)
- <span id="page-4-6"></span>[20] Solo.io. 2025. kagent: Cloud Native Agentic AI Framework. [https:](https://kagent.dev/) [//kagent.dev/.](https://kagent.dev/)
- <span id="page-4-13"></span>[21] TensorFlow. 2025. TensorFlow. [https://www.tensorflow.org.](https://www.tensorflow.org)
- <span id="page-4-14"></span>[22] vLLM (Berkley). 2025. vLLM: Easy, fast, and cheap LLM serving for everyone. [https://docs.vllm.ai.](https://docs.vllm.ai)
- <span id="page-4-4"></span>[23] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Selfconsistency improves chain of thought reasoning in language models. In ICLR. OpenReview.net.
- <span id="page-4-3"></span>[24] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V. Le, and Denny Zhou. 2022. Chainof-thought prompting elicits reasoning in large language models. In NeurIPS. Curran Associates Inc.
- <span id="page-4-1"></span>[25] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. arXiv[:2210.03629](https://arxiv.org/abs/2210.03629)