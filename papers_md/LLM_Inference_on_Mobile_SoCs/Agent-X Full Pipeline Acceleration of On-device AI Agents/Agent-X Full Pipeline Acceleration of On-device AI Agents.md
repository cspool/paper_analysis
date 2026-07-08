## Agent-X: Full Pipeline Acceleration of On-device AI Agents

Jinha Chung KAIST Republic of Korea, Daejeon jinha.chung@kaist.ac.kr

Jiin Kim KAIST Republic of Korea, Daejeon jiin.kim@kaist.ac.kr

## Abstract

LLM-based agents deliver state-of-the-art performance across tasks but incur high end-to-end latency on edge devices. We introduce Agent-X, a software-only, accuracy-preserving framework that accelerates both the prefill and decode stages of on-device agent workloads. Agent-X's two key components rewrite prompts to leverage prefix caching tailored to agent-specific input-token patterns and enable LLM-free speculative decoding for fast token generation with minimal overhead. On representative agentic workloads, Agent-X achieves a 1.61× end-to-end speedup in real systems with no accuracy loss and can be seamlessly integrated into existing on-device AI agents. To the best of our knowledge, ours is the first to systematically characterize and eliminate latency bottlenecks in on-device agents.

## 1 Introduction

The "ChatGPT effect" has taken the world by storm, and Large Language Models (LLMs) are now embedded in various applications that drive our daily lives. LLM-based AI agents elevate the applicability of LLMs through "tool calling." Equipped with external tools, an LLM can interact with its environment and autonomously execute tasks from start to finish, without further user intervention. As shown in Figure [1,](#page-0-0) given a user query ( 1 ), the AI agent selects the appropriate tools to handle the request ( 2 ). The agent then interacts with the environment (e.g., "Contacts", "Calendar", and "Email") by calling the selected tool ( 3 ) and reflecting on prior tool output(s) to select the next action ( 4 ). In academia, a rich body of prior work [\[36,](#page-12-0) [65,](#page-13-0) [78,](#page-13-1) [85\]](#page-13-2) has improved agent task accuracy, while the industry has already started deploying LLM-based agents [\[6,](#page-11-0) [27,](#page-11-1) [44,](#page-12-1) [52,](#page-12-2) [53\]](#page-12-3), boosting user productivity and convenience.

On-device AI agents, which run entirely on a user's local device, provide two unique benefits over cloud-based agents: availability and privacy. On-device agents are always available to the user for immediate use, regardless of the user's situation (lack of internet access) or the cloud provider's situation (server outage). Furthermore, growing concerns over data misuse call for measures to guarantee privacy in using LLMs, making on-device agents an attractive solution.

Building on this trend, various hardware and software have been introduced to ease the development and deployment of LLMs at the edge [\[4,](#page-11-2) [12,](#page-11-3) [14,](#page-11-4) [47,](#page-12-4) [57,](#page-12-5) [58\]](#page-12-6). Despite these advances, on-device AI agents still suffer from suboptimal latency, even for simple tasks,

Byeongjun Shin KAIST Republic of Korea, Daejeon byeongjun.shin@kaist.ac.kr

Minsoo Rhu KAIST Republic of Korea, Daejeon mrhu@kaist.ac.kr

<span id="page-0-0"></span>![](_page_0_Figure_12.jpeg)

Figure 1: Overview of an agentic system.

due to the resource-constrained nature of edge computing. Unlike cloud-based LLMs whose primary performance bottleneck lies in the decode stage, this paper makes the key observation that ondevice agents spend a significant amount of time in both the prefill and decode stages. This key insight underscores the need for fullsystem acceleration techniques that address all key components of the agentic system pipeline on edge hardware.

To this end, we propose Agent-X, a purely software end-to-end acceleration scheme for on-device agents that does not degrade accuracy. To the best of our knowledge, this is the first work to provide a detailed system-level characterization of on-device AI agents. Building on this analysis, we introduce a full-pipeline acceleration solution that exploits both the algorithmic traits of agents and the hardware characteristics of edge environments, as detailed below.

On-device agents analysis. We analyze the execution of LLMbased agents and identify two LLM instances as primary bottlenecks. Our analysis shows that, unlike conventional server-based, conversation-oriented applications where latency is dominated by decoding, both the prefill and decode stages contribute significantly to end-to-end latency in on-device environments due to the agentic workflow and hardware constraints. We further characterize these stages at the token level and observe two key properties. First, during prefill, the prompt structure limits the applicability of optimizations such as prefix caching. Second, during decode, the output is largely grounded in few-shot examples and does not fully exploit the LLM's reasoning capability.

Accuracy-preserving acceleration algorithm. Building on our characterization, we propose Agent-X, which combines Prompt-Weaver and ExSpec to accelerate the prefill and decode stages of on-device agentic LLMs, respectively. PromptWeaver dynamically reconstructs the input prompt to enable efficient prefix caching,

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

Figure 2: Structure of plan-out agents. The whole pipeline consists of two LLMs (Planner and Arbiter) and a series of tool calls (carried out by Execution unit) to fully serve the user query (blue).

reducing online computation and substantially accelerating the prefill stage. We also note that, while speculative decoding [16, 39] is widely used to speed up LLM decoding, its reliance on additional LLMs hinders deployment on edge devices. To address this, ExSpec introduces a lightweight, prompt-aware draft model that enables efficient speculative decoding at the edge.

Real system integration. We implement Agent-X with Apple's MLX-LM [10] and MLX-engine [43], and integrate it with TinyAgent [19] for a full system acceleration of on-device agents. Our evaluation shows that PromptWeaver and ExSpec speed up the prefill and decode stages of on-device agentic LLMs by 1.97× and 1.73×, respectively, achieving an average end-to-end task latency improvement of 1.61×.

Overall, Agent-X accelerates on-device agents with minimal resource overhead. Its lightweight, accuracy-preserving, and purely software design allows direct integration into existing on-device agentic workflows, delivering immediate speedups and enabling fast, private agents at the edge.

## 2 Background

#### 2.1 LLM-based Agents

**Agentic workflows.** LLM-based agents enhance LLMs' capabilities by interacting with the external environment. The means of interaction are referred to as *tools*, which are APIs available within the system, ranging from built-in system functions (e.g., access to file systems) to third-party application APIs (e.g., sending emails). For the LLM to be aware of such tools, the list of available tools, as well as their *descriptions* (how to use them), *guidelines* (caveats), and *tool-use examples* (few-shot examples [15]) are conveyed through the input prompt [7, 21, 31, 61, 63] (Figure 2). The outputs of the tool calls, called *observations*, are returned to the LLM so it can decide whether to retry or continue. This fundamental workflow, introduced in ReAct [78], has been extended with new mechanisms, such as adding reflection capabilities [65] or optimal-path search [85] to improve task accuracy.

Among existing approaches, agents that plan the full execution path *before* tool calling are gaining traction [20, 36, 54, 66]. Unlike ReAct, these "plan-out" agents consider interactions among tools when forming the plan. In ReAct, planning one tool call and observing its output require separate LLM calls, so an *N*-step plan

needs  $2 \cdot N$  LLM calls. In LLMCompiler [36], a representative, state-of-the-art plan-out agent, the full plan is generated in one LLM call and all observations are verified in the second LLM call, cutting the total to two LLM calls (an N times reduction). The global view provided with a plan-out agent's full planning capability is known to also improve accuracy [20, 36], and is employed in state-of-the-art agentic services like Gemini 3.0 [26].

Structure of plan-out agents. Figure 2 outlines the overall workflow of plan-out agents. First, given the user query highlighted in blue, the agent retrieves appropriate tools (explained in depth in Section 2.2) and constructs the prompt for the first LLM, the *Planner*. The prompt contains the system prompt, tool descriptions, guidelines, tool-use examples, and the user query. Planner outputs a list of tools plus their arguments, which may be literals (e.g., name "John") or references to prior results (e.g., feeding the output of plan #1 as argument for create\_calendar\_event with \$1).

The execution unit executes the plans in order, parallelizing the execution of tools without dependencies. Each tool call and its output is recorded, forming a list of "call-observation" pairs (green box) that is passed to the second LLM, the *Arbiter*. Based on the input consisting of guidelines, examples, and the call-observation pairs, the Arbiter decides whether the request is satisfied; if not, it signals a retry.

## <span id="page-1-1"></span>2.2 On-device AI Agents

**On-device LLMs.** Al functionalities are now available on edge devices, appearing in various forms like voice transcription [11], Circle-to-Search [25], and personal assistants [9, 24, 59]. Among them, LLMs are increasingly being deployed at the edge, due to their powerful performance. To power edge workloads, compact LLMs for resource-constrained devices have emerged [13, 29]. Hardware accelerators [34, 55, 60, 70, 79, 80] and deployment schemes [3, 18, 64, 72] for on-device LLMs have also been proposed, driving the integration of on-device LLMs into everyday life.

**On-device agents.** On-device agents make use of on-device LLMs to power local agents executing entirely on the user's device. This design mitigates the security and privacy risks inherent in cloud-based solutions. The local LLM processes user requests (e.g., setting reminders) by invoking OS or third-party APIs and completing tasks fully on-device. Among existing systems [9, 24, 59],

<span id="page-2-0"></span>![](_page_2_Figure_1.jpeg)

(a) Offline generation of tool-use example database

![](_page_2_Figure_3.jpeg)

(b) Tool retrieval

![](_page_2_Figure_5.jpeg)

(c) Tool-use example (few-shot example) retrieval

Figure 3: Illustration of the ToolRAG process.

TinyAgent [19] is an open-source macOS agent that fine-tunes LLMs for agentic tasks. It also introduces ToolRAG (Tool Retrieval Augmented Generation) [69] for efficient tool and example selection.

Tool choice and tool-use examples. ToolRAG comprises three components: (i) offline preparation of a tool-use example database, (ii) runtime tool retrieval, and (iii) tool-use example retrieval. A diverse set of user queries is collected offline. Each query is annotated with (1) a fixed-dimension text embedding, (2) the example plan to handle the query, and (3) the set of tools needed to complete the task. Each database entry is stored as a tuple of these three elements: (query, example, tools) in Figure 3(a).

At runtime, the user query is fed into a lightweight classification model (Figure 3(b)). This model outputs a probability score for each of the T available tools. A threshold of  $\tau$  is applied to these scores, retaining only tools whose probability exceeds  $\tau$ . This filtering ensures that only the most relevant tools (e.g., get\_email\_address and create\_calendar\_event) are considered. Next, the tool-use example database is filtered, leaving out entries that include tools not selected by the classification model. The cosine similarities between the user's query embedding and each embedding in the filtered tool-use example database are computed. The top-K examples with the highest similarity scores are retrieved (Figure 3(c)).

<span id="page-2-1"></span>![](_page_2_Figure_11.jpeg)

Figure 4: Illustration of applying prefix caching. Even though there exists substantial overlap between prompt B and the cached prompt, an early token mismatch limits the KV cache reuse.

Finally, the Planner prompt is assembled by combining: (1) detailed descriptions of the selected tools, (2) usage guidelines, and (3) the retrieved tool-use examples (K=3 in Figure 2). Overall, ToolRAG constructs a prompt with highly relevant examples, enabling the on-device agent's Planner to generate better execution plans to accomplish the request.

The mechanism of dynamic tool selection and contextual example retrieval in ToolRAG reflects a broadly adopted paradigm in agent design [7, 21, 31, 61, 63]. Agent systems often use similar workflows to constrain their action spaces and guide generation with task-specific examples. For example, the Model Context Protocol (MCP) [7], an emerging open-source standard for LLM tool access driven by Anthropic, OpenAI, and Google, specifies that tool descriptions and usage examples be provided to the model. Likewise, Google's Function Calling guide for Gemini [30, 42] recommends the filtering of available functions based on conversational context before issuing calls. These shared design principles demonstrate that dynamic tool selection and tool-use example retrieval is not unique to ToolRAG, but a widely employed design paradigm in designing agentic systems.

#### 2.3 LLM Inference Optimization Methods

LLM inference occurs in two stages: prefill and decode. During the prefill stage, the input prompt tokens are processed to populate the key-value (KV) cache, whose length grows with the number of tokens seen. In the decode stage, the model runs autoregressively, generating one token per step. Generally, prefill is compute-intensive, whereas decode is bounded by memory bandwidth [1, 2, 35, 56, 84]. Consequently, the latencies of conversational workloads running on cloud-based high-end GPU servers are known to be dominated by the decode stage (e.g., > 95% in [32, 76, 83]).

**Prefix caching.** In Transformer-based LLMs, each token attends only to preceding tokens. Therefore, when two inputs share a common prefix, they can reuse the same KV cache *up to the first mismatched token. Prefix caching* [37] exploits this by precomputing KV caches for shared prefixes, reducing prefill latency roughly proportional to the portion of cached tokens (i.e., amount of saved computation). As shown in Figure 4, this technique works for prefixes of any length, but cache reuse halts at the first token mismatch, even if subsequent tokens are identical. Throughout this paper, *cacheable* 

<span id="page-3-0"></span>![](_page_3_Figure_1.jpeg)

Figure 5: Illustration of speculative decoding.

<span id="page-3-1"></span>![](_page_3_Figure_3.jpeg)

Figure 6: Latency breakdown of agentic tasks.

tokens refer to portions of prompts that prefix caching can be applied to with a single static prompt, and *uncacheable tokens* refer to portions of prompts that cannot benefit from prefix caching due to an early token mismatch.

**Speculative decoding.** An LLM's decode stage is memory bandwidth-bound because of its autoregressive, one-token-at-a-time nature, yielding low compute intensity. *Speculative decoding* [16, 39] addresses this limitation by producing multiple *draft tokens* per pass. Figure 5 outlines the process, which uses a faster, less accurate *draft model* to aid the generation of a slower, more accurate *target model*.

First, the draft model autoregressively generates N draft tokens (1) with N sequential forward passes. Next, the target model inputs the most recent token ("Schedule") concatenated with the N draft tokens ("a", "meeting", "with", "Sarah") and enters the verification phase. Here, for each draft token, the target model's output logits and the draft model's logits are compared, determining which tokens are accepted. The first mismatch ("Sarah") and all subsequent drafts are discarded (2). Finally, the accepted draft tokens and the target model's chosen next token ("John") form the input for the next speculation round (3). Speculative decoding has been mathematically proven to yield outputs of comparable quality to standard autoregressive generation [16, 39].

## <span id="page-3-4"></span>3 Characterization and Motivation

#### <span id="page-3-5"></span>3.1 Agentic Workload Characterization

To understand the characteristics and implications of on-device agentic tasks, we measure and break down the end-to-end latency of TinyAgent [19]. We use 1,022 examples from the TinyAgent fine-tuning test dataset [68] as benchmarks. These queries span over various types of requests using up to a maximum of 16 different tools. Detailed configuration of experimental setup is provided in Section 5.1. Our evaluation with the TinyAgent-7B backend LLM [67] on Mac mini (M4 Pro) reveals that executing one agentic task takes 35.4 seconds on average. Even a simple task "Schedule a meeting with John tomorrow at 5pm" takes 26.7 seconds to execute, underscoring the latency challenge for on-device agents.

<span id="page-3-2"></span>Table 1: The compute throughput and memory bandwidth available in modern AI chips, both high-end server-class devices and on-device accelerators.

| Device                              | Class     | Compute power<br>(INT8 TOPS) | Memory<br>bandwidth (GB/s) |
|-------------------------------------|-----------|------------------------------|----------------------------|
| NVIDIA H100 [49]                    |           | 1,979                        | 3,350                      |
| NVIDIA H200 [50]                    | Server    | 1,979                        | 4,800                      |
| NVIDIA B200 [51]                    |           | 4,500                        | 8,000                      |
| AMD MI325X [5]                      |           | 2,615                        | 6,000                      |
| Google TPU v6e [28]                 |           | 1,836                        | 1,640                      |
| Apple M4 Max [12]                   |           | 38                           | 546                        |
| Qualcomm Snapdragon<br>X Elite [57] | On-device | 45                           | 135                        |
| AMD Ryzen<br>AI+ PRO 395 [4]        |           | 50                           | 256                        |

<span id="page-3-3"></span>![](_page_3_Figure_13.jpeg)

Figure 7: Token count breakdown of Planner inputs. Results are averaged across all examples in the TinyAgent finetuning test dataset. Static tokens are uncacheable if they are placed behind dynamic tokens.

Figure 6 breaks down the end-to-end latency of agentic task executions. The two LLM components, Planner (43.5%) and Arbiter (46.9%), together account for 90.4% of the total latency. Notably, while decode dominates (68.7%), the prefill stage remains a significant contributor (21.7%). This contrasts with conventional LLM workloads executed in the cloud using server-class devices, where decode latency is overwhelmingly dominant (e.g., over 95%, 98%, and 98% in WSC-LLM [76], FDC [83], and CENT [32]), rendering the LLM decode stage the primary bottleneck to address.

We identify two main reasons for the differing bottlenecks between conventional LLMs and agentic workloads. First, prior work [22] reports that agents typically process much longer input token sequences than they generate as output, a trend we confirmed in our own measurements. This imbalance makes the LLM prefill stage of agents far more compute-intensive than in conventional LLMs. Second, the cost of processing these longer inputs is magnified by limited hardware resources of today's on-device accelerators, rendering prefill disproportionately expensive. Table 1 compares modern AI chips and reveals that on-device accelerators provide at most 11% of the memory bandwidth and roughly 2% of the compute throughput of server-class NVIDIA H200 GPU [50]. Together, these factors make the compute-bound prefill more expensive than decode, leaving both stages as two dominant performance bottlenecks in AI agents.

Overall, because the compute-bound prefill and memory bandwidth-bound decode phases now contribute comparable amounts of latency, no single stage or model instance dominates. Effective acceleration of agentic systems must therefore optimize the *entire* Planner–Arbiter pipeline, improving both prefill and decode. In the remainder of this section, we present a token-level characterization of the Planner and Arbiter prefill and decode stages that motivates our proposed PromptWeaver and ExSpec.

<span id="page-4-0"></span>![](_page_4_Figure_1.jpeg)

Figure 8: Tool co-activation heatmap of a subset of tools in TinyAgent training dataset. The value at (x, y) depicts how likely tool<sub>y</sub> is to be activated given that tool<sub>x</sub> has been activated, i.e.,  $P(\mathsf{tool}_y|\mathsf{tool}_x)$ .

## <span id="page-4-2"></span>3.2 Prefill Stage Token Analysis

**Planner input tokens.** The Planner input consists of the static system prompt, retrieved tool descriptions and guidelines, and retrieved tool-use examples (see Figure 2). Figure 7 shows the average token-count distribution of Planner inputs. Out of the total 1,739 tokens, the static system prompt takes up 32.7%, suggesting that prefix caching could potentially accelerate prefill. However, because the dynamically retrieved tool descriptions and guidelines are inserted into the Planner input early on, the first dynamic token appears after only 1.6% of the prompt, limiting KV cache reuse if the input prompts are used as-is (see Figure 4). Encouragingly, we observe that the dynamically changing tool descriptions and guidelines are different combinations of static fragments, where the combinations depend solely on the selected tool sets. If the tool descriptions and guidelines were made static, the number of uncacheable tokens would decrease by 32% (1,711 to 1,171). Overall, the potential of prefix caching is hindered by early dynamicity introduced by tool descriptions and guidelines, which are merely rearranged chunks of static prompts.

We also observe the existence of tool co-activation locality in the Planner's input prompts, which we define as the likelihood of certain tools being called together across different queries. This skewness in tool co-activation is illustrated in Figure 8 as a heatmap. For example, get\_zoom\_meeting\_link is more likely to be co-activated with get\_email\_address (91%) or compose\_new\_email (58%) than get\_phone\_number (6%). This is because the nature of agentic tasks suggests that tools which fall under the same theme (e.g., email, contacts, maps, and notes) are more likely to be retrieved together in the same plan. This locality is further extended to inter-theme relations, where certain themes like contact and email are "closer" in their relationship than others such as contact and maps. Thus, our characterization reveals that tool co-activation locality is intrinsic to agentic applications.

In the tool-use example retrieval process, the top-K examples displaying the most similar text embedding to the current user query are retrieved (Section 2.2). While users typically query tasks requiring the use of multiple tools (82% of TinyAgent training dataset), we observe that single-tool examples are retrieved as relevant examples 57% of the time. In other words, the single-tool examples are important as tool-use examples for planning tasks. This is not to be confused with the existence of tool co-activation locality in

<span id="page-4-1"></span>![](_page_4_Figure_7.jpeg)

Figure 9: Input and output example of Planner for the query "Schedule a meeting with John tomorrow at 5pm."

the choice of tools, as the dominance of single-tool examples is observed in the choice of tool-use examples.

**Arbiter input tokens.** The Arbiter's input includes decision guidelines and examples, along with the list of call-observation pairs produced by tool execution (see Figure 2). Depending on the internal LLMCompiler state, only two static prefix variants occur; these prefixes account for 88% and 90% of the Arbiter input, respectively. Because the Arbiter input contains such a large static prefix, prefix caching can capture it effectively.

### <span id="page-4-3"></span>3.3 Decode Stage Token Analysis

Planner and Arbiter output tokens. Figure 9 shows an example tool-use example alongside the output for the query "Schedule a meeting with John tomorrow at 5pm." The generated plans share the same structural template as the tool-use example, with arguments substituted to match the user query. This occurs because the tool names, their arguments, and tool-calling orders are mostly embedded in the tool-use examples. Empirically, 96% of Planner and 87% of Arbiter output tokens overlap with those in their corresponding input prompt. Overall, our key observation is that the output tokens generated in both Planner and Arbiter's decode stage are highly correlated with provided examples and user prompt.

While their outputs display highly regular and predictable patterns, reflecting the templates provided by the few-shot examples in the prompt, the decode stages of Planner and Arbiter account for 68.7% of the end-to-end latency. Therefore, we conclude that the decode stage of agentic LLMs is inefficient in that a large portion of time is spent generating straightforward, formulaic sequences that do not require the reasoning capabilities of LLMs.

Challenges of applying speculative decoding. Despite its promise, speculative decoding requires a carefully chosen draft model for high efficiency, and selecting one is nontrivial. If the draft model is too small, it fails to produce high-quality draft tokens and yields little performance improvement. Conversely, a larger draft model achieves higher token accuracy (i.e., the fraction of draft tokens ultimately accepted during verification) but introduces substantial latency overhead.

The *Theoretical max. speedup* column of Table 2 reports the maximum speedup achievable by each draft model under the Planner workload. We compute this theoretical limit analytically, assuming that the LLM decode stage is memory bandwidth-limited and that

<span id="page-5-0"></span>Table 2: Draft token accuracy and achievable theoretical speedups of various draft models.

| Draft model           | Draft token accuracy | Theoretical<br>max. speedup | Speedup<br>(with tax) |
|-----------------------|----------------------|-----------------------------|-----------------------|
| Llama-3.2-3B-Instruct | 0.42                 | 0.96×                       | 0.83×                 |
| Llama-3.2-1B-Instruct | 0.33                 | 1.59×                       | 1.20×                 |
| Llama-160M            | 0.02                 | 0.98×                       | 0.57×                 |
| Llama-68M             | 0.02                 | 1.11×                       | 0.62×                 |

<span id="page-5-1"></span>![](_page_5_Figure_3.jpeg)

Figure 10: Proposed Agent-X system architecture.

its latency therefore scales proportionally with model size. Leveraging this assumption, we combine the draft-token accuracy (second column of Table 2) to estimate the final output-token count and its corresponding latency, which we then compare with the baseline to derive the projected speedup. As shown, even with state-of-the-art small LLMs [45] or fine-tuned draft models from prior work [46], it is difficult to balance the accuracy and draft model size, where smaller draft LLMs barely yield any speedup due to their low draft token accuracy while large draft LLMs with high draft token latency end up spending too much time generating draft tokens.

Now, recall from Figure 5 that speculative decoding requires both single-token inference by the draft LLM (autoregressive token generation) and multi-token inference by the target LLM (parallel verification). However, on-device AI systems are increasingly being optimized for single-batch LLM calls [3, 34, 55, 79, 80]. For example, in Apple's official LLM framework MLX-LM [10], a single-token autoregressive inference with TinyAgent takes 131 ms per token, while the same model's verification phase with 2 tokens takes 244 ms, resulting in a 1.86× slowdown. This phenomenon, which we refer to as the multi-token tax, degrades the overall performance of on-device speculative decoding schemes, where multiple tokens must be verified by the target LLM. Based on this slowdown, we recompute the speedup of applying speculative decoding in the Speedup (with tax) column of Table 2. The best achievable speedup among state-of-the-art small LLMs [45] or draft LLMs fine-tuned by prior work [46] is 1.20×. Therefore, we conclude that applying speculative decoding to on-device frameworks is challenging due to the multi-token tax and draft LLM latency overhead.

## 4 Agent-X: On-device AI Agents Acceleration

#### 4.1 Agent-X Overview

Figure 10 provides an overview of our Agent-X system. Building on the key observations in Section 3, Agent-X utilizes PromptWeaver and ExSpec, which respectively accelerate the prefill and decode

<span id="page-5-3"></span>![](_page_5_Figure_10.jpeg)

Figure 11: Overview of PromptWeaver, divided into the offline (KV cache storage construction) and online phase (KV cache retrieval and prompt construction).

stages of agentic systems. The objective of these techniques is to speed up their target stages without compromising task accuracy or introducing substantial overhead in the resource-constrained on-device environment.

PromptWeaver (Section 4.2) reconstructs the input prompt to minimize the amount of prompt that must be computed on the fly, effectively speeding up the prefill stage. ExSpec (Section 4.3) accelerates the decode stage by using a simple lookup table as the draft model for speculative decoding. This lightweight draft model incurs no draft token generation overhead, while providing means to avoid the multi-token tax. With these two main components, Agent-X accelerates the full system pipeline of on-device agents, targeting both the prefill (PromptWeaver) and the decode (ExSpec) stages.

# <span id="page-5-2"></span>4.2 PromptWeaver: Iso-accuracy Prompt Reconstruction

In Section 3.2, we identified three key traits of Planner inputs: (1) early dynamicity despite a large portion of static prompt, (2) tool co-activation locality, and (3) importance of single-tool examples in tool-use examples. PromptWeaver exploits these observations to reduce uncacheable tokens while preserving accuracy. It rebuilds prompts so that most content forms a cacheable static prefix and the dynamic regions' KV caches can be reconstructed from KV caches stored inside the SSD. Figure 11 provides an overview of how PromptWeaver introduces staticity without degrading accuracy. The reconstructed prompts consist of the following elements, in order: (1) all-inclusive static tool descriptions and guidelines, (2) clustered semi-cacheable tool-use examples, and (3) uncacheable dynamic tool-use examples.

Replacing early dynamic tokens. The first dynamic fragment of prompt contains descriptions and usage guidelines for the selected tools. Because the set of tools selected depends on the input prompt and is thus determined at runtime, this section is where dynamicity is introduced. PromptWeaver reconstructs this section to include descriptions and guidelines for *all tools* available in the system. This replacement transforms the early dynamic segment into a larger but fully static prefix. Although this increases the size of

<span id="page-6-0"></span>![](_page_6_Figure_1.jpeg)

Figure 12: Illustration of PromptWeaver's offline KV cache precompute mechanism.

the KV cache, it enables the prompts made static to be *precomputed* and have their KV caches stored in the SSD. Since static prefixes can be cached once and reused across requests, our approach significantly reduces the number of uncacheable tokens in the prompt. Concretely, tokens previously marked as "Static (uncacheable)" in Figure 7 are now part of the cacheable prefix, thereby reducing the dynamic token volume at runtime at the cost of a larger KV cache footprint.

In the process of inducing staticity, the input prompt grows longer. The average length of a single tool's description and guidelines is 120 tokens. For an agentic system using up to t tools, including all descriptions and guidelines requires  $120 \cdot t$  tokens. Even with a conservative estimate of 100 tools, our all-inclusive static tool description adds only 1.4 GB of additional KV cache stored on the SSD. In general, the performance benefits of our approach far outweigh the overhead introduced by this larger KV cache.

Tool-use example selection. The remaining uncacheable input tokens primarily consist of tool-use examples. These examples are dynamic because they are selected at runtime based on the user prompt (Section 2.2) [21, 61, 63]. The all-inclusive static tool description and guidelines cannot be applied to tool-use examples for two reasons. First, supplying the Planner with all possible combinations of tool-use examples requires  $2^t - 1$  different combinations for a system with t tools. For the current TinyAgent with 16 tools, this is equivalent to 800 GB of memory, increasing exponentially as more tools are supported (memory  $\propto 2^t$ ). Second, the few-shot examples (i.e., tool-use examples) serve as templates and have a direct impact on the generation quality of LLMs. Adding such an excessive amount of tool-use examples would hamper the Planner's ability to extract relations between different tools [21, 30, 61]. Instead, PromptWeaver chooses a fixed set of tool-use examples that can be reused across different queries at the offline phase through (1) co-activation locality-based tool clustering, (2) theme-based cluster ordering, and (3) cluster combination selection, as illustrated in Figure 12.

Co-activation locality-based tool clustering. In this stage, tools are clustered based on their co-activation locality (Figure 8). First, the tools are annotated with themes that relate to the tool's usage (1), information that PromptWeaver utilizes in the next theme-based cluster ordering stage. Then, PromptWeaver refers to the Planner's training dataset (TinyAgent fine-tuning training

dataset, Section 3.1) and iterates through all training data samples' ground truth labels to identify which tools are called together (2). With this information, we generate a tool co-activation matrix, which is an adjacency matrix where each graph node corresponds to a tool and edges denote the number of times the corresponding pair of tools has been called together, and apply non-negative matrix factorization (NMF) [38] to cluster the tools based on the coactivation matrix (3). This results in a total of eight clusters, each consisting of 2 to 6 tools. After the tools are clustered, we assign each cluster one tool-use example that uses exactly the tools in that cluster. These clusters then undergo theme-based ordering and cluster combination selection (detailed later) before being stored in SSD for retrieval during inference. For example, when servicing a user query, ToolRAG retrieves the set of tools likely to be used for the request (Figure 3(b)). A cluster is considered "activated" if it contains at least one of these tools, and the corresponding tool-use examples from the activated clusters are included as few-shot examples in the prompt ("Clustered tool-use examples" in Figure 11).

Theme-based cluster ordering. To maximize KV cache reuse, clusters must be given a fixed ordering so that when the same combination of clusters are activated, the resulting KV cache yields the same value. That is, because having clusters A and B in order "A-B" and "B-A" yield different KV cache values, we impose a fixed ordering so that between clusters A and B, the order is always statically fixed as "A-B" (or "B-A"). To maximize the amount of KV cache reuse under a fixed SSD capacity budget, PromptWeaver applies theme-based clustering and cluster combination selection. The objective of theme-based ordering is to place clusters that are frequently co-activated adjacent to each other. Each cluster is assigned a theme based on which tool theme is most dominant in the cluster. Then, clusters with identical themes are grouped together and placed next to each other in a fixed order (4). For example, clusters with the same theme "Email" are grouped together and placed adjacent to each other in order. If the tool get\_email\_address (Email\_A) is retrieved, clusters 1 and 3, which include the tool, are activated. When the clusters are ordered based on their themes (after ordering in 4), having stored the KV cache of clusters "1-3-2" would fully cover for tool get\_email\_address, because if cluster 2 is not activated, we can cut out the mismatched tail (cluster 2) and reuse the KV cache of "1-3." However, if clusters were not grouped by themes and put in random order (before ordering in

#### <span id="page-7-1"></span>Algorithm 1 Cluster combination selection

```
1: Input: KV cache budget N, Planner's training dataset \mathcal{D}
     Output: cluster combination KV cache C
 3: Initialize C \leftarrow \emptyset
     Initialize prefixes P \leftarrow \{\text{all prefixes of sequences in } \mathcal{D}\}
 4:
     for i = 1 to N do
 5:
          options \leftarrow \emptyset
          for all prefix p in P do
 7:
 8:
              if len(p) == 1 or p[:-1] in C then
                  options.add(p)
 9:
              end if
10:
11:
          end for
          \hat{p} = \arg \max \left[ \operatorname{coverage}(\mathcal{D}, C \cup \{p\}) - \operatorname{coverage}(\mathcal{D}, C) \right]
12.
13:
          C.add(\hat{p})
14: end for
15: return C
```

(1-2-3" and "1-3") depending on whether or not cluster 2 is activated. Thus, with a fixed, theme-based ordering, PromptWeaver can maximize prefix cache reuse across various requests.

**Cluster combination selection.** The total number of possible cluster combinations is  $2^C - 1$  for C ordered clusters. PromptWeaver must carefully select combinations of clusters to store in the SSD to maximize KV cache reuse across requests, while keeping the SSD capacity overhead low. Algorithm 1 explains our proposed combination selection algorithm. Given a cache budget of N clusters, starting from an empty KV cache set C, all available cluster prefixes of all data samples in the Planner's training dataset ( $\mathcal{D}$ , same one used in locality-based clustering) are gathered (line 4). That is, if a sample activates clusters "A-B-C", its prefix sequences "A," "A-B," and "A-B-C" are all gathered. We define a new metric to give scores for each combination of cached clusters C. For each activated cluster sequence in  $\mathcal{D}$ , the length of the longest cached prefix (i.e., how many leading clusters can reuse the KV caches in C) is measured. The *coverage* is defined by the sum of these per-sequence number of hit clusters in  $\mathcal{D}$ . A larger coverage implies a higher KV cache reuse across requests. Each step considers all candidate prefixes, either a singleton cluster (len(p) == 1 in line 8, first dotted box in **5**) or an extension to an existing sequence by one cluster of a prefix already owned (p[:-1] in C in line 8, second and third dotted boxes in 6), and computes how much coverage would increase if that candidate were added to the cached set (line 12). By greedily choosing the prefix that adds the largest amount of new coverage, the algorithm focuses on the most frequent early patterns without exploring every combination. The returned cluster combinations C have their KV cache precomputed and saved in the SSD for reuse at the online phase (**5**). In Section 5.2, we show that with a cache budget size of just 15 clusters (5.87 GB of SSD capacity overhead), 74.4% of tool-use examples are covered. This demonstrates that a small, carefully chosen subset of cluster combinations can serve the vast majority of prompt patterns, dramatically reducing SSD capacity overhead while preserving high KV cache reuse.

Preserving task accuracy with dynamic tool-use examples. LLM output quality is known to be heavily dependent on the quality of few-shot examples in the prompt [15, 41, 48]. Therefore, composing the entirety of tool-use examples with clustered examples can have critical impact on the Planner accuracy. To alleviate this, we

<span id="page-7-3"></span>![](_page_7_Figure_6.jpeg)

Figure 13: Overview of ExSpec with trigram (*n*=3) LUT and draft token generation length of 4.

append single-tool examples<sup>1</sup> of activated tools, which we observed to be the most popular form of tool-use examples, to the end of the clustered examples. We also add top-K ( $0 \le K \le 4$ ) relevant examples from ToolRAG (Figure 3(c)) to make up for any accuracy loss. In Section 5.2, our evaluations reveal that K = 1 is the optimal choice in terms of accuracy, requiring only one additional tool-use example worth of dynamic tokens.

# <span id="page-7-0"></span>4.3 ExSpec: Example-based Selective Speculative Decoding

Section 3.3 uncovered two key properties of Planner and Arbiter output tokens: (1) decoded outputs are largely predictable from input prompts, and (2) applying speculative decoding naïvely in on-device frameworks leads to suboptimal performance due to penalties from the multi-token tax and draft LLM latency overheads. We propose ExSpec to leverage this predictability without incurring these penalties. Figure 13 illustrates the two main mechanisms of ExSpec: (1) lightweight draft token generation using an *n*-gram model [17] and (2) selective fallback to autoregressive generation when speculative decoding is likely to be inefficient.

**Draft LLM-free, example-based speculative decoding.** A primary design objective of ExSpec is to remain lightweight while keeping decoded outputs tightly correlated with the input prompt. To this end, ExSpec builds a simple n-gram lookup table (LUT) on the fly each time the agent receives a new user query. The LUT is populated as follows. To keep this table task-specific, we build the table from the few-shot examples (tool-use examples for Planner and decision examples for Arbiter) and the user query, all of which constitute a single stream of tokens. We then slide a window of n consecutive tokens  $t_{1:n}$  across this token stream, shifting one token at a time, and record each pair  $\langle t_{1:n-1}, t_n \rangle$ . The prefix  $t_{1:n-1}$  serves as the key in the n-gram LUT, and the corresponding value is the token  $t_n$  that occurs most frequently with that key across all recorded pairs. Using this LUT, ExSpec generates each draft

<span id="page-7-2"></span> $<sup>^1\</sup>mathrm{We}$  add double-tool examples for a select few tools whose single-tool example was not present in the offline-generated database.

<span id="page-8-2"></span>![](_page_8_Figure_1.jpeg)

Figure 14: (a) Planner accuracy change vs. the number of appended tool-use examples. (b) Share of cacheable and uncacheable tokens (left) and reduction in uncacheable tokens (right).

token by indexing the table with the most recent n-1 generated tokens (gray tokens in Figure 13) and the corresponding output becomes the draft token (blue token). A key advantage of ExSpec's LUT design is its minimal memory footprint, only amounting to a few KB. This lightweight design is ideal for on-device AI systems, unlike conventional LLM-based drafts, which consume hundreds of MB to several GB of memory (Table 2).

Balancing n is crucial for maximizing ExSpec's performance. With a small n, for example a unigram model (n=1), the LUT always proposes the single most frequent token from the few-shot examples and user query, yielding low-quality draft tokens. Conversely, a large n provides richer context and higher-quality drafts but often fails to predict unseen sequences, defaulting to random tokens when the n-1 prefix is not available in the LUT. We find that a trigram model (n=3) offers the best trade-off between draft quality and LUT hit rate. The impact of n is further explored in Section 5.5.

Multi-token-tax evasion with selective decoding. When the current (n-1)-token context is absent from the LUT, our ngram draft model "guesses" a random token that rarely passes verification. Verification still generates the same output tokens via standard autoregressive decoding and incurs the multi-token tax, ultimately slowing down overall token generation. ExSpec mitigates this limitation by opting out of speculative decoding when no speedup is expected. Specifically, ExSpec consults the LUT before generating the first draft token: if no valid entry exists, it immediately falls back to standard autoregressive decoding (yellow tokens in Figure 13). This safeguard ensures speculative decoding is used only when its speedup outweighs the verification cost. Because the *n*-gram LUT deterministically knows which contexts it covers, this decision incurs zero overhead—an assurance LLM-based drafts cannot provide. Once the first draft token is retrieved from the LUT, the remaining tokens in that group are generated regardless of subsequent misses.

#### 5 Evaluation

## <span id="page-8-0"></span>5.1 Methodology

**Model and dataset.** We target TinyAgent [19], an open-source, on-device agent framework for macOS built on LLMCompiler [36]. Our experiments use TinyAgent-7B [67] as the backend LLM, a fine-tuned variant of WizardLM-2-7B [75]. TinyAgent's fine-tuning

<span id="page-8-3"></span>![](_page_8_Figure_9.jpeg)

Figure 15: Tool-use example coverage (left, red) and total storage overhead (right, black) by KV cache budget.

dataset [68] serves three roles: PromptWeaver uses the training split for tool clustering and combination selection, and the test split is used for evaluation. Before each task, we flush the system's page cache to isolate the cost of loading the KV cache from storage.

Hardware and software. Experiments run on an Apple Mac mini with an M4 Pro chip [12], 64 GB of memory, 512 GB of SSD storage, 12 CPU cores, and 16 GPU cores. We build PromptWeaver and ExSpec on MLX-LM [10] and MLX-engine [43]. MLX-LM is Apple Silicon's official LLM inference package, backed by MLX [33]. MLX-engine is an open-source MLX LLM engine from LM Studio. The software stack comprises MLX v0.25.2, a modified MLX-LM v0.25.1, and MLX-engine commit #ecc2cf4 on macOS Sequoia 15.5.

#### <span id="page-8-1"></span>5.2 PromptWeaver

**Planner accuracy.** In line with prior work [19, 36], we define the Planner accuracy by constructing a Directed Acyclic Graph (DAG) from the output plan, where each node represents a function call and a directed edge represents the dependency, and compare the generated DAG against the ground truth's DAG. Because the tool calls are deterministic, the Planner accuracy directly translates to the end-to-end task accuracy of the agent. PromptWeaver adds  $K(0 \le K \le 4)$  tool-use examples to make up for its accuracy loss. Figure 14(a) compares the Planner's task accuracy changes with the baseline as more examples are added. Not surprisingly, K=0 results in lower accuracy (0.832), lower than baseline's 0.836. We observe that the accuracy peaks at K=1 (0.841) and falls as more examples are added.

Figure 14(b) shows how the share of cacheable vs. uncacheable tokens changes as tool-use examples are added. Each new example is uncacheable, so the uncacheable fraction grows from 11% at K=0 to 21% at K=4. Accuracy peaks at K=1, so we adopt this setting for PromptWeaver. With K=1, PromptWeaver averages 519 uncacheable tokens, a 70% drop from the baseline's 1,711. This result highlights PromptWeaver's ability to minimize uncacheable tokens.

**Storage overhead.** Precomputed KV caches are stored in SSD, and loaded to memory on demand to reduce online prefill compute. Because the agentic workloads display long prompt lengths, the size of the KV cache is also large. Expanding the cluster budget boosts tool-use example coverage (Section 4.2) but also raises storage cost, as shown in Figure 15. At budget 0, only static tokens are cached, using 0.95 GB (0.57 GB, Planner + 0.39 GB, Arbiter). Coverage grows with larger budgets but levels off beyond 15 clusters. We therefore fix the budget at 15 clusters for the remainder of the evaluation, using 6.26 GB of storage for 74.4% coverage.

<span id="page-9-1"></span>![](_page_9_Figure_1.jpeg)

Figure 16: Prefill stage latency (left) and speedup gain (right) of PromptWeaver.

<span id="page-9-2"></span>![](_page_9_Figure_3.jpeg)

Figure 17: Decode stage latency (left) and speedup (right) of speculative decoding (SpecDec) and ExSpec.

Speedup analysis. Figure [16](#page-9-1) quantifies impact of PromptWeaver on Planner and Arbiter prefill stages. The Static design, caching only static tokens for the prompt, adds just a 1.01× speedup for Planner because while the amount of static tokens increases, the number of uncached dynamic tokens stays roughly the same (15.3% decrease). In PromptWeaver, clustered dynamic caching is applied on top of static token caching, achieving a 49.6% reduction in the number of uncached input tokens and a 1.57× speedup. Meanwhile, PromptWeaver reduces the amount of uncached input tokens by 88.9% thanks to Arbiter's mostly static input tokens, achieving 4.35× speedup. Overall, loading KV cache from SSD storage adds minor overhead compared to the speedup achieved by compute savings, accounting for 5.8% and 11.7% of prefill latency in Planner and Arbiter, respectively.

## 5.3 ExSpec

Speedup analysis. Figure [17](#page-9-2) reports the decode stage latency of speculative decoding with draft LLM Llama-3.2-1B-Instruct [\[45\]](#page-12-25) (SpecDec) and ExSpec with a trigram (=3) draft model. SpecDec experiences slowdown rather than speedup over the baseline. Other than the multi-token tax, the overhead that comes from dealing with different tokenizers between the target and the draft model [\[71\]](#page-13-15) further slows down the system to achieve a much slower speed than was expected from Table [2.](#page-5-0) On the contrary, the non-selective design of ExSpec effectively avoids any draft LLM-related overheads, speeding up the decode stage by 1.38×. With selective decoding, ExSpec can avoid the multi-token tax when no draft tokens are accepted, falling back to autoregressive generation 17 (Planner) and 37 (Arbiter) times per query. Overall, ExSpec reduces the decode latency by 1.73×, establishing itself as a lightweight and performant solution.

Draft token accuracy. Table [3](#page-9-3) compares draft-token accuracy under selective and non-selective ExSpec. Although non-selective decoding produces many more draft tokens, both modes accept

<span id="page-9-3"></span>Table 3: Draft token accuracy comparison between nonselective and selective decoding schemes of ExSpec.

| Workflow<br>component | Applied<br>method | Generated<br>draft tokens | Accepted<br>draft tokens | Draft token<br>accuracy |
|-----------------------|-------------------|---------------------------|--------------------------|-------------------------|
| Planner               | Non-selective     | 364                       | 48                       | 0.13                    |
|                       | Selective         | 194                       | 48                       | 0.25                    |
| Arbiter               | Non-selective     | 622                       | 56                       | 0.09                    |
|                       | Selective         | 218                       | 56                       | 0.26                    |

<span id="page-9-4"></span>![](_page_9_Figure_11.jpeg)

Figure 18: End-to-end latency (left) and speedup (right) of Agent-X. PW denotes PromptWeaver, ES denotes ExSpec, and PW+ES denotes the full Agent-X pipeline.

the same number, underscoring the value of reverting to autoregressive generation when acceptance chances are low. Selective decoding thus increases overall accuracy, falling back to autoregressive generation on average 17 (Planner) and 37 (Arbiter) times per query.

-gram model generation overhead. While the -gram model incurs constant-time latency per lookup, building the LUT takes () time for input token length . The LUT generation takes up 83 milliseconds per query, indicating that it incurs a negligible overhead in terms of latency.

## 5.4 Agent-X Full System Integration

End-to-end speedup. Figure [18](#page-9-4) shows the latency and speedup of end-to-end on-device agentic workloads. The impacts of Prompt-Weaver (PW), ExSpec (ES), and Agent-X (PW+ES) are shown. Applying PromptWeaver and ExSpec independently provides end-to-end speedups of 1.16× and 1.43×, respectively. Applying PromptWeaver and ExSpec together reaps an end-to-end speedup of 1.61×. Alongside the speedup, it is worth noting that Agent-X is a purely software solution that can be directly applied to existing on-device agentic systems without degrading the accuracy of the agent.

## <span id="page-9-0"></span>5.5 Discussions

Application to other platforms, agents, and models.As Prompt-Weaver and ExSpec are both purely software solutions, Agent-X can easily be ported to other hardware platforms. While our prototype is implemented on macOS due to the maturity of its local LLM ecosystem (TinyAgent), the core algorithms of Agent-X are hardware-agnostic and applicable to other platforms. Because the fine-tuned model and datasets provided by TinyAgent are available only in macOS environments, an end-to-end evaluation of Agent-X on other hardware platforms is challenging. We also evaluate Agent-X on the smaller model TinyAgent-1.1B. PromptWeaver and ExSpec achieve speedups of 1.62× (prefill) and 1.42× (decode), confirming their efficacy on smaller LLMs.

<span id="page-10-0"></span>![](_page_10_Figure_1.jpeg)

Figure 19: Decode latency (left) and speedup (right) of ExSpec with varying extraction regions for LUT.

Fine-tuned models as draft LLM. We explore using TinyAgent-1.1B, fine-tuned for agentic tasks, as the draft model in applying speculative decoding to target TinyAgent-7B. Our experiments under the same setting as Figure 17 show it is  $1.81\times$  slower than the baseline. Its draft token accuracy displays high variance across tasks, exhibiting unstable latency. Based on the TinyAgent paper [19], we estimate its fine-tuning cost as 5 ExaFLOPs ( $5\times10^{18}$  FLOPs), and even at 100% compute utilization, it would take 75 hours to fine-tune TinyAgent-1.1B on our evaluation platform. Overall, the added fine-tuning cost and the resulting latency slowdown demonstrate that using a fine-tuned draft LLM is not beneficial, underscoring the effectiveness of ExSpec.

Robustness to tool-use distribution drift. To assess the effect of tool-use pattern drifts on PromptWeaver, we evaluate a case where notes-related tools, namely create\_note, open\_note, and append\_note\_content, are disabled. PromptWeaver only experiences a small accuracy drop (0.8%p) and maintains high tool-use example coverage with 75.7% at cache budget of 15 clusters. Overall, these numbers are close to those reported in Figure 14 and Figure 15, demonstrating the robustness of PromptWeaver.

Effect of lengthened input prompt on decode latency. Prompt-Weaver increases the average input tokens from 1,739 to 3,790 to enable prefix caching. This translates to an additional 256 MB of memory for the KV cache, forcing the memory bandwidth-bound decode stage to load more data. Consequently, the normalized decode stage latency, quantified as *Time-Per-Output-Token*, increases by 2.2% (from 122 ms to 125 ms), consistent with a 1.7% rise in overall memory usage. However, the prefill speedup outweighs this minor decode overhead. Furthermore, additional tokens only apply to the Planner, reducing its impact on the full system pipeline.

**Prompt extraction region.** ExSpec constructs the n-gram LUT using a portion of the input. To study the effect of the extraction region, we compare two cases: ExSpec (all), which uses the entire input, and ExSpec (few-shot), our proposed method, which extracts the few-shot examples and the user query. As shown in Figure 19, ExSpec (all) still achieves a solid  $1.70\times$  speedup over the baseline. However, ExSpec (few-shot) delivers an additional 3% and 1% speedup over ExSpec (all) for the Planner and Arbiter, respectively. This indicates that the Planner is more sensitive to the choice of extraction region than the Arbiter. We attribute this sensitivity to the input size, where the Planner is  $2.20\times$  larger than Arbiter, therefore "polluting" the LUT with excess tokens.

**Various** n-**gram draft models**. We examine how n in ExSpec's n-gram model affects its effectiveness. A bigram model (n=2) causes

the draft token accuracy to drop sharply to 0.10, a significant decrease compared to the default (n=3), which achieves 0.25. This result highlights the importance of the context length (n-1) in our n-gram draft model in generating accurate predictions. A quadgram model (n=4) further improves accuracy to 0.31, but produces only 72% as many draft tokens as the trigram model. This reduction in the quadgram model stems from its longer context. A longer context makes ExSpec more conservative when generating draft tokens and increases the likelihood of falling back to autoregressive generation. Therefore, while a longer context increases accuracy, it reduces the number of draft tokens, ultimately leading to a 5.1% slower total decode latency compared to the trigram model.

#### 6 Related Work

On-device AI. Various techniques facilitate the adoption of AI on edge devices. FACIL [62] applies processing-in-memory to overcome the memory capacity and bandwidth limitations of edge devices. DecDEC [55] introduces an aggressive low-bit quantization scheme for efficient inference. These works accelerate LLM inference in general, and can be applied to Agent-X for further speedup. AppAgent [81] mimics human-like interactions to function without system access. Mobile-Agent [73, 74] leverages visual capabilities to identify and locate elements in the devices. Overall, these target multi-modal LLM-based agents, whereas Agent-X focuses on conventional agentic workflows with text-based LLMs.

KV cache reuse. A rich body of prior work makes use of precomputed KV caches to reduce runtime latency. Prefix caching [37] reuses KV caches from previous sequences when there are exact prefix matches to the incoming request. However, its reuse is limited when there are mismatches early on in the prompt. Prompt Cache [23] reuses the full KV cache even when there are token matches that do not start at the beginning of the input, and CacheBlend [77] selectively recomputes KV caches to maximize reuse while maintaining accuracy. Agent-X reuses KV caches by reconstructing the prompts at the text level, targeting the semantic similarity, without any recomputation at runtime.

Speculative decoding. To facilitate speculative decoding [16, 39], works like Eagle [40] reduce the training cost of draft LLMs by reusing target model's logits, whereas self-speculative decoding [82] reuses portions of the target model as draft models, completely removing the need for any retraining. Agent-X is distinct from these works in that the lookup table used as the draft model requires no training, and its lightweight, constant time lookup speed allows for speedup directly proportional to the draft token latency. PLD [8] constructs an LUT from the user prompt to accelerate input-grounded tasks like summarization. Agent-X is different from PLD in that it pinpoints specific parts of the prompt to construct an LUT, and applies selective decoding to avoid the multi-token tax when possible.

#### 7 Conclusion

We propose Agent-X, an on-device agent acceleration solution with no task accuracy degradation. With its two components Prompt-Weaver and ExSpec, it accelerates prefill stage of agentic LLMs by 1.97× and decode stage by 1.73×. Overall, Agent-X delivers an end-to-end speedup of 1.61× on real on-device agents. Because Agent-X

is a purely software-based solution, it can be applied seamlessly to existing on-device agentic systems. To the best of our knowledge, this is the first work to directly tackle the LLM bottlenecks of ondevice agents by leveraging their unique task-level characteristics, under resource-constrained on-device environments.

## Acknowledgments

This work was partly supported by Institute of Information & Communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (No.RS-2024-00395134, DPU-Centric Datacenter Architecture for Next-Generation AI Devices), (No.RS-2024-00438851, (SW Starlab) High-performance Privacypreserving Machine Learning System and System Software), (No. RS-2024-00457882, AI Research Hub Project), (No.RS-2025-02214652, Development of SoC Technology for AI Semiconductor-Converged Pooled Storage/Memory), and Samsung Electronics Co., Ltd(IO251210- 14212-01). Minsoo Rhu is the corresponding author.

## References

- <span id="page-11-23"></span>[1] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, and Ramachandran Ramjee. 2024. Taming Throughput-latency Tradeoff in LLM Inference with Sarathi-serve. In Proceedings of the USENIX Symposium on Operating Systems Design and Implementation (OSDI).
- <span id="page-11-24"></span>[2] Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, and Ramachandran Ramjee. 2023. SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills. In arxiv.org.
- <span id="page-11-20"></span>[3] Keivan Alizadeh, Seyed Iman Mirzadeh, Dmitry Belenko, S Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, and Mehrdad Farajtabar. 2024. LLM in a Flash: Efficient Large Language Model Inference with Limited Memory. In Proceedings of the ACL (Association for Computational Linguistics).
- <span id="page-11-2"></span>[4] AMD. 2025.<https://www.amd.com/en/products/processors/laptop/ryzen.html>
- <span id="page-11-25"></span>[5] AMD. 2025. AMD Instinct MI325X Accelerator. [https://www.amd.com/content/](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf) [dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf)[datasheet.pdf](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf)
- <span id="page-11-0"></span>[6] Anthropic. 2024.<https://www.anthropic.com/solutions/agents>
- <span id="page-11-9"></span>[7] Anthropic. 2024. Introducing the Model Context Protocol. [https://www.](https://www.anthropic.com/news/model-context-protocol) [anthropic.com/news/model-context-protocol](https://www.anthropic.com/news/model-context-protocol)
- <span id="page-11-30"></span>[8] Apoorv Saxena. 2023. Prompt Lookup Decoding. [https://github.com/](https://github.com/apoorvumang/prompt-lookup-decoding) [apoorvumang/prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding)
- <span id="page-11-16"></span>[9] Apple. 2010. Siri.<https://www.apple.com/siri>
- <span id="page-11-6"></span>[10] Apple. 2023.<https://github.com/ml-explore/mlx-lm>
- <span id="page-11-14"></span>[11] Apple. 2024. Apple Intelligence.<https://www.apple.com/apple-intelligence>
- <span id="page-11-3"></span>[12] Apple. 2024. Apple Introduces M4 Pro and M4 Max. [https://www.apple.com/](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max) [newsroom/2024/10/apple-introduces-m4-pro-and-m4-max](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max)
- <span id="page-11-18"></span>[13] Apple. 2024. Introducing Apple's On-device and Server Foundation Models. [https:](https://machinelearning.apple.com/research/introducing-apple-foundation-models) [//machinelearning.apple.com/research/introducing-apple-foundation-models](https://machinelearning.apple.com/research/introducing-apple-foundation-models)
- <span id="page-11-4"></span>[14] Apple Developer. 2025. [https://developer.apple.com/documentation/](https://developer.apple.com/documentation/FoundationModels) [FoundationModels](https://developer.apple.com/documentation/FoundationModels)
- <span id="page-11-8"></span>[15] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-shot Learners. In Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS).
- <span id="page-11-5"></span>[16] Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023. Accelerating Large Language Model Decoding with Speculative Sampling. In arxiv.org.
- <span id="page-11-28"></span>[17] Stanley F Chen and Joshua Goodman. 1999. An Empirical Study of Smoothing Techniques for Language Modeling. Computer Speech & Language 13, 4 (1999), 359–394.
- <span id="page-11-21"></span>[18] Yongheng Deng, Ziqing Qiao, Ye Zhang, Zhenya Ma, Yang Liu, and Ju Ren. 2025. CrossLM: A Data-free Collaborative Fine-tuning Framework for Large and Small Language Models. In Proceedings of the International Conference on Mobile Systems, Applications, and Services.
- <span id="page-11-7"></span>[19] Lutfi Eren Erdogan, Nicholas Lee, Siddharth Jha, Sehoon Kim, Ryan Tabrizi, Suhong Moon, Coleman Hooper, Gopala Anumanchipalli, Kurt Keutzer, and

- <span id="page-11-12"></span>Amir Gholami. 2024. TinyAgent: Function Calling at the Edge. In arxiv.org. [20] Lutfi Eren Erdogan, Nicholas Lee, Sehoon Kim, Suhong Moon, Hiroki Furuta, Gopala Anumanchipalli, Kurt Keutzer, and Amir Gholami. 2025. Plan-and-Act: Improving Planning of Agents for Long-horizon Tasks. In arxiv.org.
- <span id="page-11-10"></span>[21] Tiantian Gan and Qiyao Sun. 2025. RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-augmented Generation. In arxiv.org.
- <span id="page-11-27"></span>[22] Gemini Team. 2025. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. In arxiv.org.
- <span id="page-11-29"></span>[23] In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, and Lin Zhong. 2024. Prompt Cache: Modular Attention Reuse for Low-latency Inference. In Proceedings of Machine Learning and Systems (MLSYS).
- <span id="page-11-17"></span>[24] Google. 2024.<https://gemini.google/assistant>
- <span id="page-11-15"></span>[25] Google Blog. 2024. Circle (or Highlight or Scribble) to Search. [https://blog.](https://blog.google/products/search/google-circle-to-search-android) [google/products/search/google-circle-to-search-android](https://blog.google/products/search/google-circle-to-search-android)
- <span id="page-11-13"></span>[26] Google Blog. 2025. A New Era of Intelligence with Gemini 3. [https://blog.google/](https://blog.google/products/gemini/gemini-3/) [products/gemini/gemini-3/](https://blog.google/products/gemini/gemini-3/)
- <span id="page-11-1"></span>[27] Google Blog. 2025. Gemini CLI: Your Open-source AI Agent. [https://blog.google/](https://blog.google/technology/developers/introducing-gemini-cli-open-source-ai-agent/) [technology/developers/introducing-gemini-cli-open-source-ai-agent/](https://blog.google/technology/developers/introducing-gemini-cli-open-source-ai-agent/)
- <span id="page-11-26"></span>[28] Google Cloud. 2024. TPU v6e.<https://cloud.google.com/tpu/docs/v6e>
- <span id="page-11-19"></span>[29] Google DeepMind. 2023.<https://deepmind.google/models/gemini/nano>
- <span id="page-11-22"></span>[30] Google Developers. 2025.<https://ai.google.dev/gemini-api/docs/function-calling>
- <span id="page-11-11"></span>[31] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni,

Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. 2024. The Llama 3 Herd of Models. In arxiv.org.

- <span id="page-12-20"></span>[32] Yufeng Gu, Alireza Khadem, Sumanth Umesh, Ning Liang, Xavier Servot, Onur Mutlu, Ravi Iyer, and Reetuparna Das. 2025. PIM is All You Need: A CXL-enabled GPU-free System for Large Language Model Inference. In Proceedings of the International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS).
- <span id="page-12-30"></span>[33] Awni Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert. 2023. <https://github.com/ml-explore/mlx>
- <span id="page-12-13"></span>[34] Mingqiang Huang, Ao Shen, Kai Li, Haoxiang Peng, Boyu Li, Yupeng Su, and Hao Yu. 2025. EdgeLLM: A Highly Efficient CPU-FPGA Heterogeneous Edge Accelerator for Large Language Models. IEEE Transactions on Circuits and Systems I: Regular Papers (2025).
- <span id="page-12-18"></span>[35] Aditya K. Kamath, Ramya Prabhu, Jayashree Mohan, Simon Peter, Ramachandran Ramjee, and Ashish Panwar. 2025. POD-attention: Unlocking Full Prefill-decode Overlap for Faster LLM Inference. In Proceedings of the International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS).
- <span id="page-12-0"></span>[36] Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W Mahoney, Kurt Keutzer, and Amir Gholami. 2024. An LLM Compiler for Parallel Function Calling. In Proceedings of the International Conference on Machine Learning (ICML).
- <span id="page-12-21"></span>[37] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention.

- In Proceedings of the ACM Symposium on Operating System Principles (SOSP).
- <span id="page-12-27"></span>[38] Daniel D Lee and H Sebastian Seung. 1999. Learning the Parts of Objects by Non-negative Matrix Factorization. Nature 401, 6755 (1999), 788–791.
- <span id="page-12-7"></span>[39] Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast Inference from Transformers via Speculative Decoding. In Proceedings of the International Conference on Machine Learning (ICML).
- <span id="page-12-32"></span>[40] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024. EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty. In Proceedings of the International Conference on Machine Learning (ICML).
- <span id="page-12-28"></span>[41] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2021. What Makes Good In-Context Examples for GPT-3?. In arxiv.org.
- <span id="page-12-17"></span>[42] Shiyi Liu, Haiying Shen, Shuai Che, Mahdi Ghandi, and Mingqin Li. 2025. HERA: Hybrid Edge-cloud Resource Allocation for Cost-efficient AI Agents. In arxiv.org.
- <span id="page-12-8"></span>[43] LM Studio. 2024.<https://github.com/lmstudio-ai/mlx-engine>
- <span id="page-12-1"></span>[44] Manus AI. 2025. Manus.<https://manus.im/>
- <span id="page-12-25"></span>[45] Meta. 2024. Llama 3.2: Revolutionizing Edge AI and Vision with Open, Customizable Models. [https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices)[mobile-devices](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices)
- <span id="page-12-26"></span>[46] Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. 2024. SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification. In Proceedings of the International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS).
- <span id="page-12-4"></span>[47] Microsoft. 2024. [https://www.microsoft.com/en-us/windows/business/devices/](https://www.microsoft.com/en-us/windows/business/devices/copilot-plus-pcs) [copilot-plus-pcs](https://www.microsoft.com/en-us/windows/business/devices/copilot-plus-pcs)
- <span id="page-12-29"></span>[48] Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2022. MetalCL: Learning to Learn in Context. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics (NAACL).
- <span id="page-12-22"></span>[49] NVIDIA. 2024. NVIDIA H100 Tensor Core GPU. [https://resources.nvidia.com/en](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-tensor-core-gpu-datasheet)[us-hopper-architecture/nvidia-tensor-core-gpu-datasheet](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-tensor-core-gpu-datasheet)
- <span id="page-12-23"></span>[50] NVIDIA. 2024. NVIDIA H200 Tensor Core GPU. [https://nvdam.widen.net/s/](https://nvdam.widen.net/s/nb5zzzsjdf/hpc-datasheet-sc23-h200-datasheet-3002446) [nb5zzzsjdf/hpc-datasheet-sc23-h200-datasheet-3002446](https://nvdam.widen.net/s/nb5zzzsjdf/hpc-datasheet-sc23-h200-datasheet-3002446)
- <span id="page-12-24"></span>[51] NVIDIA. 2025. NVIDIA Blackwell Architecture Technical Brief. [https://resources.](https://resources.nvidia.com/en-us-blackwell-architecture) [nvidia.com/en-us-blackwell-architecture](https://resources.nvidia.com/en-us-blackwell-architecture)
- <span id="page-12-2"></span>[52] OpenAI. 2025. Introducing ChatGPT Agent: Bridging Research and Action. <https://openai.com/index/introducing-chatgpt-agent/>
- <span id="page-12-3"></span>[53] OpenAI. 2025. Introducing Deep Research. [https://openai.com/index/](https://openai.com/index/introducing-deep-research/) [introducing-deep-research/](https://openai.com/index/introducing-deep-research/)
- <span id="page-12-11"></span>[54] Varatheepan Paramanayakam, Andreas Karatzas, Iraklis Anagnostopoulos, and Dimitrios Stamoulis. 2025. Less is More: Optimizing Function Calling for LLM Execution on Edge Devices. In Proceedings of the Design, Automation and Test in Europe Conference (DATE).
- <span id="page-12-14"></span>[55] Yeonhong Park, Jake Hyun, Hojoon Kim, and Jae W. Lee. 2025. DecDEC: A Systems Approach to Advancing Low-bit LLM Quantization. In Proceedings of the USENIX Symposium on Operating Systems Design and Implementation (OSDI).
- <span id="page-12-19"></span>[56] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini. 2024. Splitwise: Efficient Generative LLM Inference Using Phase Splitting. In Proceedings of the International Symposium on Computer Architecture (ISCA).
- <span id="page-12-5"></span>[57] Qualcomm. 2024. [https://www.qualcomm.com/products/mobile/snapdragon/](https://www.qualcomm.com/products/mobile/snapdragon/laptops-and-tablets/snapdragon-x-elite) [laptops-and-tablets/snapdragon-x-elite](https://www.qualcomm.com/products/mobile/snapdragon/laptops-and-tablets/snapdragon-x-elite)
- <span id="page-12-6"></span>[58] Qualcomm. 2024. Hexagon NPU SDK. [https://www.qualcomm.com/developer/](https://www.qualcomm.com/developer/software/hexagon-npu-sdk) [software/hexagon-npu-sdk](https://www.qualcomm.com/developer/software/hexagon-npu-sdk)
- <span id="page-12-12"></span>[59] Samsung. 2017.<https://www.samsung.com/us/apps/bixby>
- <span id="page-12-15"></span>[60] Rishov Sarkar, Hanxue Liang, Zhiwen Fan, Zhangyang Wang, and Cong Hao. 2023. Edge-MoE: Memory-efficient Multi-task Vision Transformer Architecture with Task-level Sparsity via Mixture-of-Experts. In Proceedings of the International Conference on Computer-Aided Design.
- <span id="page-12-9"></span>[61] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language Models Can Teach Themselves to Use Tools. In Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS).
- <span id="page-12-31"></span>[62] Seong Hoon Seo, Junghoon Kim, Donghyun Lee, Seonah Yoo, Seokwon Moon, Yeonhong Park, and Jae W. Lee. 2025. FACIL: Flexible DRAM Address Mapping for SoC-PIM Cooperative On-device LLM Inference. In Proceedings of the International Symposium on High-Performance Computer Architecture (HPCA).
- <span id="page-12-10"></span>[63] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2023. HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face. In Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS).
- <span id="page-12-16"></span>[64] Zheyu Shen, Yexiao He, Ziyao Wang, Yuning Zhang, Guoheng Sun, Wanghao Ye, and Ang Li. 2025. EdgeLoRA: An Efficient Multi-tenant LLM Serving System on Edge Devices. In Proceedings of the International Conference on Mobile Systems,

- Applications, and Services.
- <span id="page-13-0"></span>[65] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. In Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS).
- <span id="page-13-3"></span>[66] Simranjit Singh, Andreas Karatzas, Michael Fore, Iraklis Anagnostopoulos, and Dimitrios Stamoulis. 2024. An LLM-tool Compiler for Fused Parallel Function Calling. In arxiv.org.
- <span id="page-13-13"></span>[67] Squeeze AI Lab. 2024. TinyAgent-7B. [https://huggingface.co/squeeze-ai-lab/](https://huggingface.co/squeeze-ai-lab/TinyAgent-7B) [TinyAgent-7B](https://huggingface.co/squeeze-ai-lab/TinyAgent-7B)
- <span id="page-13-12"></span>[68] Squeeze AI Lab. 2024. TinyAgent-dataset. [https://huggingface.co/datasets/](https://huggingface.co/datasets/squeeze-ai-lab/TinyAgent-dataset) [squeeze-ai-lab/TinyAgent-dataset](https://huggingface.co/datasets/squeeze-ai-lab/TinyAgent-dataset)
- <span id="page-13-8"></span>[69] Squeeze AI Lab. 2024. TinyAgent-ToolRAG. [https://huggingface.co/squeeze-ai](https://huggingface.co/squeeze-ai-lab/TinyAgent-ToolRAG)[lab/TinyAgent-ToolRAG](https://huggingface.co/squeeze-ai-lab/TinyAgent-ToolRAG)
- <span id="page-13-4"></span>[70] Chunlin Tian, Xinpeng Qin, Kahou Tam, Li Li, Zijian Wang, Yuanzhe Zhao, Minglei Zhang, and Chengzhong Xu. 2025. CLONE: Customizing LLMs for Efficient Latency-aware Inference at the Edge. In Proceedings of the USENIX Annual Technical Conference (ATC).
- <span id="page-13-15"></span>[71] Nadav Timor, Jonathan Mamou, Daniel Korat, Moshe Berchansky, Gaurav Jain, Oren Pereg, Moshe Wasserblat, and David Harel. 2025. Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies. In Proceedings of the International Conference on Machine Learning (ICML).
- <span id="page-13-7"></span>[72] Haoming Wang, Boyuan Yang, Xiangyu Yin, and Wei Gao. 2025. Never Start from Scratch: Expediting On-device LLM Personalization via Explainable Model Selection. In Proceedings of the International Conference on Mobile Systems, Applications, and Services.
- <span id="page-13-17"></span>[73] Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. 2024. Mobile-agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-agent Collaboration. In Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS).
- <span id="page-13-18"></span>[74] Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. 2024. Mobile-agent: Autonomous Multi-modal Mobile Device Agent with Visual Perception. In arxiv.org.
- <span id="page-13-14"></span>[75] WizardLM Team. 2024. WizardLM 2.<https://wizardlm.github.io/WizardLM2>

- <span id="page-13-10"></span>[76] Zheng Xu, Dehao Kong, Jiaxin Liu, Jinxi Li, Jingxiang Hou, Xu Dai, Chao Li, Shaojun Wei, Yang Hu, and Shouyi Yin. 2025. WSC-LLM: Efficient LLM Service and Architecture Co-exploration for Wafer-scale Chips. In Proceedings of the International Symposium on Computer Architecture (ISCA).
- <span id="page-13-19"></span>[77] Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and Junchen Jiang. 2025. CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. In Proceedings of the European Conference on Computer Systems (EuroSys).
- <span id="page-13-1"></span>[78] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. React: Synergizing Reasoning and Acting in Language Models. In Proceedings of the International Conference on Learning Representations (ICLR).
- <span id="page-13-5"></span>[79] Rongjie Yi, Liwei Guo, Shiyun Wei, Ao Zhou, Shangguang Wang, and Mengwei Xu. 2025. EdgeMoE: Empowering Sparse Large Language Models on Mobile Devices. IEEE Transactions on Mobile Computing (2025).
- <span id="page-13-6"></span>[80] Zhongkai Yu, Shengwen Liang, Tianyun Ma, Yunke Cai, Ziyuan Nan, Di Huang, Xinkai Song, Yifan Hao, Jie Zhang, Tian Zhi, Yongwei Zhao, Zidong Du, Xing Hu, Qi Guo, and Tianshi Chen. 2024. Cambricon-LLM: A Chiplet-based Hybrid Architecture for On-device Inference of 70B LLM. In Proceedings of the International Symposium on Microarchitecture (MICRO).
- <span id="page-13-16"></span>[81] Chi Zhang, Zhao Yang, Jiaxuan Liu, Yanda Li, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. 2025. AppAgent: Multimodal Agents as Smartphone Users. In Proceedings of the Conference on Human Factors in Computing Systems.
- <span id="page-13-20"></span>[82] Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra. 2024. Draft & Verify: Lossless Large Language Model Acceleration via Self-speculative decoding. In Proceedings of the ACL (Association for Computational Linguistics).
- <span id="page-13-11"></span>[83] Zeyu Zhang and Haiying Shen. 2024. FDC: Fast KV Dimensionality Compression for Efficient LLM Inference. In arxiv.org.
- <span id="page-13-9"></span>[84] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. 2024. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. In Proceedings of the USENIX Symposium on Operating Systems Design and Implementation (OSDI).
- <span id="page-13-2"></span>[85] Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. 2024. Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models. In Proceedings of the International Conference on Machine Learning (ICML).