# 1 Introduction

The landscape of natural language processing (NLP) has evolved with LLMs like GPT-5, showcasing unprecedented text generation capabilities [35, 57]. These models serve as the cognitive core in autonomous AI agents, revolutionizing traditional AI by integrating reasoning capability and tool use, thus enabling diverse interactions and generalization abilities [14, 25, 30, 47, 54, 56, 61, 62, 67].

The AI agents, such as the notable Auto-GPT [46] and AutoGen [59], have expanded AI's reach. They process a user's request by breaking it into subtasks and using the Internet search and other tools in an automatic process [24, 26, 67]. Each subtask represents a specific action or decision that the AI agent needs to perform to progress toward completing the user request. For example, when an AI agent receives a question "Who was the maternal grandfather of the person who directed the 1997 film Titanic?", it generates three subtasks "Identify the director of Titanic (1997)", "Find the director's mother", and "Determine the mother's father". The language model is invoked at each step to interpret the current state, execute the subtask, and generate the next subtask. This iterative process continues, with the agent potentially creating new subtasks or refining existing ones based on intermediate results, until it determines that the original request has been satisfactorily addressed. This process is reminiscent of advanced AI techniques like Chain of Thought (CoT) [58] and Reflexion [45], enhancing the agent's decision-making and problem-solving abilities. The effectiveness of these agents heavily depends on the language models' ability to accurately interpret and execute each subtask.

Despite their capabilities, these LLM-based AI agents incur high operational costs (\$0.01/1K prompt tokens for GPT-4o) from frequent cloud-based API queries, presenting significant economic challenges [32]. For example, incorporating ChatGPT for enterprise use is expected to pose a financial burden exceeding \$9,000 monthly on small businesses [10, 32]. If a company provides AI agent services using its own LLM APIs, it incurs high operational costs. For example, serving ChatGPT costs approximately \$694,444 per day in compute hardware costs [43].

![](_page_1_Figure_0.jpeg)

Figure 1. Dataflow of AIMS.

Companies can reduce the cost of providing AI agent services by leveraging users' edge devices (e.g., laptops) and incentivize users to adopt this system by offering lower service prices. Therefore, this work studies the computation breakdown for AI agents between the cloud and a user's local edge device. This approach is motivated by previous findings that an SLM can match the accuracy of the LLM in certain scenarios [12, 40, 63]. HybridLLM [8] employs a classifier to allocate a request to either SLM or LLM, ensuring that the SLM's output matches that of the LLM. We conducted experiments on AI agents based on different datasets and observed:

- (1) Using HybridLLM to allocate each subtask independently to SLM or LLM within an AI agent process may degrade the final output's accuracy, as SLM might produce a different subsequent subtask compared to LLM.
- (2) For certain requests and subtasks of a request, their simplicity allows for comparable accuracy when executed locally on a personal device instead of in the cloud.
- (3) Subtask position can influence the accuracy impact of SLM-LLM switching, with later stages showing greater effects.
- (4) SLM typically generates more subtasks for a later LLM subtask to gradually converge to a similar output.

Deploying LLM-based agents in a hybrid cloud–edge setting raises a systems question: how to reduce cloud inference cost by executing as much of the agent workflow as possible on an on-device SLM, without materially degrading end-toend task success. Unlike conventional request routing, an agent executes an interdependent sequence of subtasks; routing decisions early in the chain can change later subtasks, cascading into different computational paths and accuracy outcomes. This motivates position-aware routing, where the scheduler must consider both per-subtask characteristics and the subtask's stage in the agent's reasoning process.

To meet this need, based on the observations, we propose the Adaptive Iteration-level Model Selector (AIMS), a lightweight scheduler that maximizes the fraction of subtasks executed on the local SLM while retaining accuracy relative to a cloud LLM, using a similarity-based accuracy target learned from offline profiling. In the current design, AIMS does not take an explicit latency or monetary budget as an

online input; instead, we report latency empirically and show that scheduling overhead is small.

As shown in Figure 1, AIMS utilizes offline fine-tuned models to estimate the output similarity of a request or subtask on SLM and LLM to select the most suitable model. It takes the following steps: 1) if the outputs for the entire request are similar, SLM processes the entire request; 2) otherwise, for each of its subtasks, if its outputs are similar, SLM processes the subtask; 3) if the outputs differ, AIMS identifies a convergence point in the subtask series where the SLM and LLM outputs align and then continues using the SLM until that point; 4) If no convergence is detected, the subtask is decomposed to simpler subtasks and processed by the SLM only if all decomposed subtasks produce outputs similar to those of the LLM.

Our hybrid deployment model considers edge devices such as smart phone and gaming laptops, running 1-4B parameter models locally, while cloud resources provide access to frontier models (GPT-5, Claude) via API calls. The system must dynamically balance local compute utilization against network communication costs and cloud service charges.

AIMS differs from existing approaches like HybridLLM through its holistic treatment of AI agent subtasks. Rather than making isolated decisions, AIMS recognizes the interconnected nature of agent reasoning. The system employs position-aware decision-making, setting higher similarity thresholds for later subtasks to meet their increased accuracy requirements. Furthermore, AIMS identifies the convergence point to increase local processing opportunities. These novelties enable AIMS to more effectively balance cost and accuracy throughout the agent reasoning process.

Our contributions are as follows:

- Experimental analysis on subtask allocation in AI agents. We made several insightful observations from our experimental analysis. For example, unlike allocating a single request between the SLM and LLM that only needs to ensure the similarity of their outputs, allocating the subtasks for a request for AI agent has a unique challenge: if a subtask is allocated to SLM, the subsequent subtasks will vary, which affects the accuracy of the final output.
- Proposal of AIMS. Building on the observations from our experimental analysis, we propose AIMS. AIMS intelligently determines the allocation of subtasks between the local SLM and cloud-based LLM in order to use SLM as much as possible while maintaining the accuracy level.
- Comprehensive experiments of AIMS. Across nine datasets (HotpotQA, GSM8K, DROP, HumanEval, Web-Shop, MATH, WebArena, WorFBench, ToolBench), AIMS delivers a 27.5% relative improvement in output accuracy over HybridLLM on average (macro), with absolute gains of 12–17%, while increasing SLM usage

by up to 31.4% (relative) on average. In cost terms, for the Qwen3-4B+GPT-5 pair, AIMS reduces cloud LLM spend by 40.6% versus HybridLLM (macro) and by 83.4% versus an All-LLM policy. Using our tokenfootprint accounting and current API prices, this translates to savings of approximately \$3,820 (GPT-5) to \$7,450 (Claude Sonnet 4) per one million requests compared to All-LLM, while simultaneously improving accuracy over HybridLLM and Minions [31].

This adaptive model selection framework, which is the first work in the realm of autonomous AI agents to our knowledge, represents a possibility of more efficient, cost-effective AI agent applications.

