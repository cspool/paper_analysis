# Abstract

In the realm of AI, large language models (LLMs) like GPT-5, central to the operation of AI agents, predominantly operate in the cloud, incurring high operational costs. With local-based small language models (SLMs) becoming more accurate, the necessity of cloud-exclusive processing is being reconsidered. An AI agent's response to a user's request comprises a series of subtasks or iterations. Existing approaches choose either an LLM or SLM for the entire request to ensure similar outputs, but this is ineffective for AI agents as SLMs may generate differing subtasks, compromising final accuracy. In this paper, we first conduct experimental analysis to understand the features of AI agent operations. Leveraging our findings, we propose the Adaptive Iteration-level Model Selector(AIMS), a lightweightschedulerto automatically partition an AI agent's subtasks between local-based SLM and cloud-based LLM. AIMS considers the varying subtask features and strategically decides the location for each subtask in order to use SLM as much as possible while maintaining the accuracy level. Our experimental results demonstrate that AIMS achieves up to a 27.5% relative improvement in accuracy and up to 31.4% relative increase in SLM usage compared to HybridLLM. It offloads 83.4% of subtasks to a local SLM while attaining similar accuracy on average compared with the cloud-only LLM approach.

CCS Concepts: • Computing methodologies → Intelligent agents; Planning and scheduling; Machine learning; • Networks → Network algorithms.

Keywords: LLM systems, AI agents, hybrid cloud-edge computing, model routing, cost optimization, small language models, scheduling

#### ACM Reference Format:

Shiyi Liu, Haiying Shen, Shuai Che, Mahdi Ghandi, and Mingqin Li. 2026. AIMS: Cost-Efficient LLM-based Agent Deployment in Hybrid Cloud-Edge Environments. In 21st European Conference on Computer Systems (EUROSYS '26), April 27–30, 2026, Edinburgh, Scotland Uk. ACM, New York, NY, USA, 17 pages. https://doi.org/ 10.1145/3767295.3803622

<sup>∗</sup>This work was done when the author was at Microsoft.

![](_page_0_Picture_13.jpeg)

This work is licensed under a Creative Commons Attribution 4.0 International License.

EUROSYS '26, Edinburgh, Scotland, UK © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2212-7/26/04 https://doi.org/10.1145/3767295.3803622

