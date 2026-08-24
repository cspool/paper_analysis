# **CCS** Concepts

• Networks  $\rightarrow$  Data center networks; Network management; Network monitoring; • Computing methodologies  $\rightarrow$  Multiagent systems.

#### Keywords

Large Language Models (LLMs), RAG, Network Planning

#### **ACM Reference Format:**

Zhaodong Wang\* Samuel Lin\* Guanqing Yan\* Soudeh Ghorbani\*† Minlan Yu‡ Jiawei Zhou§ Nathan Hu\* Lopa Baruah\* Sam Peters\* Srikanth Kamath\* Jerry Yang\* Ying Zhang\*. 2025. Intent-Driven Network Management with Multi-Agent LLMs: The Confucius Framework. In ACM SIG-COMM 2025 Conference (SIGCOMM '25), September 8–11, 2025, Coimbra, Portugal. ACM, New York, NY, USA, 16 pages. https://doi.org/10.1145/3718958. 3750537

#### 1 Introduction

Network management is a vital component of large-scale services' networks, playing a pivotal role in ensuring reliability, performance, and scalability across vast numbers of interconnected servers. Despite significant research efforts in traffic engineering [19, 44], network provisioning [16, 38, 46], and automated diagnosis [30, 52], enterprises still require substantial engineering resources to manage their networks effectively. Large Language Models (LLMs)

![](_page_0_Picture_13.jpeg)

This work is licensed under a Creative Commons Attribution 4.0 International License. SIGCOMM '25, Coimbra, Portugal

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1524-2/25/09 https://doi.org/10.1145/3718958.3750537 [12, 18, 47, 53] have emerged as a promising solution to enhance network management.

Network management in production environments involves intricate, multi-step tasks that require diverse tools and expertise to navigate complex solution spaces. For instance, diagnosing a failed service request demands a meticulous process of troubleshooting networking issues, analyzing routing paths, and narrowing down potential causes. Similarly, evaluating the impact of a topology expansion plan requires multiple steps, involving traffic forecast generation, topology augmentation, failure simulation, and result analysis. Given their complexity and domain-specific nature, simply relying on LLMs to handle them in a single step is not effective. Instead, a more nuanced approach is needed, one that incorporates domain expertise and iterative refinement.

To address these challenges, we introduce Confucius, a novel multi-agent LLM framework. Confucius decomposes intricate management tasks into smaller, structured subtasks, each of which can be executed using specialized domain-specific tools and databases. Confucius introduces three key components that effectively incorporate domain-specific knowledge into the general multi-agent LLM framework:

Enhancing planning with structured network procedures: Confucius integrates existing structured network procedures, such as codified Methods of Processes (MOPs) or workflows [51], with LLM reasoning. These programs are often written in Domain-Specific Languages (DSLs), which use predefined functions to encode smaller operations. Confucius introduces programming primitives that bridge the gap between human-friendly structured data and foundational models. This integration aids the LLM in breaking down complex network management tasks into multiple smaller tasks. It also combines multiple agents' outputs to achieve better planning outcomes.

Connecting tools with Domain-Specific Languages: Our key idea is to leverage the numerous existing network management tools, rather than developing new ones. However, utilizing these tools effectively requires deep domain expertise in formatting the right input and commands. We propose a set of primitives that convert human-friendly instructions into DSL-compliant inputs for each tool. Based on our experience, we have identified three widely used DSLs in network management: topology graph, network time series data, and network data model [46]. Confucius has built-in modules that provide translation to these three DSLs, enabling seamless interaction between the Confucius agent and many existing network management tools.

Enhancing long-term and short-term memory with domainspecific retrievals: Confucius develops advanced memory management mechanisms to effectively handle conversation context, utilizing a hierarchical tree structure for short-term memory. For cases

<span id="page-1-0"></span>

| Category                  | Use Cases                           | Example Requests to LLM                                                                                                                                                         |  |  |
|---------------------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|--|
| Network<br>Design         | Topology<br>Design                  | Update max capacity for all fibers in NA to X<br>Double the num of FSWs in each POD.                                                                                            |  |  |
|                           | Configuration Generation            | Generate configuration to remove a BGP peer for switch X.                                                                                                                       |  |  |
|                           | Understanding Capacity<br>Situation | What is the total deployed power per rack type in one DC?<br>What is the EBB planned capacity in six months in region A?                                                        |  |  |
|                           | Capacity What-if<br>Analysis        | Update layer 3 router's max capacity to a value from "node_cap_manager", use "hose" demand type with p90 percentile run "cap_planner", compare result with latest gold version. |  |  |
| Network<br>Operations     | Create Operations<br>Workflows      | Write a workflow that upgrades switch role X software version.  Which building block can I use to create a new device in FBNet?                                                 |  |  |
|                           | Generate Migration Command          | Generate soft drain command for switch X.                                                                                                                                       |  |  |
|                           | Monitor Operations Status           | Show operations that took the longest to complete in last 3 days.                                                                                                               |  |  |
| Monitoring<br>& Diagnosis | Monitoring Network Health           | Can you show me gold traffic from region A to B on May 20 for Ads storage service?                                                                                              |  |  |
|                           | Anomaly Detection                   | How many distinct source IPs were observed in 1.1.1.1 on Mar 8th?                                                                                                               |  |  |
| Knowledge Sharing         |                                     | Where can I find data about production network performance?                                                                                                                     |  |  |

Table 1: Use Case Examples.

requiring a large volume of context, Confucius employs Retrieval-Augmented Generation (RAG) [11, 27, 31, 32] as a form of long-term memory. For instance, Confucius leverages RAG in a separate database to index hundreds of thousands of network data models, enabling efficient searching and retrieval by LLMs. Furthermore, it allows developers to configure the level of detail to store in memory and extract related information based on specific queries.

Ensuring correctness systematically: To guarantee the safety and reliability of management tasks, Confucius is designed to tightly integrate with existing verification and validation systems. Additionally, Confucius provides a set of primitives that facilitate frequent human feedback. Moreover, Confucius includes a benchmarking system that enables developers to easily evaluate their application under different configurations, prompting algorithms, and foundation models.

Confucius has been successfully deployed in production for two years, serving thousands of users and supporting over 60 network management applications. Notably, Confucius has resulted in significant time savings for developers, reducing the average development time by 17 engineer-hours per week, while maintaining high accuracy. Our evaluation demonstrates that Confucius improves accuracy by up to 21% compared to solutions that rely solely on foundation models. We share our experience developing Confucius and onboarding applications, providing valuable insights into the challenges and opportunities of using LLMs for production networks. To the best of our knowledge, this paper presents a pioneering comprehensive framework for developing and deploying LLM-assisted network management applications. We hope that this paper inspires future research in this exciting new domain, driving innovation towards truly intent-based network management.

#### 2 Motivation

We have successfully developed and deployed multiple applications using Confucius over the past two years. This section provides an overview of our production use cases, highlighting key challenges encountered. We illustrate these challenges with two examples, discuss the benefits of LLMs, and outline adoption challenges.

#### 2.1 Network Management Use Cases

Network management involves complex tasks that require manual steps and deep domain knowledge. Table 1 shows the categories of network management apps supported by Confucius, along with example queries. We broadly categorize the use cases into four categories of the life cycle of network management.

Network Design involves generating designs of topology that meet capacity and performance requirements. This task requires balancing optimal decisions against evolving technology, complex requirements, and limited resources. While traditional approaches create abstract diagrams that are manually translated into concrete data models [46], LLMs can assist in automatically converting abstract designs to concrete data models, reducing both time and errors. Among different choices of models and network products, LLMs can assist the selection based on high-level intent and requirement. In §2.2, we provide more details into capacity planning as an example of network design.

Network Operations involve executing tasks like configuration updates, software installations, and hardware replacements, following established MOPs. These tasks are critical to network reliability and performance, but manual execution can be time-consuming and error-prone. LLMs can improve these processes by suggesting existing MOPs, generating new ones, executing complex instructions, monitoring operation status, and lowering the barrier for in-house tools. LLMs can transform new product introduction and deployment into a more efficient process.

Network Monitoring involves collecting data from various vendors' APIs, but managing these APIs can be challenging due to vendor heterogeneity. Even with standardized APIs like Thrift [16], navigating complex structures and parsing retrieved data can be difficult. LLMs offer a promising solution by suggesting and writing APIs, as well as automatically parsing retrieved data. LLMs can not only significantly improve efficiency but also develop new troubleshooting processes through self-learning. We provide a detailed example of fault diagnosis in §2.3.

Knowledge Base and Onboarding involve network domain-specific terminologies and tools, which can overwhelm new engineers with complex concepts like EBB [19] and FA [9]. Network data requires specific context for proper usage. LLMs can provide this context by serving as a knowledge management platform.

