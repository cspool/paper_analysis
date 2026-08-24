# 1 Introduction

Since generative AI began unlocking advanced capabilities for AI Agents and general Compound AI Systems [Zaharia et al. 2024], building Agentic AI systems has become the focus of hundreds of startups [CBInsights 2025], thousands of academic publications, and truly massive open online courses [Song and Chen 2024; Song et al. 2025]. Academic and industry researchers have prototyped new capabilities for everything from multi-agent tutoring [Schmucker

et al. 2024], to physical asset management [Timms et al. 2024], to self-driving science and engineering [Juraj Gottweis 2025; Novikov et al. 2025]. However, whether and to what extent AI Agents are practically useful is an open debate [Challapally et al. 2025]. Many capabilities demonstrated in academic spheres have yet to make an appearance in production systems. Problems solved and unsolved in industry for securing, scaling, and deploying production Agentic AI systems remain largely unpublished.

A few recent publications with similar motivation, offer lengthy analyses without broader concrete use-case analysis [Krishnan 2025; Liu et al. 2025] or limit scope to evaluation systems [Yehudai et al. 2025] or usability [Shome et al. 2025]. We take the position that reliable, deployable Agentic AI systems, like their wider family of Compound AI Systems [Zaharia et al. 2024], require not only demonstration of AI capability but also robust supporting systems integration. Further, that the most impactful system innovation will follow from analyzing the application use-cases first.

Hence, this study targets industry case studies, seeking to understand potential impact and progress (or lack thereof) across use-cases and domains, focusing on the relatively recent onset of foundation-model-infused AI agents. In light of this focus and industry cases incorporating multi-modal, hybrid, and foundation models broadly, we do not restrict the study to purely LLM-based agents for which there are numerous surveys of academic progress [Guo et al. 2024; Wang et al. 2024]. For case study selection, we prioritize production-grade cases self-described as "AI agent(s)" or "Agentic AI system(s)", rather than restricting selection according to one of the many (academic) definitions of "AI agent". We re-derive from cross-comparison what is considered successful (or unsuccessful) –useful – Agentic AI in practice.

Our central questions are, what is actually working (and not) in practice from a computational systems perspective? Why, what is making Agentic AI *useful* or not and to what extent? Is driving down inference latency the only or most important contribution systems researchers can make? Where can academic and industry research communities—and systems and AI research communities—better align to realize the potential of Agentic AI in practice?

In contrast to academic literature, we find deployed (useful) Agentic AI systems commonly focus on relatively well-defined, simple tasks repeatedly executed by human customers or employees. These include information retrieval, static and semi-static workflow execution across application domains. Successful cases reduce time-to-completion (increase operational throughput), and/or lower knowledge and skill requirements for completing tasks involving multiple system interfaces and domain-specific knowledge and

procedures. Reduction in time-to-completion is commonly measured relative to human time to complete the same task, yielding a wide spectrum of use-case-derived latency requirements. Hence, the range of contribution opportunities for the systems community includes exploiting relaxed latency constraints in scheduling and resource management, and is not limited to single-inference latency minimization. Further, we find useful Agentic AI systems rely on a combination of human and machine verifiers, with the most sophisticated (autonomous) systems supported by strong verifiers from mathematical, computer and computational sciences. The lack of strong verifiers appears to contribute more to the lag between industry and academia than a lack of AI capabilities. Revisiting complexity analysis and systems for verification is a critical and timely contribution area for interdisciplinary and industry-academia collaboration.

### 2 Methodology

We found openly available case study data, such as published literature, blog posts, open-source code repositories, present an incomplete picture of the landscape for systems research and development. To ground and expand openly available data, we engaged industry participants through interviews and questionnaires.

Interviewers were selected to maintain organizational neutrality, assigned roles, and completed a series of pre-, post-, and ininterview procedures. The structure of interviews was determined by a preset list of 11 topic groups (below), and the availability of respective answers from open sources that need only be verified via interview.

- (1) The root problem (benefit) the system is addressing (providing): What is the ultimate benefit? What is the system replacing and why?
- (2) Key success metrics and evaluation mechanism: What tools, techniques, systems, etc. are used to ensure the system meets user and stakeholder objectives? Is data corresponding to the expected or past system behavior available for the evaluation?
- (3) Key aspects of the system design and implementation: What programming framework was used? What is the general architecture? What are the steps, stages, and cycles? How are common components (e.g. routers, LLM-as-a-Judge, other verifiers, HIL) combined and why? What is the ratio of automation to human interaction and why—by design or limitation?
- (4) The state of the system or its development: Is the system in production, or was it never meant for production (purely for AI research, learning, upskilling)? Was the system prototyped for production but abandoned—why, and what were the critical limitations?
- (5) Known constraints or requirements of end-users and stakeholders: What are the security, regulatory, SLO/SLA requirements?
- (6) Advantage of an agentic AI system solution over alternative approaches: what is the advantage in your view?
- (7) System dependencies and complexity: what is the quantity, quality, and availability of tools, verifiers, data, etc.?

- (8) End-user quantity, expertise levels, and organizational domains.
- (9) Estimated cost versus value or benefit. Including sunk and ongoing costs of developing and operating the system versus its estimated value.
- (10) System stakeholders: Who ultimately benefits from deployment? Who is impacted by safety, security, etc. failures and limitations?
- (11) Your role and activities: What is your involvement with the agentic AI system(s) you are describing?

For breadth, understanding how far-reaching cross-case-study observations were, we iteratively crafted a questionnaire for mass distribution. The questionnaire was distributed to technical groups across the AI Alliance Agents-in-Production Meetup [1](#page-1-0) , the Berkeley RDI Agentic AI Summit [2](#page-1-1) , and collaborators' professional networks. Our questionnaire design informed by interviews, seeks to avoid response priming, facilitate downstream quantitative analysis, and facilitate broad participation by restricting the length, terminology, and the technical-depth and disclosure-depth necessary to complete the questionnaire. Further, we agreed upfront to aggregate and anonymize all data. As an aside, interviews revealed that what is considered confidentially-innovative in the space of Agentic AI varied significantly across organizations. In summary, our study integrates openly available data with perspectives from industry practitioners to answer, what is useful Agentic AI?

### 3 Use-Case Data

<span id="page-1-2"></span>Table 1: Anonymized in-depth case study descriptions.

