# Business Operations

Insurance claims workflow automation

Customer care internal operations assistance

Human resources information retrieval and task assistance

#### Communications (U.S. and Latin America)

Automotive communication services

Communication automation services

#### Scientific Discovery

Biomedical sciences workflow automation

#### Software & Business Operations

Data analysis for enterprise

Enterprise cloud engineer and business assistance

Site reliability incident diagnoses and resolution

Software products question answering

#### Software DevOps

Spark version code and runtime migration

Software development life cycle assistance end-to-end

Data for the study combines public data with 12 in-depth case studies (Table [1\)](#page-1-2). Summary findings are presented in aggregate per confidentiality agreements with the sources. These 12 were selected based on originators' availability for interviews, applicationdiversity, and development status preferring those in production (total 8) or pre-production piloting (4). The cases spanned business

<span id="page-1-0"></span><sup>1</sup> [ttps://luma.com/x16vikh7.](ttps://luma.com/x16vikh7) Note: last accessed: 6 Oct. 2025.

<span id="page-1-1"></span><sup>2</sup>[https://rdi.berkeley.edu/events/agentic-ai-summit.](https://rdi.berkeley.edu/events/agentic-ai-summit) Note: last accessed: 6 Oct. 2025.

<span id="page-2-0"></span>Table 2: Case Counts by Source Company Characteristics

| Stage  |   | Continents |   | Countries |   | Case Use |   |
|--------|---|------------|---|-----------|---|----------|---|
| Mature | 6 | 1          | 5 | One       | 5 | External | 8 |
| Late   | 1 | 2 to 4     | 4 | Tens      | 6 | Internal | 4 |
| Growth | 1 | 5 to 6     | 3 | Hundreds  | 1 |          |   |
| Early  | 3 |            |   |           |   |          |   |
| Seed   | 1 |            |   |           |   |          |   |

operations (3), software development and operations (2), a tightly integrated combination thereof (4), scientific discovery (1), and enterprise communication services (2). They also differed in their intended use, 4 targeting internal software and business operations, and 8 targeting external (enterprise) consumers. Table [2](#page-2-0) additionally lists statistics on the spread of case sources by company stage, continent and country spread. The questionnaire response data – over 400 responses and growing – is used complementarily for broader validation.

### 4 Summary Findings

Agentic AI is being applied to human-facing and highly interdisciplinary problems. The tasks and correctness conditions lack formal specification [\[Stoica et al.](#page-4-15) [2024\]](#page-4-15). The performance or evaluation metrics and systems do not always match the ultimate goal or benefit of the system. Still, some things are working (in production). Others are not. The following unpacks these observations from industry case studies complemented with mass survey results and public data.

### 4.1 HIL and Fundamental Complexity

Across software development, enterprise and scientific discovery systems, we are observing Agentic AI being applied to tasks specified by humans. This was true of every single case in our study and 94.8% direct consumers of survey respondents' systems. The common goal of these systems is to reduce the human time necessary for completing scientific, software, legal, or business processes. Human interfaces designed to delight were a secondary requirement for productivity rather than the goal or primary requirement. An important exception however included cases governed by policies requiring human oversight of Agentic AI systems, such as the EU Artificial Intelligence Act. For these cases, the engineers designed with HIL in mind from the start. In general, AI coding agents are among if not the most advanced Agentic AI systems in production, but still exhibit a spectrum of HIL rather than absence of HIL. At one extreme, state-of-the-art (SoA) systems claim multiple hours of autonomous execution of delegated development tasks (e.g. up to 30 hrs [\[Anthropic 2025\]](#page-4-16)). However, human input is commonly still required for major actions such as PR approval [\[GitHub 2025\]](#page-4-17). On the other extreme, human approval of fine-grained code modifications (e.g. in-IDE auto-completion) is still ubiquitous. SoA IDEs even provide options for varying the level of Agentic AI autonomy [\[Cursor](#page-4-18) [Team 2025;](#page-4-18) [Deshmukh et al. 2025;](#page-4-19) [GitHub 2025\]](#page-4-17).

We have not encountered a production-track Agentic AI system implemented without human-in-the-loop (HIL), even with the scope narrowed to business, software engineering, and scientific applications. Figure [1](#page-3-0) introduces terminology and illustrates our

observations of how production-track systems split functionality between HIL and automated methods across the key runtime stages of task specification, solution generation, solution verification, and ongoing evaluation. The categories we focus on here to illuminate technical opportunities are HIL in verification and evaluation.

Applied Agentic AI raises the level of abstraction for the fundamental compute unit, but is not yet asking nor answering the questions complexity theory enabled computer scientists to answer to date. Classical complexity theory enabled asking fundamental questions, is a solution computable in the first place? In how much time (our lifetime)? Is there a difference between the generation versus verification time and space complexity? It enabled answering these questions prior to building impractical solutions, and it led to constructive solutions (approximations) even for hard problems. We do not have the equivalent scaffolding for Agentic AI systems.

