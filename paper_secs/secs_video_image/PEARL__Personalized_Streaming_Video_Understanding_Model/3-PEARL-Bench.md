# 3 PEARL-Bench

#### 3.1 Task Definition

In the task of Personalized Streaming Video Understanding, a streaming video is processed as a continuous sequence of scenes. Throughout the stream, a user can dynamically introduce new concepts at any timestamp via instructions, forming an evolving set of user-defined concepts. For a subsequent query, the model must retrieve the relevant concepts and visual context to generate an accurate response. Specifically, as illustrated in Fig. [2,](#page-5-0) we define two types of concepts:

1. Frame-level Concepts: Static entities registered from a single frame. For example, defining a specific person or object at any timestamp.

2. Video-level Concepts: Dynamic actions unfolding over a continuous clip. For instance, defining a personalized gesture or a series of special actions.

Based on their temporal and functional requirements, we also categorize the queries into three types:

- 1. **Concept-Definition QA:** Introduces new concepts at specific timestamps. The model registers the concept into memory based on the current scene.
- Real-Time QA: Queries established concepts at the immediate moment.
   The model grounds its response purely on the present scene, evaluating its proficiency in answering real-time questions without historical distraction.
- 3. Past-Time QA: Inquires about the historical states or activities of established concepts. The model must retrieve relevant historical sequences, requiring long-term temporal reasoning and precise evidence retrieval.

The task is inherently multi-turn, enabling flexible concept definitions and queries regarding established concepts at arbitrary future time steps. This interactive format lays the foundation for the next generation of persoanlized AI assistants.

#### 3.2 Benchmark Overview

Existing personalized benchmarks suffer from notable limitations and are largely disconnected from real-world scenarios, as shown in Table 1. MyVLM [26], Yo'LLaVA [8], MC-LLaVA [7], UnifyBench [12] and MMPB [15] are all image-based, supporting neither video input nor streaming scenarios, and lacking multiturn interaction. PVChat [10] and This-isMy [32] introduces video modality but is limited to short offline videos (each video is shorter than 5 seconds), with no support for streaming or multi-turn concept interaction. Moreover, none of the above benchmarks supports Video-level personalization, *i.e.*, recognizing personalized concepts defined by continuous actions unfolding across frames. PEARL-Bench is the first benchmark to simultaneously support long-form streaming video input, multi-turn concept interaction, and both Frame-level and Video-level personalized concept types. As shown in Table 2, PEARL-Bench comprises 132 videos and 2,173 annotations in total, with an average duration of 1,458 seconds per video. All annotations are associated with precise timestamps.

<span id="page-4-0"></span>Table 1: Comparison of PEARL-Bench with ex- Table 2: Data Statistics isting personalized benchmarks.

Of PEARL-Bench.

| Benchmark       | Modality      | Strooming | g Multi-turn | Concept Type |             | Multi-Concept  |
|-----------------|---------------|-----------|--------------|--------------|-------------|----------------|
| Dencimark       | Modanty       | Streaming |              | Frame-level  | Video-level | Withti-Concept |
| MyVLM [26]      | Image         | -         | Х            | /            | Х           | Х              |
| Yo'LLaVA [8]    | Image         | _         | ×            | /            | X           | X              |
| MC-LLaVA [7]    | Image         | -         | ×            | /            | ×           | /              |
| UnifyBench [12] | Image         | -         | ×            | /            | ×           | ×              |
| MMPB [15]       | Image         | -         | ×            | ✓            | ×           | ✓              |
| PVChat [10]     | Video (short) | Х         | Х            | /            | Х           | <b>✓</b>       |
| This-is-My [32] | Video (short) | Х         | ×            | /            | ×           | /              |
| PEARL-Bench     | Video (long)  | /         | /            | /            | /           | <b>√</b>       |

|                   | Frame-level | Video-level | Total |
|-------------------|-------------|-------------|-------|
| #Videos           | 112         | 20          | 132   |
| Avg. Duration (s) | 1,657       | 303         | 1,458 |
| #Concept-Def QA   | 418         | 80          | 498   |
| #Real-Time QA     | 922         | 359         | 1,281 |
| #Past-Time QA     | 394         | -           | 394   |
| #Total QA         | 1,734       | 439         | 2,173 |

