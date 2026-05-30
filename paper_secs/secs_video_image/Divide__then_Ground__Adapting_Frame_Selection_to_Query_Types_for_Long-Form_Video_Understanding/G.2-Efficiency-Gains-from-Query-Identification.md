# **G.2 Efficiency Gains from Query Identification**

To balance efficiency and accuracy, **DIG** employs a Query Identification module. We apply resource-intensive key frame selection only to localized queries, defaulting to efficient uniform sampling for global ones. This adaptive strategy minimizes computational cost without compromising downstream performance (see Section [6\)](#page-6-1). Table [11](#page-25-1) quantifies these gains by comparing our adaptive approach against the baseline that applies our specific selection universally. On LongVideoBench [\[55\]](#page-14-1), where queries are predominantly localized, the QI module incurs a marginal overhead (3*.*6%) due to the additional classification step. However, on datasets with a diverse mix of query types, such as VideoMME [\[56\]](#page-14-2) and MLVU [\[54\]](#page-14-0), the adaptive strategy yields significant time savings (19*.*9% and 13*.*3%, respectively). This demonstrates that the QI module effectively optimizes resource allocation by bypassing unnecessary computation for global queries.

<span id="page-25-1"></span>**Table 11:** *Impact of Query Identification on efficiency. We compare the frame selection time (in minutes) of applying our specific selection universally (w/o. QI) versus DIG's adaptive approach (w. QI). "Percent" denotes the proportion of localized queries.*

|                     | Percent | w/o QI | w/ QI                 |
|---------------------|---------|--------|-----------------------|
| MLVU [54]           | 82.8    | 295.7  | (↓<br>256.3<br>13.3%) |
| LongVideoBench [55] | 97.8    | 134.1  | (↑<br>138.9<br>3.6%)  |
| VideoMME [56]       | 77.0    | 384.2  | (↓<br>307.9<br>19.9%) |

#### <span id="page-25-0"></span>Query Identification Prompt

You are a helpful assistant in a video-based question-answering process.

### **Core Task & Definitions**

You will classify the given query into one of two categories:

- 1. **Global Query (isGlobal: true):** The query requires going through and understanding the entire video content.
- 2. **Localized Query (isGlobal: false):** The query that can be fully answered by extracting and analyzing several specific segments within the video.

#### **Instructions for Analysis and Response**

In your analysis, please follow this structured reasoning process to classify the query:

- **Step 1. Understand the Query:** First, read the query to understand its general meaning and core intent.
- **Step 2. Infer Video Style (Hypothetically):** Based on the query's phrasing, make a reasonable inference about the style of the video (e.g., is it a narrative film, an educational lesson, a documentary, etc.)?
- **Step 3. Identify Referents:** Analyze if the query has specific referents. A referent is an entity (person, object), action, event, or even a specific piece of information, depending on the type of video you inferred. For instance, in 'What does Professor Smith write about quantum physics?', the referent is 'Professor Smith' and 'quantum physics' since the video style is likely a lesson.
- **Step 4. Evaluate Referents in Context:** Based on the results from step 3 and the criteria below, determine whether the query is Global or Localized.
  - (i) **The query is Global** if it meets either condition:
    - 1. Lacks a specific referent. The examples include: Summary-based: "primary focus," "in summary," "what is the video about?"
    - 2. Has a referent, but answering still requires a holistic understanding from going through the entire video. The examples include: "what is the boy's overall role?"
  - (ii) **The query is Localized** if it has specific referents, and the answer can be found by focusing on specific, related segments where it appears. Here are some examples:
    - Entity-based: "the person in the red shirt," "the black dog," "Professor Smith," "the little girl."
    - Action/Event-based: "what is [X] doing," "how does [X] build,"
    - Temporal/Sequential: "at the beginning," "after the explosion,"

Please provide your answer in the following format: {"analysis\_step1": str, "analysis\_step2": str, "analysis\_step3": str, "analysis\_step4": str, "isGlobal": true/false}

**User Query**: <Question>

**Figure 11:** *Query Identification Prompt. The LLM is first provided with the task definition, followed by an application of the chain-of-thought [\[77\]](#page-15-8) technique to arrive at a judgment.*

#### <span id="page-26-0"></span>Reward Assignment Prompt

You are a reward model for a video-based question-answering system.

### **Task**

You will receive a question and a sampled video frame. Your task is to evaluate the relevance of this frame for answering the question. Please assign a reward score that indicates how useful or informative the provided frame is in the context of the given question.

#### **Instructions for Analysis and Response**

In your analysis, please perform the following steps to finish your evaluation:

- 1. Describe the visual content of the sampled frame, focusing on elements relevant to the question, if such elements are present.
- 2. Assign a relevance reward between 0 and 100 based on: (1) The sampled frame's direct usefulness in answering the question (2) Whether the frame suggests that adjacent frames might provide additional information that help answer the question more effectively.

Please provide your answer in the following format: {"description": str, "reward": int}.

#### **User Input**

Video Duration: <Duration> seconds; Sampled Frame Timestamp: <Timestamp> seconds; Question: <Question>

**Figure 12:** *Reward Assignment Prompt. The LMM is first presented with the task definition and associated metadata. Then, the chain-of-thought reasoning technique [\[77\]](#page-15-8) is applied to assign the reward for the input frame.*

### <span id="page-26-1"></span>Inference Prompt

Question: What is the video mainly about?

- A. Planes invented by the Wright Brothers.
- B. The structural difference between the planes created by Whitehead and planes created by the Wright Brothers.
- C. Who invented the first plane.
- D. How Whitehead and the Wright Brothers cooperated to invent the first motorized flight.

Please select the best answer from the options provided and directly provide the letter representing your choice without giving any explanation.

**Figure 13:** *Prompt Template Example. Example of the prompt template used by LMMs to perform direct inference.*