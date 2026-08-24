# <span id="page-4-0"></span>Example Search Procedure for Countdown.

#### [INSTRUCTION]

We will follow this search process:

- At each state, first choose two numbers from the number set. - Next, try the four operations (+, −, ×, and /) to obtain the new number and add the new number to the number set.
- Continue this process until we reach the target number.

#### [EXAMPLE PROBLEM] Numbers: [40, 19, 23, 7]

Target: 29

#### [EXAMPLE PROCEDURE]

Current number set: [40, 19, 23, 7]

- –|- Pick two numbers (40, 19) (numbers left: [23, 7])
- —-|- Try 40+19=59. Current number set: [59, 23, 7]
- ——|- Pick two numbers (59, 23) (numbers left: [7])
- ——–|- Try 59+23=82. Current number set: [82, 7]
- ———-|- Try 82+7=89. Evaluate 89!=29. Drop this branch.
- ———-|- Try 82-7=75. Evaluate 75!=29. Drop this branch.
- ———-|- Try 82\*7=574. Evaluate 574!=29. Drop this branch. ———-|- Try 82/7=11.7. Evaluate 11.7!=29. Drop this branch
- ——–|- Try 59-23=36. Current number set: [36, 7].
- ———-|- Try 36+7=43. Evaluate 43!=29. Drop this branch.
- ———-|- Try 36-7=29. Evaluate 29==29. Target found!

[SOLUTION]

40+19=59, 59-23=36, 36-7=29

Figure 2: Illustration of search procedure for Countdown (simplified). We provide both detailed instruction and example solving trace in our prompts to LCLMs.

Here, a state in the search procedure represents a partial travel plan up to a date. LCLMs

need to check various constraints and explore feasible arrangements at each step. The entry  $y_i$  is also a filled template recording the state computation for each scheduling decision.

### <span id="page-5-0"></span>3.2 Task Difficulty and Diverse Challenges

Three difficulty levels. To evaluate models with different capabilities, we construct three difficulty levels in LONGPROC by selecting subsets of data points that require approximately 500, 2K, and 8K output tokens respectively (counted using Llama-3's tokenizer). Please refer to Appendix B for more details on how we obtain data points of different output lengths. Pseudocode to Code omits the 8K token set due to limited program lengths in the source SPoC dataset; Travel Planning excludes the 0.5K token set as even the simplest data points require more output tokens. Table 2 provides statistics and comparisons across tasks.

**Diverse challenges.** We also highlight that the six tasks in LONGPROC, while sharing a common procedural generation framework, exhibit diverse challenges. The right side of Table 2 characterizes their key differences across three aspects:

- Accessing Information: Tasks vary in how they access and process context information. Some tasks, such as HTML to TSV and theory-of-mind tracking, require sequential processing of information in the input. Others, like Path Traversal and Travel Planning, need targeted retrieval of specific information at each step (e.g., identifying available out-going connections in Path Traversal).
- **Deductive Reasoning:** Tasks differ in the reasoning process required for executing the procedure. Theory-of-mind Tracking, Countdown, and Travel Planning involve different deductive reasoning capabilities. For instance, Countdown tests arithmetic computation, while Travel Planning needs compatibility checks between various scheduling constraints.
- Executing Search: Countdown and Travel Planning implement search-based procedures. In these cases, LCLMs are required to explore multiple possible actions at a step, and the solving process involves backtracking. This creates more complex execution traces that must record exploration paths and backtracking decisions.

Because all these tasks can be solved through pre-defined procedures that are already provided to LCLMs in prompts (Figure 2), their difficulty emerges from the extended generation requirements, where LCLMs must utilize information and preserve logical consistency across long distances. Therefore, our task design allows LONGPROC to evaluate multiple capabilities that are **central to long-context and long-form generation**.

### <span id="page-5-1"></span>3.3 Reliable Evaluation

Being able to reliably evaluate long outputs is a key feature of LONGPROC. Unlike existing benchmarks that rely on n-gram overlap metrics (Zhang et al., 2024; Shaham et al., 2023; Stelmakh et al., 2022; Fan et al., 2019) or LLM-based evaluation (Malaviya et al., 2024; Bai et al., 2024), LONGPROC enables reliable rule-based evaluation since its outputs are typically structured and deterministic. We evaluate task outputs as follows (see Appendix C for additional details):

- **HTML to TSV:** We compute row-level F1 scores over the model output rows  $\mathbf{Y} = y_1, y_2, ...$  and ground truth rows  $\mathbf{Y}^* = y_1^*, y_2^*, ...$
- **Pseudocode to Code:** Following Kulal et al. (2019), we evaluate translated functions using the unit tests. A function is correct if and only if it passes all test cases. This execution-based evaluation accommodates minor variations in implementation (e.g., using printf or cout).
- Path Traversal and ToM Tracking: These tasks require complete traces in a predefined format for deterministic processes. We use exact match evaluation  $Y = Y^*$ .
- Countdown and Travel Planning: For these search-based tasks, we evaluate the final solutions using rule-based validators. For Countdown, we verify calculation correctness of equations and whether we achieve the target. For Travel Planning, we verify satisfaction of all specified constraints.

<span id="page-6-1"></span>

|                                         | Context Size          | Average Scores |           | ores   |
|-----------------------------------------|-----------------------|----------------|-----------|--------|
|                                         |                       | 0.5K           | 2K        | 8K     |
| Open-we                                 | eight models with les | s than 15      | 5B parai  | neters |
| Llama-3.2-1B-Inst                       | 128K                  | 4.0            | 0.1       | 0.0    |
| Llama-3.2-3B-Inst                       | 128K                  | 13.5           | 4.5       | 0.1    |
| Llama-3.1-8B-Inst                       | 128K                  | 29.7           | 20.7      | 5.3    |
| Llama-3-8B-ProLong                      | 128K                  | 20.1           | 9.0       | 4.4    |
| Mistral-7B-Inst-v0.3                    | 32K                   | 18.6           | 12.4      | 1.2    |
| Phi3-7B-128k-Inst                       | 128K                  | 19.6           | 11.5      | 1.1    |
| Phi3-14B-128k-Inst                      | 128K                  | 25.4           | 12.2      | 2.5    |
| Qwen2.5-3B-Inst                         | 128K                  | 27.3           | 5.7       | 1.3    |
| Qwen2.5-7B-Inst                         | 128K                  | 27.8           | 23.0      | 3.8    |
| R1-Distill-Qwen2.5-7B $^{\mathcal{R}}$  | 128K                  | 16.3           | 7.7       | 2.4    |
| R1-Distill-Llama-3-8B $^{\mathcal{R}}$  | 128K                  | 51.6           | 24.6      | 7.5    |
| Ор                                      | en-weight models w    | ith 15-75      | B paran   | ıeters |
| AI21-Jamba-1.5-Mini (52B)               | 128K                  | 19.0           | 9.0       | 1.1    |
| Qwen2.5-32B-Inst                        | 128K                  | 68.4           | 50.3      | 17.1   |
| Qwen2.5-72B-Inst                        | 128K                  | 68.7           | 46.4      | 19.5   |
| Llama-3.1-70B-Inst                      | 128K                  | 72.9           | 58.0      | 24.2   |
| Llama-3.3-70B-Inst                      | 128K                  | 77.6           | 57.5      | 24.9   |
| R1-Distill-Qwen2.5-32B $^{\mathcal{R}}$ | 128K                  | 80.6           | 60.9      | 22.4   |
| R1-Distill-Llama-3-70B $^{\mathcal{R}}$ | 128K                  | 83.7           | 70.4      | 33.3   |
|                                         |                       | Propi          | rietary n | nodels |
| Claude-3-5-sonnet-2410                  | 200K                  | 78.4           | 57.5      | 22.0   |
| GPT-4o-mini-24-07                       | 128K                  | 55.7           | 38.1      | 7.6    |
| GPT-4o-2024-08                          | 128K                  | 94.8           | 83.4      | 38.1   |
| Gemini-1.5-flash-001                    | 1,000K                | 78.9           | 52.3      | 15.3   |
| Gemini-1.5-pro-001                      | 2,000K                | 89.2           | 79.4      | 54.0   |

Table 3: Average performance across tasks of different LCLMs on LONGPROC at three difficulty levels (0.5K, 2K, 8K). The reasoning models are labeled as  $\mathcal{R}$ . All models show performance degradation with increased output length. Even frontier models struggle with 8K-token procedural generation tasks.

We note that the output format requirements in LONGPROC do not constrain model performance. Our experiments show top models (e.g., GPT-40, Gemini-1.5-Pro) achieve near-perfect performance on the easiest set of these tasks, consistently maintaining proper formatting (§ 4). Also, structured output generation (e.g., JSON, TSV) is an important capability for practical applications. We believe a truly capable model should be able to comply with user-specified output formats.

