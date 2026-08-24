# <span id="page-17-0"></span>**A.5 Prompts**

Table [10](#page-17-3) shows stages to construct CLIPPER, mapped to their corrresponding prompts.

| Prompt                                  | Figure           |
|-----------------------------------------|------------------|
| Chapter outline generation              | 3                |
| Book summary generation                 | 4                |
| Chapter-level claim extraction          | 5                |
| Book-level claim extraction             | 6                |
| Claim deduplication                     | 7                |
| Claim verification                      | 8, 9, 10, 11, 12 |
| Chapter-level claim extraction (NA¨IVE) | 15               |
| Book-level claim extraction (NA¨IVE)    | 14               |

<span id="page-17-3"></span>Table 10: Figure references for each prompt.

## **A.6 Using DeepSeek-Distill to measure CoT groundedness**

We evaluate the model on 66 annotated claims from [§2.3](#page-3-0) and measure its agreement with human annotations (Table [11\)](#page-19-1). Among the models tested, DeepSeek-Distill aligns most closely with human judgments, with only one instance of disagreement, outperforming other models like GPT-4o (10 disagreements) and LLaMA-3.1-70B-Instruct (3 disagreements). Although Llama-70B performs comparably, it fails to provide clear explanations for its decisions and instead generating generic reasoning messages that lack specificity to samples. Therefore, we use DeepSeek-Distill to measure CoT groundedness in our dataset.

