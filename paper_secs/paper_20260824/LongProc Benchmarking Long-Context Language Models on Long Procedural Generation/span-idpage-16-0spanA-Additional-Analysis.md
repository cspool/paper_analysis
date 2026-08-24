# <span id="page-16-0"></span>**A Additional Analysis**

### **A.1 Comparison with Low Dispersion Tasks**

We compare models' performance on LONGPROC against their performance on RULER [\(Hsieh et al.,](#page-11-5) [2024\)](#page-11-5), a commonly used long-context recall benchmark with relatively low dispersion. Table [4](#page-16-2) shows the comparison on LONGPROC and RULER. We mark the total lengths (input plus output) under each task and include one model from each of the three model groups.

| The   | results | suggest | substantial                          | perfor |
|-------|---------|---------|--------------------------------------|--------|
| mance | gaps    | between | high-dispersion                      |        |
|       |         |         | tasks in LONGPROC and low-dispersion |        |

<span id="page-16-2"></span>

|               | HTML<br>42K | 17K  | Path ToM<br>12K | 64K  | RULER<br>128K |
|---------------|-------------|------|-----------------|------|---------------|
| Llama-3.1-8B  | 23.4        | 0.0  | 0.0             | 89.8 | 81.3          |
| Llama-3.1-70B | 47.0        | 0.0  | 0.0             | 90.5 | 75.8          |
| GPT-4o-24-08  | 75.4        | 34.0 | 0.0             | 95.6 | 91.3          |

Table 4: Performance comparison between LONGPROC and RULER. We show **overall text lengths (input + output)** under the task.

**tasks in RULER**. While models across all scales (even those below 10B parameters) achieve strong performance on RULER at 64K tokens and some even maintain strong recall capabilities up to 128K tokens, none demonstrate robust generation capabilities on LONGPROC with much shorter overall lengths. This contrast highlights the challenge of processing diffused information, aligning with recent calls for developing truly challenging long-context benchmarks that incorporate high information dispersion [\(Goldman et al.,](#page-11-6) [2024\)](#page-11-6).

### <span id="page-16-1"></span>**A.2 Additional Details on the Comparison Between Reasoning Models and Instruct Models**

<span id="page-16-3"></span>

|                      | Countdown (2K)<br>acc term acc/term |    |     |    |    | Countdown (8K)<br>acc term acc/term |    |    | Travel Plan (2K)<br>acc term acc/term |    |    | Travel Plan (8K)<br>acc term acc/term |
|----------------------|-------------------------------------|----|-----|----|----|-------------------------------------|----|----|---------------------------------------|----|----|---------------------------------------|
| Qwen2.5-32B-Inst     | 87                                  | 88 | 99  | 55 | 78 | 71                                  | 95 | 99 | 96                                    | 5  | 82 | 6                                     |
| R1-Distill-Qwen-32B  | 88                                  | 88 | 100 | 51 | 52 | 98                                  | 54 | 56 | 96                                    | 22 | 36 | 61                                    |
| Llama-3.1-70B-Inst   | 89                                  | 91 | 98  | 61 | 68 | 90                                  | 86 | 94 | 92                                    | 12 | 60 | 20                                    |
| R1-Distill-Llama-70B | 86                                  | 86 | 100 | 47 | 51 | 92                                  | 71 | 77 | 92                                    | 35 | 64 | 55                                    |

Table 5: Comparison of accuracy (acc), termination rate (term; whether model complete generation and output EOS tokens within the allocated budget), and accuracy among terminated generations (acc/term). Reasoning models underperform compared to instruct models mainly because they fail to terminate within constraints, thus not providing final solutions.

In [§5,](#page-7-0) we compare the performance of reasoning models and instruct models across tasks. Recall that we observe reasoning models underperform their instruct counterparts on Travel Planning (2K) and Countdown (8K). This poor performance is due to the fact that reasoning models cannot finish generation and produce final solutions within a 16K output token budget, due to their use of a separate "thinking" stage. Note that instruct models are given a budget of 4K for Travel Planning (2K) and a budget of 10K for Countdown (8K), significantly lower than the budget of reasoning models (16K for all settings).

Table [5](#page-16-3) shows models' accuracy (acc), termination rate (term, which measures whether models complete generation within the allocated token budget), and accuracy among terminated generations (acc/term). On Countdown (8K) and Travel Planning (2K), where reasoning models underperform their instruct counterparts, reasoning models actually achieve the same or higher accuracy among terminated runs. However, their substantially lower termination rates lead to inferior overall accuracy. On Travel Planning (8K), reasoning models demonstrate much higher accuracy among terminated generations. Nevertheless, considering that reasoning models are allocated significantly more tokens yet achieve lower termination rates on Countdown (8K) and Travel Planning (2K), this reveals the tokeninefficiency of reasoning models for certain problems, opening an interesting research direction for future work.

