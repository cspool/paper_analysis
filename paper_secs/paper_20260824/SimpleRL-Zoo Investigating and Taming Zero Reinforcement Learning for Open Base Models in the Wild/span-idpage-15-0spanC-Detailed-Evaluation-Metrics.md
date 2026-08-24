# <span id="page-15-0"></span>**C Detailed Evaluation Metrics**

**Reasoning Behavior Ratio:** To better understand the model's reasoning patterns throughout the training process, we adopt the cognitive behavior framework proposed by [Gandhi](#page-11-8) [et al.](#page-11-8) [\(2025\)](#page-11-8) and use GPT-4o [\(Hurst et al.,](#page-11-9) [2024\)](#page-11-9) to identify reasoning-related behaviors, including "Backtracking", "Verification", "Subgoal Setting", and "Enumeration". We report the ratio of responses that contain such cognitive behaviors. While some recent studies suggest tracking reflection behavior using related keywords [\(Yeo et al.,](#page-12-2) [2025;](#page-12-2) [Xie et al.,](#page-12-3) [2025\)](#page-12-3) as monitoring signals, we argue that these keywords only exhibit only a weak correlation with high-level reasoning patterns like reflection and verification. As a result, they fail to adequately capture the development of these reasoning processes. Further details can be found in Appendix [I.](#page-21-0)

**Clip Ratio:** In the early stages of training, the base model exhibits weak instructionfollowing ability and often fails to stop appropriately, resulting in irrelevant or excessively

```
Simple Prompt
Question:
{input}
Answer:
Let's think step by step.
Complex Prompt
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{input}
Please reason step by step, and put your final answer 
within \\boxed{}.<|im_end|>
<|im_start|>assistant
{output}
```

<span id="page-16-1"></span>Figure 10: Comparison between simple prompts and more complex prompts.

long outputs. After training collapses, the model may also generate repetitive or overly extended responses. Since the model has a fixed maximum context length, such outputs may be truncated during both training and evaluation. To monitor this issue, we define the proportion of truncated outputs as the "Clip Ratio".

**Average Stopped Length:** Generations that are truncated often result from issues such as repetitive patterns or incomplete reasoning, which typically do not contribute to effective trajectories. To account for this factor, we introduce a new metric to track the average length of responses that are stopped under normal conditions. It is a more reliable metric to consider only valid responses, thereby eliminating the interference caused by unstopped responses.

**Pass@k Accuracy:** We track the pass@k accuracy, which represents the percentage of questions for which at least one correct response is obtained when sampling k responses per question. Pass@k serves as an indicator of the model's exploration capabilities and is particularly relevant for RL, as it reflects the model's ability to generate responses that can achieve a positive reward. Previously, some researchers believed that RL training might merely reorder responses within the original model distribution, as evidenced by the lack of improvement in pass@k accuracy following RL training [\(Shao et al.,](#page-12-5) [2024\)](#page-12-5).

