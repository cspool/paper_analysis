# <span id="page-16-0"></span>D EVALUATION METRICS

Completeness Rate. We measure the proportion of cases where the dynamic visual context alone suffices to produce the correct answer. Concretely, after the multi-turn search, we re-answer the question using only the gathered dynamic frame set and disallow further search, following the CSV procedure in Sec. [2.2](#page-3-1) and the prompt illustrated in Fig. [11.](#page-16-1) The resulting correctness is computed with the same task-specific accuracy used elsewhere, averaged over the whole dataset.

Consistency Rate. Consistency evaluates whether the intermediate reasoning coherently supports the final answer under the given question. We prompt a LLM model (GPT-4o) with the question, the reasoning text extracted from <think>...</think>, and the final answer from <answer>...</answer>, using the format in Fig. [12](#page-16-3) that requires a structured output: a short analysis in <think> followed by <answer> equal to "Yes" or "No". In implementation, we parse the LLM's output to obtain the binary decision; "Yes" is counted as 1 and "No" as 0, and any parsing failure is treated as 0. The Consistency Rate is the dataset average of these binary outcomes.

```
Consistency Score Evaluation
<system prompt>
You are a careful and logical reviewer. Your task is to verify whether the given reasoning
 process and the final answer are consistent in addressing the given question.
Please carefully read the following information:
Question: <Question>
Reasoning Process: <Reasoning>
Final Answer: <Answer>
Please follow this format strictly:
<think> Your analysis here </think> <answer> Yes/No </answer>
```

<span id="page-16-3"></span>Figure 12: The template for calculating consistency.

