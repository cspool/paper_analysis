# <span id="page-17-2"></span>G Details of Reflective Words

To evaluate models' *Long CoT* reasoning capabilities, we focus on their use of specific *reflective words* that signal backtracking and verifying during the reasoning process. Following prior work [\[28\]](#page-10-14), we consider a curated list of 17 reflective words: ["*re-check*", "*re-evaluate*", "*re-examine*", "*re-think*", "*recheck*", "*reevaluate*", "*reexamine*", "*reevaluation*", "*rethink*", "*check again*", "*think again*", "*try again*", "*verify*", "*wait*", "*yet*", "*double-check*", "*double check*"]. We adopt two evaluation metrics:

<span id="page-18-1"></span>Table 6: Comparison between the GRPO baseline and ARM on GPQA and StrategyQA benchmarks.

| Models             |      | GPQA-Main |      | GPQA-Diamond | StrategyQA |        |  |
|--------------------|------|-----------|------|--------------|------------|--------|--|
|                    | Acc. | Tok.      | Acc. | Tok.         | Acc.       | Tok.   |  |
| Qwen2.5-7BSFT+GRPO | 35.0 | 2324      | 37.4 | 2604         | 72.9       | 646    |  |
| ARM-7B             | 34.8 | 1306      | 36.9 | 1536         | 73.8       | 229    |  |
| ∆                  | -0.2 | −43.8%    | -0.5 | −41.0%       | +0.9       | −64.6% |  |

Table 7: Definitions and results of reflection-related ratios on AIME'25.

<span id="page-18-2"></span>

| Ratio Name                        | Formula       | Qwen2.5-7BSFT+GRPO | ARM-7B |
|-----------------------------------|---------------|--------------------|--------|
| reflection_ratio                  | Nref<br>N     | 93.8               | 95.0   |
| correct_ratio_in_reflection_texts | Nref+<br>Nref | 14.2               | 13.9   |

reflection\_ratio, measuring the proportion of outputs containing at least one reflective word, and correct\_ratio\_in\_reflection\_texts, assessing the correctness within reflective outputs. The formulas for these metrics are summarized in Table [7,](#page-18-2) where N denotes the total number of responses, Nref the number of responses containing reflective words, and Nref<sup>+</sup> the number of correct reflective responses.

Given its competition-level difficulty, we conduct our analysis on AIME'25 using ARM-7B and Qwen2.5-7BSFT+GRPO. For ARM-7B, we use the Instruction-Guided Mode (Inst*Long CoT*) to specifically assess its *Long CoT* reasoning. The results, averaged over 8 runs, are reported in Table [7.](#page-18-2) As shown, both models exhibit a high frequency of reflective word usage, with reflection\_ratio exceeding 93%, indicating that reflection behavior is well-integrated during *Long CoT* reasoning. The correct\_ratio\_in\_reflection\_texts remains comparable for both models, and relatively low due to the high complexity of the AIME'25 tasks. These results demonstrate that Ada-GRPO does not hinder the model's *Long CoT* reasoning capabilities.

