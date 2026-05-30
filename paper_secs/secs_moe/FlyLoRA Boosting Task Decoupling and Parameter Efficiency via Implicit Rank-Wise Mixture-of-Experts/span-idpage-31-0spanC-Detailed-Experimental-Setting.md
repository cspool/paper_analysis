# <span id="page-31-0"></span>C Detailed Experimental Setting

#### C.1 Datasets

To comprehensively evaluate the effectiveness of our proposed method across diverse domains and task types, we conduct extensive experiments on five carefully selected benchmarks. These datasets span critical capabilities including general knowledge reasoning, scientific understanding, mathematical problem solving, and code generation. Table 17 summarizes the key characteristics of each dataset, while detailed descriptions are provided below:

- MMLU [25] serves as a comprehensive benchmark for evaluating broad knowledge understanding and reasoning capabilities. It comprises multiple-choice questions spanning 57 distinct academic subjects, ranging from elementary mathematics and US history to computer science and professional law. The diversity of domains makes it particularly suitable for assessing model generalization across different knowledge types.
- ScienceQA [48] provides a multimodal framework for science question answering, with content derived from elementary and high school curricula aligned with California Common Core Content Standards. The questions originate from IXL Learning's expert-curated educational resources. Following the established practice in [37], we utilize only the textual components to focus on linguistic understanding.

<span id="page-32-1"></span>Table 15: **Performance Comparison of Different A Initialization Strategies.** Single-task and multi-task accuracy comparison on MMLU using Llama-3.1-8B.

| A Initialization Strategy | Before                       | $\Delta$ after merging |
|---------------------------|------------------------------|------------------------|
| Gaussian                  | 40.88±1.61                   | -2.02                  |
| Rademacher                | $40.42 \pm 0.23$             | -2.35                  |
| FJLT                      | $40.57 \pm 1.34$             | -2.50                  |
| Two-Phase                 | $40.76{\scriptstyle\pm1.04}$ | -4.86                  |

<span id="page-32-2"></span>Table 16: CKA and Corresponding Accuracy Drop ( $\Delta$ ) Between Single-Task Adapter and Merged Model. Evaluating using Llama-3.1-8B.

| Method                              | Task                  | MMLU          | ScienceQA      | GSM8K          | HumanEval      |
|-------------------------------------|-----------------------|---------------|----------------|----------------|----------------|
| $LoRA_{(r=8)}$                      | $^{\rm CKA}_{\Delta}$ | 0.78<br>-6.48 | 0.39<br>-60.34 | 0.58<br>-30.15 | 0.75<br>-13.04 |
| $\overline{\text{FlyLoRA}_{(k=8)}}$ | $^{\rm CKA}_{\Delta}$ | 0.85<br>-2.02 | 0.53<br>-43.05 | 0.71<br>-21.81 | 0.84<br>-4.27  |

- **GSM8K** [12] offers 8,500 high-quality grade school mathematics word problems that demand multi-step arithmetic reasoning. Each problem is accompanied by a detailed, step-by-step solution, making it ideal for evaluating logical reasoning and procedural accuracy in mathematical contexts.
- CodeAlpaca-20k [7] contains 20,022 synthetically generated instruction-response pairs specifically designed for code-related tasks. This dataset facilitates effective instruction tuning for programming applications by providing diverse coding prompts paired with corresponding solutions.
- **HumanEval** [9] consists of 164 hand-crafted Python programming problems developed to assess functional correctness in code generation. Crucially, these problems were manually created to prevent data contamination, ensuring they do not appear in the training corpora of existing code generation models.

<span id="page-32-3"></span>Table 17: **Details of MMLU, ScienceQA, GSM8K, CodeAlpaca and HumanEval Datasets.** We list the number of training and testing samples and task types for the following datasets used in our experiments.

| Dataset            | <b>Training Samples</b> | <b>Testing Samples</b> | Task Types       |
|--------------------|-------------------------|------------------------|------------------|
| MMLU [25]          | 99,842                  | 14,042                 | Multiple Choice  |
| ScienceQA [48]     | 12,726                  | 4,241                  | Multiple Choice  |
| GSM8K [12]         | 7,473                   | 1,319                  | Math Problems    |
| CodeAlpaca-20k [7] | 20,022                  | _                      | Code Instruction |
| HumanEval [9]      | _                       | 164                    | Code Generation  |

