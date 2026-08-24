# A.6 Evaluation Dataset Detail

We provide an overview of all datasets used in the following sections.

- ASDiv: A diverse simple English math word problem corpus for evaluating the capability of various MWP solvers. It contains 2,305 MWPs that cover more text patterns and most problem types taught in elementary school.
- GSM8K: A high-quality benchmark comprising 8,500 human-written grade school math word problems that require multi-step reasoning and basic arithmetic, each labeled with a natural language solution and verified answer. The 1,319-question test set emphasizes sequential reasoning and is primarily solvable by upper-grade elementary school students.
- MATH500: a challenging benchmark of 500 high school competition-level problems spanning seven subjects, including Algebra, Geometry, Number Theory, and Precalculus. Each problem is presented in natural language with LaTeX-formatted notation, offering a strong measure of mathematical reasoning and generalization across diverse topics.
- AIME2024: a dataset containing 30 problems from the 2024 American Invitational Mathematics Examination (AIME), a prestigious high school mathematics competition for topperforming students. Each problem is designed to require deep mathematical insight, multi-step reasoning, and precise problem-solving skills.
- AMC: The AMC dataset consists of all 83 problems from AMC12 2022 and AMC12 2023, extracted from the AoPS wiki page. We used a subset of this data containing 40 problems.
- MinervaMath: MinervaMath is a high-difficulty math problem dataset containing 272 challenging problems.

### A.7 Reproduce Details

ConciseCoT & TALE-EP For the prompt-based baseline, we list the prompts used in Prompt [5.](#page-19-0) OverThink For the MATH12K dataset, we sample each problem 8 times. The shortest correct sample is selected as the chosen sample, and the longest sample is selected as the rejected sample. The model is trained for 1 epoch.

ThinkPruner In our reproduction, we use the competition-level training data provided in the original paper and train the model for 10 epochs with a learning rate of 1e-6. The maximum response length is set to 4096 tokens. We follow their early stopping strategy to select the optimal checkpoint for evaluation.

CoT-Valve Since CoT-Valve does not report performance on all datasets, we reproduced the results using the public datasets released by CoT-Valve. We followed the training settings officially reported in the paper, using LoRA=2 to fine-tune all models. The dataset version used is Mix-Chain-Z-GSM8K. All models were fine-tuned for 5 epochs on 8 GPUs with 80GB of memory each.

L1 In L1 reproduction on the 7B System-2 model, we utilize the *L1-Exact* reward function and limit the token length to between 100 and 4,096 tokens, while setting the token difference penalization parameter α to 0.0003, as described in the paper. We follow their original prompt by appending "Think for *ntoken* tokens" to the end of the question. In inference, the token budget is set to the same number as the average tokens from our method across the evaluated benchmarks.

