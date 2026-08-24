# <span id="page-10-3"></span>B Prompt Templates and Generated Examples

We prompt the LLM to produce a direct thought: a concise, structured, step-by-step explanation that outlines how to derive the answer from the given question. A direct thought always consists of a

small number of clear logical steps that directly contribute to the solution, and terminates with a single, explicit final answer. We use the prompt below to elicit such direct thoughts (the model is instructed to return only a detailed step-by-step solution containing only "Step-by-Step Solution" and "Final Answer").

#### Prompt for Generating the Direct CoT P

Given a question, please tell me how to get this answer step by step.

Question:

*{question}*

```

```

Answer:

```python

*{answer}* ```

Only return a detailed step-by-step solution (containing only "Step-by-Step Solution" and "Final Answer"). The detailed step-by-step solution is:

The following example illustrates a direct thought for a simple programming task: converting an integer into its binary representation. This example demonstrates the defining characteristics of direct CoTs, namely, concise step-by-step reasoning that directly derives the solution, without redundant exploration or verbose commentary.

#### Example of a Direct CoT

#### Step-by-Step Solution

- 1. Understand the Problem: The task is to convert a given positive integer n into its binary representation without leading zeros. The constraints are lenient (n ≤ 10<sup>6</sup> ), so even a straightforward approach is efficient.
- 2. Input Handling: Read the integer n from standard input. Since the input consists of a single integer, we can directly use input() and convert it to an integer.
- 3. Binary Conversion: Python's built-in bin() function converts an integer to its binary string representation (prefixed with 0b). To remove the prefix, slice the string from the third character onward ([2:]).
- 4. Output the Result: Print the resulting binary string. This ensures no leading zeros are present.
- 5. Efficiency: The approach is optimal because conversion and slicing both run in O(log n) time, and memory usage is minimal.

#### Final Code:

```

```python n = int(input()) print(bin(n)[2:])

Given the direct thought as a reference, this stage instructs the model to prune redundant reasoning from the original chain of thought while preserving the essential logical structure. Specifically, the model is required to 1) remove unnecessary reasoning steps from the original CoT, 2) retain all key supporting content that aligns with the logic of direct CoT, and 3) strictly preserve the original wording and sentence order without introducing new information. This ensures that the compressed reasoning remains faithful to the original thought process while aligning with the concise, goal-oriented structure of the direct CoT. The following detailed prompt is used to elicit such coarse-grained pruning behavior.

## Prompt for Coarse-grained Pruning

Compress the given thinking by referring to the provided solution. The goal is to remove irrelevant reasoning paths while retaining all content along the core reasoning path. Compression must be based on thinking, ensuring that the original wording and structure are preserved as much as possible. Follow these strict rules:

- 1. Use thinking as the foundation: Do not rewrite or replace its content with solution——only use solution to determine which parts are relevant.
- 2. Remove unnecessary reasoning: Aggressively remove alternative paths that are not part of the core reasoning path.
- 3. Retain key supporting content: Keep examples, reflections, and tests that help illustrate, verify, or analyze the core reasoning path.
- 4. Preserve original words: Do not paraphrase, reorder, or change any words.
- 5. Do not add new words: Do not introduce new concepts, symbols, or abbreviations.

If you understand, compress the following thinking based on the given solution.

Solution:

``` *{solution}*

Thinking:

``` *{think}* ```

```

The compressed thinking is:

## <span id="page-11-0"></span>C Implementation Details

Software and Hardware. For fine-tuning, we utilized the unsloth library[1](#page-11-1) for its memoryefficient optimizations. For inference, we employed the vLLM engine[2](#page-11-2) to maximize throughput and efficiency. All experiments were conducted on NVIDIA H20 GPUs and Intel Xeon Platinum 8480+ CPUs.

<span id="page-11-1"></span><sup>1</sup> https://pypi.org/project/unsloth/2025.5.6/

<span id="page-11-2"></span><sup>2</sup> https://pypi.org/project/vllm/0.8.4/

Fine-tuning Configuration. We performed fullparameter fine-tuning for all models in our experiments. Key hyperparameters included precision set to bf16, num\_train\_epochs set to 10, and a max\_seq\_length of 16384. We used a per\_device\_train\_batch\_size of 1 with gradient\_accumulation\_steps set to 16, resulting in an effective batch size of 16. For the optimizer, we used AdamW with a cosine\_with\_min\_lr learning rate scheduler. The warmup\_ratio was set to 0.03, and the scheduler's min\_lr\_rate was 0.1 of the peak learning rate. To stabilize training, we applied gradient clipping with a max\_grad\_norm of 0.2. Based on preliminary experiments, we set the peak learning rate to 4×10−<sup>5</sup> for the DeepSeek-R1-Distill-Qwen-7B and 2 × 10−<sup>5</sup> for the DeepSeek-R1-Distill-Llama-8B. Due to the high computational cost of fullparameter fine-tuning, the model is fine-tuned by a single run with a fixed random seed 42.

Inference and Evaluation Protocol. All inference benchmarks were run using the vLLM engine with dtype set to bfloat16 and gpu\_memory\_utilization set to 0.9. To ensure deterministic and reproducible outputs, we set the sampling temperature to 0.0 and set enable\_prefix\_caching to False. The default token budget for generation is adjusted based on the task difficulty. Specifically, it is 2K for GSM8K, 4K for MATH500, 6K for HumanEval+, and 10K for AIME24, AIME25, LiveCodeBench, and LeetCodeDataset. Results with other token budget settings are shown in Appendix [F.](#page-13-1)

Baseline Details. Following established practices, we used a consistent scoring model; as our primary model is DeepSeek-R1-Distill checkpoints, we employed DeepSeek-R1-Distill-Qwen-7B for all model-scoring tasks. To ensure a fair comparison, we standardize the input format across all methods by preserving the original question and final answer, and applying compression only to the CoT reasoning steps. To balance compression ratio and content retention, we set the target compression ratio to 0.5 for all baseline methods, except for TokenSkip, where we follow its original design that allows a controllable compression ratio between 0.5 and 1.0. Additionally, since the original SPIRIT method is computationally expensive when applied to extremely long CoTs, we adopt a modified version to ensure fair comparison: specifically, we compute perplexity once per reasoning step and

iteratively remove steps until the target ratio is met. This variant retains the core idea of SPIRIT while improving scalability in our evaluation setting.

Hyperparameters for Our Method. Our method involves several stages. For the LLMguided Coarse-grained Pruning stage, we employed DeepSeek-V3 for economic reasons. When generating the direct thought P, we used a deterministic setting (temperature=0.0, top\_p=1.0), while for making the final pruning result, we increased exploration (temperature=1.0, top\_p=1.0). For Pattern Matching, the similarity threshold τ was set to 0.6. Finally, during Surprisal-based Fine-grained Pruning, the maximum token budget was set to 4096 to ensure a deep level of compression.

## <span id="page-12-0"></span>D Effect of Different Components.

To validate the contribution and necessity of each component in our two-stage pruning framework, we conduct a detailed ablation study. Specifically, we evaluate the following three variants: *ASAP w/o Coarse-grained Pruning*, *ASAP w/o Fine-grained Pruning*, and *ASAP w/o Any Pruning*. We present results on the HumanEval+, Live-CodeBench v1\_v3, and LeetCodeDatsets benchmarks in Table [6,](#page-12-1) Table [7,](#page-12-2) and Table [8.](#page-13-3)

<span id="page-12-1"></span>

| Variants                              | Acc ↑<br>Tok ↓<br>Lat ↓ |  |      |  |  |  |  |
|---------------------------------------|-------------------------|--|------|--|--|--|--|
| ASAP                                  | 78.66 2464              |  | 0.98 |  |  |  |  |
| w/o Coarse-grained Pruning 78.05 2839 |                         |  | 1.10 |  |  |  |  |
| w/o Fine-grained Pruning              | 67.07 2897              |  | 1.10 |  |  |  |  |
| w/o Any Pruning                       | 75.61 2973              |  | 1.12 |  |  |  |  |

Table 6: Ablation study of different pruning strategies for ASAP on HumanEval+. We report accuracy (Acc), average number of generated tokens (Tok), and average generation latency (Lat) measured in seconds.

<span id="page-12-2"></span>

| Variants                              | Acc ↑<br>Tok ↓<br>Lat ↓ |  |      |  |  |  |  |
|---------------------------------------|-------------------------|--|------|--|--|--|--|
| ASAP                                  | 54.74 5177              |  | 2.09 |  |  |  |  |
| w/o Coarse-grained Pruning 53.92 6107 |                         |  | 2.77 |  |  |  |  |
| w/o Fine-grained Pruning              | 51.14 6599              |  | 3.20 |  |  |  |  |
| w/o Any Pruning                       | 52.12 6611              |  | 3.15 |  |  |  |  |

Table 7: Ablation study of different pruning strategies for ASAP on LiveCodeBench v1\_v3. We report accuracy (Acc), average number of generated tokens (Tok), and average generation latency (Lat) measured in seconds.

<span id="page-13-3"></span>

| Variants                              | Acc ↑<br>Tok ↓<br>Lat ↓ |  |      |  |  |  |  |
|---------------------------------------|-------------------------|--|------|--|--|--|--|
| ASAP                                  | 27.63 7541              |  | 3.48 |  |  |  |  |
| w/o Coarse-grained Pruning 24.12 7954 |                         |  | 3.75 |  |  |  |  |
| w/o Fine-grained Pruning              | 25.44 8326              |  | 4.77 |  |  |  |  |
| w/o Any Pruning                       | 25.00 8485              |  | 4.72 |  |  |  |  |

Table 8: Ablation study of different pruning strategies for ASAP on LeetCodeDataset. We report accuracy (Acc), average number of generated tokens (Tok), and average generation latency (Lat) measured in seconds.

