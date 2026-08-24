# B Additional Details on Methodology and Experiments

### <span id="page-16-2"></span>B.1 Baseline Implementation Details

We describe the implementation details of the baseline methods as follows. For the Random router, a model-strategy pair is randomly selected for each input. We report the average performance over 50 runs. For the KNN-Router, we set the parameter k to 10. For each model-strategy pair, we compute its performance over all neighboring queries, where the score is defined as the average accuracy minus the average output length multiplied by a parameter λ. We tune λ to balance cost and performance, and report the best result. For RouteLLM, since this is a binary router, we label each query according to its average accuracy on the candidate models, and train a binary classifier to decide which model to route each query to. We tune the balance parameter and report the best result. The strong and weak models used are QwQ and Qwen2.5-7B, respectively. For EmbedLLM, we assign a unique ID to each model-strategy pair and learn a 768-dimensional embedding for each pair. Following the original method, we frame the learning process as a reconstruction task: given a matrix of model correctness across prompts, a reconstruction network is trained to recover this matrix, thereby enforcing the embeddings to capture the behavioral characteristics of each model-strategy pair.

### <span id="page-16-1"></span>B.2 Prompts Used in Experiments

### CoT

Please reason step by step before providing the final answer, and put your final answer within \\boxed{{}}.

### CoD

Think step by step, but only keep minimum draft for each thinking step, with 5 words at most, and put your final answer within \\boxed{{}}.

### PAL

Write a Python code snippet to solve the following problem. Do not use any plotting libraries or the input() function.

### <span id="page-16-0"></span>B.3 Profiles of Strategies and Models

Vanilla: Vanilla prompting retains the original question content without adding any additional prompt information.

Chain-of-Thought (CoT): Chain-of-Thought (CoT) prompting guides the model to articulate a step-by-step reasoning process before providing the final answer. This results in longer responses and slower reasoning speed, typically generating the longest answers, but it performs best on complex problems such as mathematical reasoning.

Chain-of-Draft (CoD): Chain-of-Draft (CoD) prompts the model to generate only intermediate drafts with explicit constraints on output length, encouraging concise reasoning. These drafts represent the model's thinking process, often containing important calculation steps and key reasoning information. It simplifies the intermediate steps of the reasoning chain while retaining good performance, resulting in shorter answers.

Program-Aided Language (PAL): Program-Aided Language (PAL) transforms the reasoning process into executable code. This approach leverages the determinism of programming languages to ensure logical consistency and high accuracy, making it particularly effective for mathematical, symbolic, or algorithmic tasks. PAL relies on a suitable code execution environment and consistently produces results with high reliability and stable, moderately sized outputs. However, it may not well suited for commonsense reasoning tasks.

Qwen2.5-3B-Instruct: Qwen2.5-3B-Instruct is a lightweight 3B parameter model with fast inference and low resource usage. It is suitable for simple tasks such as basic question answering and short-form text generation, but is limited in handling complex reasoning or multi-step tasks.

Qwen2.5-7B-Instruct: Qwen2.5-7B-Instruct is a mid-small 7B parameter model that balances speed and performance. It is capable of multi-turn dialogue, basic code and math tasks, and offers improved language understanding over smaller models, while maintaining efficient inference.

Qwen2.5-14B-Instruct: Qwen2.5-14B-Instruct is a mid-sized 14B parameter model that excels at complex reasoning, document summarization, and structured mid-length text generation. It demonstrates strong performance on tasks requiring deeper understanding and context retention.

DeepSeek-R1-7B: DeepSeek-R1-7B is a 7B distilled model with slightly slower but stable inference compared to other models of similar size. It is well-suited for medium-length answers that require deep reasoning, and often generates more detailed and comprehensive responses.

DeepSeek-R1-14B: DeepSeek-R1-14B is a 14B distilled model with slower inference but strong logical and mathematical capabilities. It is ideal for mathematical proofs and tasks requiring rigorous step-by-step reasoning, offering robust performance in logic-intensive scenarios.

QwQ-32B: QwQ-32B is a large 32B quantized model with slow inference. It excels at complex logic, coding, and multi-step reasoning tasks, though it may produce verbose outputs. Its large capacity enables handling of challenging prompts and long-context tasks.

Qwen3-4B (thinking): Qwen3-4B in 'thinking' mode generates longer reasoning chains and detailed thought processes. While it has slower inference and higher resource usage, it excels at solving complex logic and reasoning problems, making it suitable for tasks that require step-by-step explanations or in-depth analysis.

Qwen3-4B (non-thinking): Qwen3-4B in 'non-thinking' mode is optimized for short, direct answers. It provides fast inference with low resource cost, but is limited in deep reasoning or step-by-step explanations, making it best for straightforward queries or when efficiency is prioritized.