# <span id="page-6-0"></span>4 A Comprehensive Evaluation of LCLMs on LONGPROC

**Setup.** We evaluate 23 LCLMs on LONGPROC across different output length configurations. For frontier closed-sourced models, we include GPT-40 (Achiam et al., 2023), Claude 3.5 (Anthropic, 2024), and Gemini 1.5 (Gemini Team, 2023; 2024). For open-weight models, we test various model families across different parameter scales and architectures, including ProLong (Gao et al., 2024), Llama-3 (Llama-3 Team, 2024), Mistral-v0.3 (Jiang et al., 2023), Phi-3 (Abdin et al., 2024), Qwen-2.5 (Yang et al., 2024), and Jamba (Lieber et al., 2024). We also cover several recently released reasoning models such as R1-distilled models (Guo et al., 2025).

Recall that LONGPROC includes three difficulty levels based on generation lengths: 500, 2K, and 8K tokens. Since different tokenizers produce varying numbers of tokens when encoding the same output, we allow an additional 0.5K-1K token buffer in generation length for all models to accommodate variations across different tokenizers. We use greedy decoding for all models given the deterministic nature of procedural generation. For reasoning models, we allow up to 16K tokens for generation to accommodate the additional

<span id="page-7-1"></span>

|                        | HTML to TSV  |      |      |      | Pseudocode to Code |      |      | Path Traversal  |      |  |  |
|------------------------|--------------|------|------|------|--------------------|------|------|-----------------|------|--|--|
| Llama-3.1-8B-Inst      | 43.4         | 29.0 | 23.4 | 63.0 | 28.0               | -    | 17.0 | 0.0             | 0.0  |  |  |
| Qwen2.5-7B-Inst        | 45.2         | 32.1 | 17.0 | 60.0 | 31.0               | -    | 0.0  | 0.0             | 0.0  |  |  |
| Qwen2.5-32B-Inst       | 74.5         | 46.9 | 25.4 | 81.0 | 65.0               | -    | 29.0 | 0.0             | 0.0  |  |  |
| R1-Distill-Qwen2.5-32B | 78.2         | 53.9 | 38.8 | 70.0 | 59.0               | -    | 97.0 | 61.0            | 0.0  |  |  |
| Llama-3.3-70B-Inst     | 78.0         | 60.2 | 51.7 | 76.0 | 64.0               | -    | 70.0 | 1.0             | 0.0  |  |  |
| R1-Distill-Llama3-70B  | 74.8         | 60.2 | 46.4 | 74.0 | 57.0               | -    | 95.0 | 90.0            | 31.0 |  |  |
| GPT-4o-2024-08         | 87.0         | 76.4 | 65.5 | 90.0 | 84.0               | -    | 98.0 | 77.0            | 34.0 |  |  |
| Gemini-1.5-pro-001     | 81.3         | 75.3 | 70.0 | 81.6 | 50.0               | -    | 97.0 | 96.0            | 81.0 |  |  |
|                        | 0.5K         | 2K   | 8K   | 0.5K | 2K                 | 8K   | 0.5K | 2K              | 8K   |  |  |
|                        | ToM Tracking |      |      |      |                    |      |      |                 |      |  |  |
|                        |              |      |      |      | Countdown          |      |      | Travel Planning |      |  |  |
| Llama-3.1-8B-Inst      | 17.0         | 0.0  | 0.0  | 8.0  | 12.0               | 3.0  | -    | 55.0            | 0.0  |  |  |
| Qwen2.5-7B-Inst        | 2.0          | 0.0  | 0.0  | 32.0 | 36.0               | 2.0  | -    | 39.0            | 0.0  |  |  |
| Qwen2.5-32B-Inst       | 65.0         | 8.0  | 0.0  | 96.0 | 87.0               | 55.0 | -    | 95.0            | 5.0  |  |  |
| R1-Distill-Qwen2.5-32B | 67.0         | 50.0 | 0.0  | 91.0 | 88.0               | 51.0 | -    | 54.0            | 22.0 |  |  |
| Llama-3.3-70B-Inst     | 87.0         | 45.0 | 0.0  | 77.0 | 89.0               | 61.0 | -    | 86.0            | 12.0 |  |  |
| R1-Distill-Llama3-70B  | 76.0         | 59.0 | 7.0  | 99.0 | 86.0               | 47.0 | -    | 71.0            | 35.0 |  |  |
| GPT-4o-2024-08         | 100.0        | 77.0 | 0.0  | 99.0 | 95.0               | 67.0 | -    | 91.0            | 24.0 |  |  |
| Gemini-1.5-pro-001     | 92.0         | 71.0 | 28.0 | 94.0 | 84.0               | 46.0 | -    | 100.0           | 45.0 |  |  |

Figure 3: Performance comparison across tasks and output lengths (grey blocks indicate unavailable configurations).

"thinking" tokens. Otherwise, they often fail to complete generation. In contrast, instructiontuned models do not effectively utilize additional tokens, as they typically terminate early even when given a higher token limit. We provide in-context examples along with solving procedures in prompts for all tasks, except for HTML to TSV and Pseudocode to Code (see Appendix [H](#page-42-0) for detailed prompts).

**Results.** Table [3](#page-6-1) summarizes LCLM **average performance** across tasks at different lengths.

**Existing models struggle in extensive long procedural generation.** Frontier proprietary models demonstrate the best performance. GPT-4o and Gemini-Pro achieve near-perfect scores on 0.5K tasks and maintain strong performance at 2K. However, they experience substantial performance degradation at 8K, falling well short of their claimed context windows of 128K tokens or more. Open-weight models lag significantly behind proprietary models. Models under 15B parameters struggle with 2K tasks, with the best performer, R1-Distill-Llama-3-8B, reaching only 24.6. Mid-sized models (13B–70B parameters) handle some 8K tasks but achieve substantially lower performance than frontier models.

**Model scale is critical for task performance.** We observe substantial performance differences across scales within model families, as evidenced by the gaps between Llama3.1-8B versus Llama3.1-70B and Qwen-8B versus Qwen-72B. While differences caused by model families are less pronounced than scale-related gaps, notable performance variations still exist between similarly-sized models. For example, Llama-3.1-70B outperforms Qwen2.5-72B by approximately 30% (relative gain) on both 2K and 8K token sets.

**Reasoning models outperform their instruction-tuned counterparts.** For example, R1- Distill-Qwen2.5-32B outperforms Qwen2.5-32B-Inst by approximately 10% relative gain across all three difficulty levels. This performance gap suggests a potential synergy between long CoT training and long-form generation capabilities. We provide a more detailed analysis of reasoning models in [§5.](#page-7-0)

