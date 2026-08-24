# C. Detailed Experiment Setups

## C.1. Model

We use DeepSeek-R1[\(DeepSeek-AI et al.,](#page-8-1) [2025\)](#page-8-1), Qwen3-32B[\(Team,](#page-8-15) [2025a\)](#page-8-15), QwQ-32B[\(Team,](#page-8-16) [2025b\)](#page-8-16), Llama-3.3- Nemotrom-Super-49B-V1[\(Bercovich et al.,](#page-9-0) [2025\)](#page-9-0), Distill-Qwen-7B, Distill-Qwen-1.5B[\(Yu et al.,](#page-9-17) [2024\)](#page-9-17), and Qwen-2.5- 3B-Instruct[\(Team,](#page-9-15) [2024\)](#page-9-15) models in our paper. We introduce their licenses and key characteristics as follows:

- DeepSeek-R1. An open-source 671 B→37 B MoE reasoning model trained largely through reinforcement learning, which elicits self-verification, reflection and lengthy chain-of-thought traces while supporting 128K-token context; it matches proprietary o1 on math / code benchmarks using only public data.
- Qwen3-32B. The 32.8 B-parameter third-generation Qwen model that toggles between "thinking" and "non-thinking" modes, delivering state-of-the-art reasoning, multilingual chat and up to 131 K context in a single dense checkpoint.

