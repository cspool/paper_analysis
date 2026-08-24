# Walk Before You Run! Concise LLM Reasoning via Reinforcement Learning

Mingyang Song, Mao Zheng Tencent Hunyuan nickmysong@tencent.com

#### Abstract

As test-time scaling becomes a pivotal research frontier in Large Language Models (LLMs) development, contemporary and advanced post-training methodologies increasingly focus on extending the generation length of long Chain-of-Thought (CoT) responses to enhance reasoning capabilities toward DeepSeek R1-like performance. However, recent studies reveal a persistent overthinking phenomenon in state-of-the-art reasoning models, manifesting as excessive redundancy or repetitive thinking patterns in long CoT responses. To address this issue, in this paper, we propose a simple yet effective two-stage reinforcement learning framework for achieving concise reasoning in LLMs, named ConciseR. Specifically, the first stage, using more training steps, aims to incentivize the model's reasoning capabilities via Group Relative Policy Optimization with *clip-higher* and *dynamic* sampling components (GRPO++), and the second stage, using fewer training steps, explicitly enforces conciseness and improves efficiency via Length-aware Group Relative Policy Optimization (L-GRPO), Significantly, ConciseR only optimizes response length once all rollouts of a sample are correct, following the "walk before you run" principle. Extensive experimental results demonstrate that our **ConciseR** model, which generates more concise CoT reasoning responses, outperforms recent state-of-the-art reasoning models with zero RL paradigm across AIME 2024, MATH-500, AMC 2023, Minerva, and Olympiad benchmarks. The code, training dataset, and model checkpoints will be publicly released<sup>1</sup>.

> **[图片提取文字 (无描述)]:**
> AIME 2024 Pass@1 (k=32) Accuracy (%) Average Response Length AIME 2024 Pass@1 (k=32) Accuracy (%) ConciseR-Zero-7B ConciseR-Zero-7B-Preview 15+ Training Steps
![](_page_0_Figure_5.jpeg)

<span id="page-0-1"></span>Figure 1: A detailed evaluation of accuracy and response length throughout the training steps.

<span id="page-0-0"></span>https://github.com/nick7nlp/ConciseR

