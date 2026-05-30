# Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models

Keivan Alizadeh Iman Mirzadeh <sup>∗</sup> Hooman Shahrokhi† Dmitry Belenko

Frank Sun Minsik Cho Mohammad Sekhavat Moin Nabi Mehrdad Farajtabar

Apple

## Abstract

Large Language Models (LLMs) typically generate outputs token by token using a fixed compute budget, leading to inefficient resource utilization. To address this shortcoming, recent advancements in mixture of expert (MoE) models, speculative decoding, and early exit strategies leverage the insight that computational demands can vary significantly based on the complexity and nature of the input. However, identifying optimal routing patterns for dynamic execution remains an open challenge, limiting the full potential of these adaptive methods. To address this need, we study adaptive computation in LLMs more systematically. We propose a novel framework that integrates smaller auxiliary modules within each Feed-Forward Network layer of the LLM. This design enables dynamic routing of tokens based on task complexity: tokens can be processed by either the small or big modules at each layer, or even bypass certain layers entirely. This allows us to introduce a novel notion of a token's difficulty, defined by its potential to benefit from additional computational resources. Importantly, by employing oracles to identify optimal patterns of adaptive computations, we gain valuable insights into the internal workings of LLMs and the routing processes in a simplified heterogeneous MoE setup. We show that trained routers operate differently from oracles and often yield suboptimal solutions. Notably, activating a large module in just one layer outperforms models that use large modules across all layers, underscoring the gap between practical implementations of routing in MoE models and theoretical optima for adaptive computation.

