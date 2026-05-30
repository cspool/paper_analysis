# 6 RELATED WORKS

#### 6.1 MIXTURE-OF-EXPERTS

Mixture-of-Experts (MoE) was initially proposed in the early 1990s [\(Jacobs et al., 1991;](#page-11-8) [Jordan](#page-11-9) [& Jacobs, 1994\)](#page-11-9) and later introduced into large-scale neural networks as a sparse submodule for efficiency [\(Shazeer et al., 2017\)](#page-12-0). Advances like GShard [\(Lepikhin et al., 2020\)](#page-11-3) and Switch Transformer [\(Fedus et al., 2022\)](#page-10-0) integrated sparse MoE into Transformer models, achieving significant results. More recently, MoE has been used in commercial-scale language models such as Mixtral-8x7B [\(Jiang et al., 2024\)](#page-11-10), DeepSeekMoE 16B [\(Dai et al., 2024\)](#page-10-7), and Snowflake Arctic 17B [\(Snowflake, 2024\)](#page-12-15).

#### 6.2 ROUTING MECHANISMS IN MOE

Various routing methods have been developed for expert selection. Static routers, such as BASE [\(Lewis et al., 2021\)](#page-11-1), use predefined rules like combinatorial optimization, while Hash routing [\(Roller et al., 2021\)](#page-12-1) relies on deterministic hash functions, and THOR [\(Zuo et al., 2021\)](#page-13-8) assigns experts randomly with regularization. Learned routers adaptively select experts based on token input, using approaches like REINFORCE [\(Bengio et al., 2013;](#page-10-8) [Schulman et al., 2015;](#page-12-16) [Clark et al.,](#page-10-9) [2022\)](#page-10-9) for reinforcement learning, and TopK routing [\(Shazeer et al., 2017;](#page-12-0) [Zhou et al., 2022\)](#page-13-1) for token or expert selection, though TopK introduces discontinuities that hinder gradient estimation.

#### 6.3 DIFFERENTIABLE MIXTURE-OF-EXPERTS

Recent work on fully differentiable MoE models addresses the challenges of discrete optimization, basically through token merging and expert merging approaches. Soft MoE [\(Puigcerver et al., 2023\)](#page-12-2) uses token merging, assigning fixed slots to each expert as a linear combination of input tokens. SMEAR [\(Muqeeth et al., 2023\)](#page-12-3) merges experts into an ensemble via weighted averaging. However, both methods require a full probability map of input tokens, making them unsuitable for autoregressive models. Lory [\(Zhong et al., 2024\)](#page-13-3) preserves autoregressiveness by segmenting sentences to merge experts but underperforms compared to TopK routing.

