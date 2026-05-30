# 1 Introduction

Serving large language models (LLMs) (OpenAI, 2026; Anthropic, 2026; Google DeepMind, 2025; Meta, 2025; Qwen, 2026; DeepSeek-AI, 2024; MiniMax et al., 2025; Moonshot AI, 2025) over long contexts remains a central systems challenge. As context windows grow from 128K to 1M tokens and beyond—driven by demands for agentic multi-turn reasoning, long-document understanding, and native multimodal processing—the quadratic cost of self-attention becomes a dominant bottleneck in both prefill latency and memory consumption (Dao et al., 2022; Dao, 2023).

A productive line of work tackles this challenge through *sparse attention*: instead of attending to all key–value pairs, each query selects a small subset of the most relevant tokens and computes attention only over that subset. DeepSeek-V3.2 (DeepSeek-AI, 2025) adopts a *token-level* sparse attention paradigm, in which a lightweight *indexer* scores every historical token for each query, selects the top-*k* highest-scoring keys, and forwards only those keys to a downstream Sparse Multi-Head Latent Attention (Sparse MLA). This design has also been adopted in GLM-5 (GLM-5-Team, 2026) and provides strictly finer-grained selection than block-level methods such as MoBA (Lu et al., 2025) and Native Sparse Attention (Yuan et al., 2025).

However, the token-level sparse paradigm introduces a subtler bottleneck. Although the downstream attention is sparse and cheap, the indexer itself must score every token in the

<sup>\*</sup>Equal contribution.

<sup>&</sup>lt;sup>†</sup>Corresponding author: muhan@pku.edu.cn

prefix for every query. Concretely, if the prefix length is *L* and the indexer runs once per query per layer, the per-layer indexing cost is O(*L* 2 )—the same asymptotic scaling as dense attention. As context lengths push toward 128K or 1M tokens, the indexer can transition from a negligible overhead into the dominant cost component.

This observation motivates a natural question: *can we reduce the indexer's search cost without changing the final sparse attention pattern it produces?* In other words, can we rewrite the search *path* while preserving the search *result*?

We answer affirmatively with **HISA** (**H**ierarchical **I**ndexed **S**parse **A**ttention). HISA replaces the flat, full-prefix token scan with a two-stage hierarchical search (shown in Figure [1\)](#page-2-0):

- 1. **Block-level coarse filtering.** The prefix is partitioned into contiguous blocks of size *B*. A pooled representative vector is computed for each block via mean pooling over its constituent indexing keys. The query scores all ⌈*L*/*B*⌉ block representatives and retains only the top-*m* blocks, immediately pruning the majority of the prefix from further consideration.
- 2. **Token-level refinement.** The token-level indexer then scores at most *mB* tokens from the candidate blocks using the same scoring mechanism as the original DSA indexer, except that the candidate pool is restricted to the tokens within the selected blocks rather than the full set of *L* tokens considered in DSA. The final top-*k* token set is then selected from this reduced candidate pool.

Crucially, HISA produces outputs with the same structure as the original DSA indexer: for each query, a set of *k* token indices. As a result, the downstream Sparse MLA operator remains entirely unchanged. HISA is therefore a **drop-in replacement** that requires no retraining, no architectural changes to the attention mechanism, and no modification to the KV cache layout. The per-query indexing complexity drops from O(*L*) to O(*L*/*B* + *mB*), and the per-layer cost drops from O(*L* 2 ) to O(*L* <sup>2</sup>/*B* + *LmB*).

Our contributions are as follows:

- We identify the indexer as an emerging bottleneck in token-level sparse attention systems and formalize the problem of **search-path optimization** for sparse indexers.
- We propose HISA, a hierarchical block-to-token indexing strategy that is trainingfree, operator-compatible, and asymptotically faster than the flat indexer.
- We provide optimized TileLang GPU kernel implementations for both stages of HISA and demonstrate 2–4× kernel-level speedup at 64K contexts.
- We empirically validate that HISA achieves performance comparable to the original DSA on the Needle-in-a-Haystack and LongBench benchmarks.

