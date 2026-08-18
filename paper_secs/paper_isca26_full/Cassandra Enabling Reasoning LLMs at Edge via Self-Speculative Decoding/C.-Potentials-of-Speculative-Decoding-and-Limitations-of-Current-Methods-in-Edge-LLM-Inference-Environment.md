# C. Potentials of Speculative Decoding and Limitations of Current Methods in Edge LLM Inference Environment

Given the absence of a definitive solution for optimizing edge LLM inference, speculative decoding emerges as a promising approach for accelerating LLMs on consumer-grade xPUs. However, existing research on speculative decoding has primarily targeted large-scale models deployed in server environments, leaving several challenges unresolved in edge settings.

The most critical limitation is that many speculative decoding approaches rely on additional training. Although recent

TABLE II
COMPARISON OF DIFFERENT SPECULATIVE DECODING METHODS

| Scheme                    | Speculative<br>Sampling | Draft &<br>Verify | MagicDec | Lookahead<br>Decoding | Eagle-3 | Cassandra |
|---------------------------|-------------------------|-------------------|----------|-----------------------|---------|-----------|
| <br>Training free?        | X                       | <b>✓</b>          | ✓        | <b>√</b>              | X       | <b>✓</b>  |
| No Capacity<br>Overhead?  | Х                       | 1                 | <b>√</b> | <b>√</b>              | Х       | 1         |
| Low-Batch<br>Performance? | 1                       | Х                 | Х        | 1                     | 1       | 1         |
| Cross-Task<br>Reliability | X                       | 1                 | ✓        | Х                     | Х       | 1         |

methods have reduced the training cost, such requirements remain impractical for users with resource-constrained devices. For instance, Eagle-3 speculative decoding model [28], a state-of-the-art training-based approach, requires approximately two days of training on four NVIDIA A100 GPUs to construct the draft model. This level of computational demand presents a significant barrier for edge users, who typically operate with limited hardware resources.

To address these issues, recent studies have explored training-free self-speculative decoding [52], [56], [67]. These approaches construct the draft model directly from the original model, eliminating the need for additional training. For example, Draft&Verify [67] generates draft outputs via layer skipping, while MagicDec [52] leverages sparse retrieval of the KV cache. Despite their advantages, existing self-speculative methods typically rely on coarse-grained approximations and primarily focus on reducing overhead in the attention mechanism and KV cache. As a result, although they demonstrate reasonable performance in batched inference scenarios, their benefits are often limited in low-batch, small-model settings that are common in edge environments.

In addition, several speculative decoding methods [2], [28] employ independent draft models with separate weights and KV caches. This design introduces non-negligible memory overhead, which is particularly problematic for resource-constrained devices, as it reduces the maximum feasible sequence length under a fixed memory budget.

As summarized in Table II, existing speculative decoding approaches exhibit distinct trade-offs, and no single method fully satisfies the requirements of edge deployment. Therefore, we propose Cassandra, a novel approach designed to overcome these limitations and enable efficient speculative decoding in edge LLM inference scenarios.

# C. Potentials of Speculative Decoding and Limitations of Current Methods in Edge LLM Inference Environment

Given the absence of a definitive solution for optimizing edge LLM inference, speculative decoding emerges as a promising approach for accelerating LLMs on consumer-grade xPUs. However, existing research on speculative decoding has primarily targeted large-scale models deployed in server environments, leaving several challenges unresolved in edge settings.

The most critical limitation is that many speculative decoding approaches rely on additional training. Although recent

TABLE II
COMPARISON OF DIFFERENT SPECULATIVE DECODING METHODS

| Scheme                    | Speculative<br>Sampling | Draft &<br>Verify | MagicDec | Lookahead<br>Decoding | Eagle-3 | Cassandra |
|---------------------------|-------------------------|-------------------|----------|-----------------------|---------|-----------|
| <br>Training free?        | X                       | <b>✓</b>          | ✓        | <b>√</b>              | X       | <b>✓</b>  |
| No Capacity<br>Overhead?  | Х                       | 1                 | <b>√</b> | <b>√</b>              | Х       | 1         |
| Low-Batch<br>Performance? | 1                       | Х                 | Х        | 1                     | 1       | 1         |
| Cross-Task<br>Reliability | X                       | 1                 | ✓        | Х                     | Х       | 1         |

methods have reduced the training cost, such requirements remain impractical for users with resource-constrained devices. For instance, Eagle-3 speculative decoding model [28], a state-of-the-art training-based approach, requires approximately two days of training on four NVIDIA A100 GPUs to construct the draft model. This level of computational demand presents a significant barrier for edge users, who typically operate with limited hardware resources.

To address these issues, recent studies have explored training-free self-speculative decoding [52], [56], [67]. These approaches construct the draft model directly from the original model, eliminating the need for additional training. For example, Draft&Verify [67] generates draft outputs via layer skipping, while MagicDec [52] leverages sparse retrieval of the KV cache. Despite their advantages, existing self-speculative methods typically rely on coarse-grained approximations and primarily focus on reducing overhead in the attention mechanism and KV cache. As a result, although they demonstrate reasonable performance in batched inference scenarios, their benefits are often limited in low-batch, small-model settings that are common in edge environments.

In addition, several speculative decoding methods [2], [28] employ independent draft models with separate weights and KV caches. This design introduces non-negligible memory overhead, which is particularly problematic for resource-constrained devices, as it reduces the maximum feasible sequence length under a fixed memory budget.

As summarized in Table II, existing speculative decoding approaches exhibit distinct trade-offs, and no single method fully satisfies the requirements of edge deployment. Therefore, we propose Cassandra, a novel approach designed to overcome these limitations and enable efficient speculative decoding in edge LLM inference scenarios.

