# B Broader impact

SALE significantly reduces the computational cost of the long-context LLM prefilling, thereby lowering deployment costs and enabling broader adoption of AI technologies. This advancement also facilitates the development of applications that rely on processing long contexts. Additionally, it contributes to a reduction in energy consumption of LLM services.

## C Additional implementation details

We select five input samples from the Retrieve.KV task in InfiniteBench to perform calibration for SALE, and the final configuration must satisfy the error bound requirement across all five samples. The per-head threshold calibration for Llama-3.1 on RTX4090 server takes approximately five minutes to complete.

For the local area discussed in Section [3.2,](#page-2-0) we set its size to be no smaller than 128 tokens. Since the comparison results of *Relative Attention Score* and τ for each thread are not visible to others, an all-reduce operation must be performed across all threads within the GPU thread block to aggregate these results, which incurs considerable overhead. To reduce the frequency of all-reduce operations, we group every four consecutive key blocks into a segment and perform result aggregation at the segment level. At the end of each row, any remaining key blocks (less than 4 blocks) are also treated as local area blocks for implementation simplicity. As a result, the number of tokens in local area may exceed 128, but will not surpass 256.

## D Additional experiment details

We use the same input samples to search the optimal hyperparameters for SpargeAttn, and use the first input sample to search sparse pattern configuration for MInference based on its open-source implementation.

During evaluation process, to ensure proper model behavior, we truncate samples that exceed the maximum context window length. Following common practice, we retain the tokens from both the beginning and the end of the sequence and remove those from the middle portion.

For the data format during model inference, we employed BFloat16 for FlexPrefill due to requirements specified in its repository, while Float16 was used for all other methods.

