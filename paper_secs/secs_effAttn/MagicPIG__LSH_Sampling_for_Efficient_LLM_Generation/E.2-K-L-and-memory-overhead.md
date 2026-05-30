# E.2 (K, L) and memory overhead

(K, L) will change two overheads brought by MagicPIG: the memory occupied by hash tables on the CPU and extra computation for random projections (hash functions) on the GPU (as shown in Table [8\)](#page-21-0).

<span id="page-21-0"></span>Table 8 The overhead of Locality sensitive hashing during decoding. We report the size of random projectors (on GPU) and hash tables (on CPU), the computation overhead CO (refers to the ratio between computation introduced by random projections in LSH and the computation of the original model's linear projections (e.g., WQ, WK, W<sup>V</sup> , and MLP)). Notice that when the context length exceeds 64K, we need to use 32-bit integers to store the indices for the KV cache in hash tables. Llama-3.1-8B/70B-Instruct [\(Dubey et al.,](#page-13-0) [2024\)](#page-13-0) and Code-Llama-34b-16K [Rozi`ere et al.](#page-15-14) [\(2024\)](#page-15-14) use group query attention, thus the sizes of hash tables are reduced.

| Models                 | (K, L)    | Context length | Projectors | Hash tables | CO   |
|------------------------|-----------|----------------|------------|-------------|------|
| Llama-3.1-8B-Instruct  | (10, 150) | 96K            | 384KB      | 14GB        | 3.8% |
| Llama-3.1-8B-Instruct  | (11, 300) | 96K            | 825KB      | 28GB        | 8.5% |
| Llama-3.1-8B-Instruct  | (10, 150) | 64K            | 384KB      | 4.7GB       | 3.8% |
| Llama-3.1-70B-Instruct | (10, 150) | 64K            | 384KB      | 11.8GB      | 1.8% |
| Code-Llama-13b-16K     | (10, 150) | 16K            | 384KB      | 7.3GB       | 5.2% |
| Code-Llama-34b-16K     | (10, 150) | 16K            | 384KB      | 1.8GB       | 2.2% |

LLM decoding is a memory-bandwidth-bound process and the majority of time is spent loading the data (parameters/KV cache) to GPU cores rather than actually doing the computation [\(Miao et al.,](#page-14-18) [2023;](#page-14-18) [Zhang](#page-15-16) [et al.,](#page-15-16) [2023a;](#page-15-16) [Chen et al.,](#page-13-19) [2024\)](#page-13-19). Besides, the time-consuming part, i.e., the long-context attention computation, is moved to the CPU. Thus, the 1.8% ∼ 8.5% extra computation on GPU will only make a minor difference in execution time. However, the enlarged size of hash tables prevents us from always increasing (K, L) to get more accurate results.

As shown in Table [8,](#page-21-0) under the same (K, L), the memory overhead of hash tables grows linearly with context length and the total number of key-value heads in models (which is determined by model sizes).

