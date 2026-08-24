# <span id="page-12-0"></span>G Analysis of Negatively Weighted Attention Heads

To better understand the role of attention heads assigned negative weights by Sentinel (Qwen-2.5- 0.5B-Instruct), we analyze their attention distributions on 100 examples from the HotpotQA dataset. This analysis examines which input components these heads predominantly attend to, and whether their negative contributions correspond to known non-informative attention behaviors.

Analysis Setup. We analyze attention patterns on 100 HotpotQA examples by grouping input tokens into four categories: *(i)* sink tokens (e.g., special tokens and structurally dominant positions), *(ii)* supporting evidence sentences, *(iii)* question tokens, and *(iv)* remaining context. For each attention head, we compute the average proportion of attention mass assigned to each category.

Results. As shown in Table [8,](#page-13-1) attention heads assigned strong negative weights by Sentinel predominantly attend to sink tokens or question tokens, while allocating little to no attention to supporting evidence. In contrast, positively weighted heads focus primarily on evidence-bearing context.

Implications. This analysis shows that negatively weighted heads capture structurally dominant but semantically uninformative behaviors, such as attention sinks or self-focused query attention. Explicitly down-weighting these heads allows Sentinel to suppress spurious attention patterns and decode context utilization more robustly than methods that rely on raw attention or positively identified retrieval heads alone.

## H Efficiency Analysis

Implementation. Sentinel is implemented as a lightweight readout module attached to the standard SDPA prefill of Qwen2.5-0.5B-Instruct. During each transformer layer, the model performs a normal self-attention forward while Sentinel computes only the final-query attention row against all

<span id="page-13-0"></span>

| Method                    | LongI     | LongBench-En (2K Constraint) |       |        |           | LongBench-Zh (2K Constraint) |        |       |  |
|---------------------------|-----------|------------------------------|-------|--------|-----------|------------------------------|--------|-------|--|
|                           | SingleDoc | MultiDoc                     | Summ. | En-AVG | SingleDoc | MultiDoc                     | Zh-AVG | AVG   |  |
| Raw Attention (ratio 0.1) | 25.79     | 36.54                        | 20.39 | 27.57  | 35.03     | 16.33                        | 25.68  | 26.62 |  |
| Raw Attention (ratio 0.2) | 33.19     | 41.09                        | 21.63 | 31.97  | 48.45     | 17.23                        | 32.84  | 32.41 |  |
| Raw Attention (ratio 0.3) | 34.91     | 43.74                        | 22.39 | 33.68  | 55.09     | 18.14                        | 36.62  | 35.15 |  |
| Raw Attention (ratio 0.4) | 37.63     | 45.95                        | 22.88 | 35.49  | 58.78     | 17.82                        | 38.30  | 36.89 |  |
| Raw Attention (ratio 0.5) | 37.47     | 44.70                        | 23.25 | 35.14  | 60.63     | 17.42                        | 39.03  | 37.09 |  |
| Sentinel (ratio 0.1)      | 37.72     | 41.47                        | 22.58 | 33.93  | 58.96     | 19.36                        | 39.16  | 36.55 |  |
| Sentinel (ratio 0.2)      | 39.90     | 45.97                        | 23.37 | 36.42  | 59.50     | 17.92                        | 38.71  | 37.56 |  |
| Sentinel (ratio 0.3)      | 39.45     | 46.51                        | 23.86 | 36.61  | 60.98     | 18.68                        | 39.83  | 38.22 |  |
| Sentinel (ratio 0.4)      | 39.93     | 46.62                        | 23.38 | 36.65  | 59.51     | 18.77                        | 39.14  | 37.89 |  |
| Sentinel (ratio 0.5)      | 38.60     | 46.77                        | 23.54 | 36.30  | 61.41     | 18.44                        | 39.92  | 38.11 |  |

Table 7: Performance across different compression ratios with chunk size fixed at 1024 under a 2K-token context constraint. The **Summ.** column corresponds to query-conditioned summarization tasks (QMSum).

<span id="page-13-1"></span>

| Layer | Head | Probe Weight | Sink | Supporting | Question | Others |
|-------|------|--------------|------|------------|----------|--------|
| 11    | 1    | -13.16       | 0.89 | 0.01       | 0.05     | 0.04   |
| 3     | 0    | -12.83       | 0.74 | 0.01       | 0.18     | 0.03   |
| 3     | 10   | -10.22       | 0.08 | 0.00       | 0.84     | 0.02   |
| 21    | 9    | -9.95        | 0.01 | 0.00       | 0.98     | 0.01   |
| 14    | 5    | -9.47        | 0.00 | 0.03       | 0.85     | 0.06   |
| 3     | 5    | -9.11        | 0.74 | 0.04       | 0.03     | 0.18   |
| 9     | 11   | -8.15        | 0.96 | 0.00       | 0.03     | 0.01   |

Table 8: Examples of attention heads assigned strong negative weights by Sentinel, showing attention mass concentrated on sink or question tokens rather than supporting evidence.

keys, extracts context-only attention distributions, and incrementally accumulates sentence-level features. Probe features are streamed across layers without materializing full attention matrices. After the forward pass, sentence representations are scored by a lightweight logistic-regression probe implemented as a single GPU nn.Linear layer. Consequently, contextual utilization decoding and sentence selection remain entirely on-device and introduce negligible computation beyond the underlying model forward pass. All measurements are conducted on a single A800 80GB GPU with sequential processing (batch size 1). HuggingFace experiments use Transformers v4.50.2 and vLLM experiments use vLLM v0.18.0.

Efficiency. Table 9 shows that Sentinel introduces only modest overhead beyond a standard proxy-model prefill. Using a maximum input length of 10240 tokens, Sentinel requires 74.6 ms and 2323 MB of peak VRAM, compared to 46 ms and 2321 MB for a standard Qwen2.5-0.5B prefill. Profiling indicates that most of the additional latency originates from CPU-side preprocessing (e.g., sentence segmentation and context reconstruction), while the probe computation itself contributes only a small fraction of the end-to-end runtime.

Compared with generative compression, Sentinel is substantially more efficient. Under the same 10240-token input, generative compression is evaluated by generating a 2000-token compressed context, matching Sentinel's compression budget. HuggingFace Transformers requires 32.2–40.6 s and vLLM requires 23.4–27.4 s, whereas Sentinel completes contextual utilization decoding and sentence selection in only 74.6 ms, yielding an approximately 300–500× speedup. Unlike generative compressors, Sentinel directly ranks and selects sentences through a single forward pass without autoregressive decoding.

Sentinel also incurs negligible memory overhead, increasing peak VRAM by only 2 MB relative to standard prefilling. When the maximum input length is reduced to 1024 tokens, memory usage further decreases to 1268 MB.

Preserving Compression Effectiveness. Despite its lightweight implementation, Sentinel maintains strong compression effectiveness. Under the optimized deployment stack, Sentinel achieves a MultiFieldQA-Zh score of 60.88 using the default 1024-token input length and 62.48 when the maximum input length is extended to 10240 tokens without retraining the probe. Both results remain competitive with, and even slightly exceed, the 60.06 score obtained using the original uncompressed context reported in our main experiments. The higher score under the 10240-token setting is likely due to the elimination of chunking, allowing contextual utilization signals to be extracted from the entire retrieved context in a single forward pass.

Minor differences arise from implementationlevel variations between the optimized deployment stack and the experimental setup used in the

<span id="page-14-1"></span>

| Method                        | Model        | Input Length | Runtime  | VRAM (MB) | MultiFieldQA-Zh |
|-------------------------------|--------------|--------------|----------|-----------|-----------------|
| Prefill                       | Qwen2.5-0.5B | 10240        | 46 ms    | 2321      | –               |
| Prefill                       | Qwen2.5-7B   | 10240        | 334 ms   | 16157     | –               |
| Generative Compression (HF)   | Qwen2.5-0.5B | 10240        | 32.2 s   | 2329      | –               |
| Generative Compression (HF)   | Qwen2.5-7B   | 10240        | 40.6 s   | 16188     | –               |
| Generative Compression (vLLM) | Qwen2.5-0.5B | 10240        | 23.4 s   | –         | –               |
| Generative Compression (vLLM) | Qwen2.5-7B   | 10240        | 27.4 s   | –         | –               |
| Sentinel                      | Qwen2.5-0.5B | 10240        | 74.6 ms  | 2323      | 62.48           |
| Sentinel                      | Qwen2.5-0.5B | 1024         | 183.8 ms | 1268      | 60.88           |

Table 9: Efficiency comparison between Sentinel and generative compression on MultiFieldQA-Zh. Sentinel uses a frozen Qwen2.5-0.5B-Instruct proxy model and compresses retrieved contexts to a 2000-token budget. Generative compression constructs compressed contexts through autoregressive generation. Input Length denotes the maximum number of input tokens processed in a single forward pass. When the retrieved context exceeds this limit, Sentinel processes the document in multiple chunks and aggregates sentence-level features across chunks. When the retrieved context exceeds this limit, Sentinel processes the document in multiple chunks and aggregates sentence-level features across chunks. All measurements are obtained on a single A800 80GB GPU with sequential processing (batch size 1).

main evaluation, including the use of a lightweight rule-based Chinese sentence splitter instead of the spaCy-based pipeline. Since sentence boundaries define the compression units used by Sentinel, small segmentation differences can affect sentencelevel feature aggregation and selection. Overall, Sentinel preserves strong compression effectiveness while maintaining low latency and memory overhead.

## <span id="page-14-0"></span>I LLM Evaluation Settings

For LLM-based evaluation, we adopt the official prompt templates and decoding settings from Long-Bench [\(Bai et al.,](#page-8-8) [2024\)](#page-8-8) to ensure consistency and comparability across methods. Unless otherwise specified, all decoding parameters are fixed for all datasets: the temperature is set to 0.0, the nucleus sampling parameter top\_p is 1.0, the random seed is fixed to 42, only a single generation is sampled (n = 1), and streaming is disabled.