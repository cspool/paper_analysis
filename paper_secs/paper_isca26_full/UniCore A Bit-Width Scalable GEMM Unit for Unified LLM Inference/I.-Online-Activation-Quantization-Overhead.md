# I. Online Activation Quantization Overhead

Figure 22 reports the latency impact of online activation quantization. On Llama-2-7B, activation quantization accounts for 7.1%–20.7% of prefill latency but only 0.3%–1.6% during

<span id="page-12-2"></span>TABLE IV: WikiText-2 perplexity (PPL) of different methods in different group sizes (GS). The PPL of FP16 is 5.47.

| Llama-2-7B |        | UniCore | M-ANT | BitMoD | INT  |
|------------|--------|---------|-------|--------|------|
|            | GS-128 | 5.98    | 6.36  | 6.39   | 6.54 |
| W4A4       | GS-64  | 5.84    | 6.02  | 5.99   | 6.14 |
|            | GS-32  | 5.76    | 5.85  | 5.82   | 5.95 |

<span id="page-12-3"></span>![](_page_12_Figure_12.jpeg)

Fig. 22: Normalized Latency of online activation quantization compared with GEMM operations in W4A4KV4 configuration on Llama-2-7B and Compute/Memory ratio of the quantization kernel with or without crest factor (CF) calculation.

decode for sequence lengths 512–8192. Importantly, quantization can largely overlap with GEMM execution. While prior works do not explicitly evaluate online activation quantization, its cost is largely comparable across designs under the same quantization granularity and memory system. Figure 22 shows that UNICORE's CF computation raises arithmetic intensity from 0.63 to 0.87 (extra reductions, sqrt, and division), yet remains memory-bound with no measurable overhead.

