# <span id="page-13-1"></span>A.3 COMPARISON WITH MLC-LLM RESULTS

We compare the results of SnapKV based self-speculation on MLC-LLM and our backend. As the measurement methods are different, we put them in two tables as shown in Table [4](#page-13-2) and [5.](#page-14-2) The verification time of MLC-LLM includes one step of draft decode time. Our backend is highly optimized for speculative decoding setting, minimizing the drafting and verification overhead, thus, leading to better speedup. However, the trend that speedup increases with batch size is the same, aligning with our theoretical analysis in Section [3.](#page-2-1)

Table 4: Results of Our Backend

<span id="page-13-2"></span>

| Target      | Backend | Task  | GPU    | Prefill | Bsz | γ | γTD(1) | TV(γ) | Ω(γ,α) | TAR   | TSD  | x    |
|-------------|---------|-------|--------|---------|-----|---|--------|-------|--------|-------|------|------|
| Llama3.1-8B | Ours    | PG-19 | 8xH100 | 32000   | 16  | 3 | 10.96  | 6.91  | 3.42   | 6.41  | 5.41 | 1.18 |
| Llama3.1-8B | Ours    | PG-19 | 8xH100 | 32000   | 32  | 4 | 16.69  | 10.39 | 4.10   | 9.23  | 6.75 | 1.37 |
| Llama3.1-8B | Ours    | PG-19 | 8xH100 | 32000   | 64  | 5 | 23.96  | 17.45 | 4.59   | 14.85 | 9.17 | 1.62 |

Table 5: Results of MLC-LLM

<span id="page-14-2"></span>

| Target      | Backend | Task  | GPU    | Prefill | Bsz | γ | TD(1) | TV(γ) | NumGen | ARTrput | SDTrput | x    |
|-------------|---------|-------|--------|---------|-----|---|-------|-------|--------|---------|---------|------|
| Llama3.1-8B | MLC-LLM | PG-19 | 8xH100 | 32000   | 16  | 4 | 3.64  | 13.60 | 724    | 2471.4  | 2133.0  | 0.86 |
| Llama3.1-8B | MLC-LLM | PG-19 | 8xH100 | 32000   | 32  | 4 | 4.19  | 16.13 | 1455   | 3311.5  | 3664.5  | 1.11 |
| Llama3.1-8B | MLC-LLM | PG-19 | 8xH100 | 32000   | 64  | 5 | 5.27  | 28.26 | 2719   | 3930.0  | 4959.2  | 1.26 |

### A.4 FURTHER SNAPKV AND STREAMINGLLM RESULTS

We show the raw experiment data. We compare both StreamingLLM-based self-speculation and SnapKV-based self-speculation, and also a small draft model with StreamingLLM KV cache.

<span id="page-14-0"></span>Table 6: Comparison of SnapKV, StreamingLLM, and Tiny Draft (StreamingLLM KV) Speculation. Each with optimal γ and KV budget

| Target      | Draft          | Task  | GPU    | Prefill | Bsz | γ | γTD(1) | TV(γ) | Ω(γ,α) | TAR   | TSD   | x    |
|-------------|----------------|-------|--------|---------|-----|---|--------|-------|--------|-------|-------|------|
| Llama3.1-8B | Llama3.2-1B(S) | PG-19 | 8xH100 | 32000   | 16  | 3 | 4.43   | 6.71  | 2.43   | 6.18  | 4.86  | 1.27 |
| Llama3.1-8B | StreamingLLM   | PG-19 | 8xH100 | 32000   | 16  | 3 | 10.33  | 6.73  | 3.09   | 6.18  | 5.74  | 1.08 |
| Llama3.1-8B | SnapKV         | PG-19 | 8xH100 | 32000   | 16  | 3 | 10.55  | 6.84  | 3.41   | 6.18  | 5.27  | 1.17 |
| Llama3.1-8B | Llama3.2-1B(S) | PG-19 | 8xH100 | 32000   | 32  | 3 | 4.71   | 9.70  | 2.43   | 9.10  | 6.22  | 1.46 |
| Llama3.1-8B | StreamingLLM   | PG-19 | 8xH100 | 32000   | 32  | 3 | 11.55  | 9.74  | 3.06   | 9.10  | 7.20  | 1.26 |
| Llama3.1-8B | SnapKV         | PG-19 | 8xH100 | 32000   | 32  | 4 | 15.79  | 10.36 | 4.03   | 9.10  | 6.64  | 1.37 |
| Llama3.1-8B | Llama3.2-1B(S) | PG-19 | 8xH100 | 32000   | 64  | 3 | 5.05   | 15.86 | 2.44   | 14.84 | 8.88  | 1.67 |
| Llama3.1-8B | StreamingLLM   | PG-19 | 8xH100 | 32000   | 64  | 3 | 12.82  | 15.93 | 3.08   | 14.84 | 9.57  | 1.55 |
| Llama3.1-8B | SnapKV         | PG-19 | 8xH100 | 32000   | 64  | 5 | 22.91  | 17.70 | 4.55   | 14.84 | 9.05  | 1.64 |
| Llama3.1-8B | Llama3.2-1B(S) | PG-19 | 8xH100 | 32000   | 128 | 3 | 5.79   | 28.51 | 2.43   | 26.07 | 14.43 | 1.81 |
| Llama3.1-8B | StreamingLLM   | PG-19 | 8xH100 | 32000   | 128 | 4 | 18.96  | 30.34 | 3.57   | 26.07 | 14.06 | 1.85 |
| Llama3.1-8B | SnapKV         | PG-19 | 8xH100 | 32000   | 128 | 6 | 33.33  | 31.60 | 5.07   | 26.07 | 12.96 | 2.01 |

### <span id="page-14-1"></span>A.5 RESULTS OF QWEN AND MISTRAL MODELS

Table 7: Results of Qwen and Mistral Models. Each with optimal γ and KV budget

| Target          | Draft                  | Task  | GPU    | Prefill | Bsz | γ | γTD(1) | TV(γ) | Ω(γ,α) | TAR   | TSD   | x    |
|-----------------|------------------------|-------|--------|---------|-----|---|--------|-------|--------|-------|-------|------|
| Mistral-7B-v0.3 | SnapKV                 | PG-19 | 8xH100 | 32000   | 32  | 3 | 11.71  | 9.62  | 3.49   | 8.92  | 6.12  | 1.46 |
| Mistral-7B-v0.3 | SnapKV                 | PG-19 | 8xH100 | 32000   | 64  | 3 | 13.64  | 15.64 | 3.47   | 14.49 | 8.44  | 1.72 |
| Mistral-7B-v0.3 | SnapKV                 | PG-19 | 8xH100 | 32000   | 128 | 5 | 27.49  | 30.65 | 4.72   | 25.41 | 12.31 | 2.06 |
| Qwen-2.5-7B     | SnapKV                 | PG-19 | 4xH100 | 32000   | 32  | 3 | 11.40  | 9.26  | 3.40   | 8.20  | 6.07  | 1.35 |
| Qwen-2.5-7B     | SnapKV                 | PG-19 | 4xH100 | 32000   | 64  | 4 | 17.67  | 15.67 | 4.06   | 13.11 | 8.20  | 1.6  |
| Qwen-2.5-7B     | SnapKV                 | PG-19 | 4xH100 | 32000   | 128 | 5 | 27.22  | 28.51 | 4.62   | 22.79 | 12.06 | 1.89 |
| Qwen-2.5-32B    | SnapKV                 | PG-19 | 8xH100 | 32000   | 8   | 3 | 23.67  | 11.98 | 3.50   | 10.42 | 10.19 | 1.02 |
| Qwen-2.5-32B    | SnapKV                 | PG-19 | 8xH100 | 32000   | 16  | 3 | 25.27  | 15.29 | 3.52   | 13.36 | 11.52 | 1.16 |
| Qwen-2.5-32B    | SnapKV                 | PG-19 | 8xH100 | 32000   | 32  | 3 | 28.99  | 21.90 | 3.51   | 19.43 | 14.49 | 1.34 |
| Qwen-2.5-32B    | Qwen-2.5-7B            | PG-19 | 8xH100 | 32000   | 8   | 2 | 9.04   | 11.31 | 2.32   | 10.42 | 8.74  | 1.19 |
| Qwen-2.5-32B    | Qwen-2.5-7B            | PG-19 | 8xH100 | 32000   | 16  | 2 | 11.61  | 14.59 | 2.32   | 13.36 | 11.31 | 1.18 |
| Qwen-2.5-32B    | Qwen-2.5-7B            | PG-19 | 8xH100 | 32000   | 32  | 2 | 16.72  | 20.87 | 2.31   | 19.43 | 16.27 | 1.19 |
| Qwen-2.5-32B    | Qwen-2.5-7B(Streaming) | PG-19 | 8xH100 | 32000   | 8   | 2 | 6.77   | 11.31 | 2.27   | 10.42 | 7.97  | 1.31 |
| Qwen-2.5-32B    | Qwen-2.5-7B(Streaming) | PG-19 | 8xH100 | 32000   | 16  | 2 | 7.21   | 14.59 | 2.26   | 13.36 | 9.64  | 1.39 |
| Qwen-2.5-32B    | Qwen-2.5-7B(Streaming) | PG-19 | 8xH100 | 32000   | 32  | 3 | 11.78  | 21.82 | 2.62   | 19.43 | 12.85 | 1.51 |

