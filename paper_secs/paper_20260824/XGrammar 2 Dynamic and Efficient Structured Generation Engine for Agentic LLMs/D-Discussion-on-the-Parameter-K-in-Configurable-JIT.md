# D Discussion on the Parameter K in Configurable JIT

end function

The parameter *K* depends on both the GPU and the CPU. Tuning it with elaboration can improve the efficiency and stability. We swept *K* under the setup described in Section 4.2, using Llama's tool-calling format. The results are summarized in Table 5.

<span id="page-11-2"></span>

| 𝐾  | Compilation | Avg. TPOM | Max TPOM | P99 TPOM |
|----|-------------|-----------|----------|----------|
| 0  | 14.45 ms    | 12.76 𝜇s  | 76.08 𝜇s | 48.15 𝜇s |
| 5  | 18.24 ms    | 12.80 𝜇s  | 74.78 𝜇s | 44.59 𝜇s |
| 10 | 20.07 ms    | 12.68 𝜇s  | 67.80 𝜇s | 42.49 𝜇s |

Table 5: Effect of on compilation time and TPOM metrics.

Across this sweep, average TPOM is almost unchanged while compilation time increases with . P99 and max TPOM decrease from =0 to =10, indicating a trade-off between compilation cost and tail latency.

