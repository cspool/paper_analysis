# *E. Latency Predictor Accuracy and SLO Safety*

Across our experimental settings, we measured the accuracy of the latency predictor for unseen kernel configurations that are predicted via wave scaling. The average misprediction is 3.9%, translating to 4.55 µs for prefill kernels and 0.84 µs for decode kernels, against average runtimes of 118.75 µs and 16 µs, respectively. This margin is negligible, and combined with the governor's continuous monitoring, it is sufficient to ensure zero SLO violations across all evaluated configurations.

For new kernels without a matching donor kernel, Power-Weave conservatively uses maximum frequency until their runtime contribution is assessed. Such kernels account for 1.9% of total runtime on average, below the 5% re-profiling threshold, so re-profiling was never triggered in our experiments.

As a result, PowerWeave maintains zero SLO violations across all evaluated workloads and load conditions. Moreover, PowerWeave uses continuous batching, which means that even as the prefill-to-decode ratio changes over time, the workload composition shifts gradually, giving PowerWeave's live weight adaptation sufficient time to adapt its frequencies.

