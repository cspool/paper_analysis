# *A.6 Evaluation and Expected Results*

Case Study 1 reproduces Figure [12.](#page-9-2) The simulator is deterministic, so generated results should match the reported trends when using the same traces and configuration files.

Case Study 2 reproduces Figure [17.](#page-12-0) Because it measures real GPU execution, small timing variations are expected from thermals, system load, NCCL non-determinism, and SGLang micro-batching. In our runs, variation is typically within ±5%. The high-level result is stable: prefill-aware placement improves MoE kernel performance by about 5–25% over the default placement.

