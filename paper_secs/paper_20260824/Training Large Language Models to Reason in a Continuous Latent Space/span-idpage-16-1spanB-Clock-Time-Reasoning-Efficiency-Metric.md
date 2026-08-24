# <span id="page-16-1"></span>B Clock-Time Reasoning Efficiency Metric

We present a clock-time comparison to evaluate reasoning efficiency. The reported values represent the average inference time per test case (in seconds), with a batch size of 1, measured on an Nvidia A100 GPU. For the no-CoT and CoT baselines, we employ the standard generate method from the transformers[3](#page-16-2) library. Our results show that clock time is generally proportional to the number of newly generated tokens, as detailed in Table [1.](#page-8-0)

| Method  | GSM8k | ProntoQA | ProsQA |
|---------|-------|----------|--------|
| No-CoT  | 0.03  | 0.03     | 0.08   |
| CoT     | 0.26  | 0.85     | 0.47   |
| Coconut | 0.09  | 0.11     | 0.15   |

Table 4 Inference time (in seconds) comparison across tasks and methods.

