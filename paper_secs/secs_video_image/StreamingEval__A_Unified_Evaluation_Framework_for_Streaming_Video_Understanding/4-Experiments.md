# 4 Experiments

We evaluate 12 mainstream multimodal and online models under the StreamingEval framework. We first describe the experimental setup and datasets, then report overall results, followed by analyses under different memory\_bank budgets and input image resolution ranges, and finally summarize the key findings.

## 4.1 Experiment Settings

We evaluate representative multimodal models and online video models within the StreamingEval framework. All experiments are conducted on a single RTX 4090 (48GB) GPU with BF16 inference, using a unified prompt and decoding configuration. TTFT is measured in wall-clock time. Streaming inputs are fed at 1fps. Due to space limitations, more experimental setups are shown in the Appendix [A.2.](#page-11-0)

