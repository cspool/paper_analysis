# 4 Empirical Evaluation

In Section 4.1 we test Mamba's ability to solve the two synthetic tasks motivated in Section 3.1. We then evaluate on three domains, each evaluated on autoregressive pretraining as well as downstream tasks.

- Section 4.2: language model pretraining (scaling laws), and zero-shot downstream evaluation.
- Section 4.3: DNA sequence pretraining, and fine-tuning on a long-sequence classification task.
- Section 4.4: audio waveform pretraining, and the quality of autoregressively generated speech clips.

Finally, Section 4.5 shows Mamba's computational efficiency at both training and inference time, and Section 4.6 ablates various components of the architecture and selective SSMs.

