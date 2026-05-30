# C Additional sliding window size generalization on LongBench

The sequence lengths of LongBench tasks are substantially shorter than those in LV-Eval and InfiniteBench. We therefore select six relatively long tasks from LongBench, whose average sequence lengths range from 8k to 18k. To evaluate the context-length generalization ability of AHN-augmented models on these tasks, we fix the attention-sink size to 128 tokens and vary the sliding-window size from 896 to 8064. We compare AHN-augmented models against both Sliding Window Attention (SWA) and Compressive Transformers using average pooling (CT-Average). As shown in Figure 7, AHN-augmented models consistently outperform these baselines across different inference window sizes.

