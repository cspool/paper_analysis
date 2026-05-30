# C Single-Stage Training vs. Two-Stage Training

The research conducted by [Karamcheti et al.](#page-9-21) [\(2024\)](#page-9-21), suggests that including a distinct projector pretraining phase may not be essential. Their study indicates that a single-stage training process can lead to improved performance outcomes. Omission of the pretraining phase results in a significant cost reduction of about 20 to 25 percent and avoids the need for stage-specific data collection.

To validate these insights, we conducted a series of experiments using the VisualRWKV framework. The results, as illustrated in Figure [5,](#page-14-0) show that the two-stage training outperforms single-stage training, indicating that the two-stage approach is still very necessary. The different results associated with single-stage training could be due to the diverse training setups used by various researchers. Given these results, we have made a strategic decision to adopt a two-stage training protocol for all subsequent experiments in this paper.

