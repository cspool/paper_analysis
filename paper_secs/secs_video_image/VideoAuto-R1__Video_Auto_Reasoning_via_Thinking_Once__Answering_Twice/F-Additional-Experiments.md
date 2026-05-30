# F Additional Experiments

In this section, we present additional experiments and analyses to complement the findings reported in the main paper.

#### F.1 Performance with Different Frames

In the main paper, we report the best-performing configurations of our model. Here, we present the complete results in Table 15 and analyze how the number of input frames affects performance on both perception and reasoning benchmarks.

Under a 16K video-token budget using Qwen2.5-VL, increasing the number of frames from 64 to 256 yields noticeable improvements on most perception benchmarks for both the Qwen baseline and VideoAuto-R1. For example, accuracy on VideoMME improves from 63.1% to 66.0%, and on LongVideoBench from 59.7% to 60.9%. However, the reasoning-oriented benchmark VideoMMMU shows weaker dependence on frame count, where performance slightly decreases with additional frames. This trend persists when switching to Qwen3-VL, which supports a larger 128K video-token budget and up to 2,048 frames.

Moreover, VideoAuto-R1 achieves consistent improvements compared to the Qwen baseline. For instance, even under a 64-frame budget, VideoAuto-R1 improves upon the baseline performance from 63.1% to 64.6% on VideoMME, and from 66.2% to 69.7% on MMVU, demonstrating the effectiveness of our proposed approach across both low and high frame regimes.

