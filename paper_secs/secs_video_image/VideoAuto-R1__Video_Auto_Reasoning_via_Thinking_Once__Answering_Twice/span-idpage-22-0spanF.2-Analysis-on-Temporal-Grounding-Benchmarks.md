# <span id="page-22-0"></span>F.2 Analysis on Temporal Grounding Benchmarks

In the main paper, we emphasize that for grounding benchmarks, the initial answer is typically sufficient, so we exit early by default to save computation. In Table 16, we report the detailed grounding results when using the first boxed answer, the second boxed answer, and the confidence-based auto strategy.

Initial vs. Reviewed Answer. Unlike video QA benchmarks, temporal grounding shows almost no gap between the first and reviewed answers. For VideoAuto-R1, mIoU is the same for ActivityNet and NExT-GQA when comparing the first and second boxed answers. On NExT-GQA, the grounding QA accuracy also remains the same.

<span id="page-23-1"></span>Table 15 Evaluation Results on Video QA Benchmarks with Different Frames. For the Qwen2.5-VL models, we allow up to 16K total video tokens. For the Qwen3-VL models, we allow up to 128K total video tokens.

| Model                       | Frames | Video Perception Benchmark |      |                                      |      | Video Reasoning Benchmark |      |
|-----------------------------|--------|----------------------------|------|--------------------------------------|------|---------------------------|------|
|                             |        |                            |      | VideoMME MVBench LongVideoBench MMVU |      | VideoMMMU                 | MVP  |
| Qwen2.5-VL-7B               | 64     | 63.1                       | 67.0 | 59.7                                 | 66.2 | 54.6                      | 35.8 |
| Qwen2.5-VL-7B               | 128    | 65.9                       | 67.0 | 60.6                                 | 66.2 | 54.7                      | 35.8 |
| Qwen2.5-VL-7B               | 256    | 66.0                       | 67.1 | 60.9                                 | 65.7 | 52.7                      | 36.5 |
| VideoAuto-R1(Qwen2.5-VL-7B) | 64     | 64.6                       | 71.0 | 60.0                                 | 69.7 | 58.7                      | 39.2 |
| VideoAuto-R1(Qwen2.5-VL-7B) | 128    | 66.7                       | 71.0 | 60.4                                 | 69.1 | 56.6                      | 39.3 |
| VideoAuto-R1(Qwen2.5-VL-7B) | 256    | 67.3                       | 71.0 | 60.5                                 | 68.6 | 56.7                      | 39.4 |
| Qwen3-VL-8B                 | 64     | 67.3                       | 69.4 | 63.4                                 | 69.9 | 61.0                      | 40.4 |
| Qwen3-VL-8B                 | 256    | 70.9                       | 69.4 | 66.0                                 | 69.6 | 59.9                      | 40.5 |
| Qwen3-VL-8B                 | 2048   | 72.5                       | 69.4 | 67.6                                 | 69.9 | 59.8                      | 40.5 |
| VideoAuto-R1(Qwen3-VL-8B)   | 64     | 67.9                       | 71.8 | 63.9                                 | 71.0 | 65.0                      | 42.7 |
| VideoAuto-R1(Qwen3-VL-8B)   | 256    | 70.4                       | 72.0 | 67.1                                 | 71.0 | 63.8                      | 42.9 |
| VideoAuto-R1(Qwen3-VL-8B)   | 2048   | 71.7                       | 72.0 | 67.4                                 | 71.1 | 64.0                      | 43.0 |

<span id="page-23-2"></span>Table 16 Comparison of Different Inference Strategies on Temporal Grounding Benchmarks. We compare the results using the first boxed answer, the second boxed answer, or the confidence-based early-exit answer. We observe that on grounding benchmark, the first boxed answer is typically sufficient, so we early-exit without further reasoning to save computation.

| Model                           | Inference Strategy | ActivityNet |      |      |      | NExT-GQA |      |
|---------------------------------|--------------------|-------------|------|------|------|----------|------|
|                                 |                    | 0.3         | 0.5  | 0.7  | mIoU | Acc      | mIoU |
| VideoAuto-R1<br>(Qwen2.5-VL-7B) | First Answer       | 69.2        | 48.5 | 27.3 | 47.6 | 80.6     | 36.7 |
|                                 | Second Answer      | 69.2        | 48.5 | 27.3 | 47.6 | 80.6     | 36.7 |
|                                 | Auto               | 69.2        | 48.5 | 27.3 | 47.6 | 80.6     | 36.7 |

We hypothesize two reasons for this phenomenon. First, since the grounding procedure does not require multi-step logical deduction, the model can map the queried event to a time span directly from perception. Once the model has localized a segment in the first answer, additional textual reasoning has limited room to further improve the IoU. Second, since we lack the SFT stage to teach the model how to explicitly reason on the grounding task, the model cannot easily refine the predicted segments. Consequently, the reasoning stage rarely corrects localization errors, leading to nearly identical scores. In practice, this suggests that for grounding tasks, RL still shows significant improvements compared to baseline or SFT, but it is often unnecessary to rely on long and language-based thinking rationales.

Reasoning Traces on QA vs. Grounding. To better understand this behavior, we examine representative reasoning traces of VideoAuto-R1 between grounding and QA tasks, as shown in Figure [9,](#page-28-0) [10,](#page-29-0) and [4.](#page-12-0) On video QA benchmarks, the thinking rationale usually contains multi-step analysis: enumerating visual evidence, performing arithmetic, or checking answer options. In contrast, grounding traces are much shorter. The model typically identifies the relevant event or shot, notes when it appears and disappears in the video, and then outputs the corresponding timestamps or intervals.

These qualitative observations align with the quantitative results in Table [4:](#page-8-1) for temporal grounding benchmarks, explicit reasoning provides limited additional benefit over the direct localization. Therefore, we use the direct answering results on grounding benchmarks for VideoAuto-R1.

### <span id="page-23-0"></span>F.3 Analysis of the Impact of Cold-Start SFT

In our training framework, we deliberately omit chain-of-thought SFT and proceed directly to RL. Traditionally, SFT is used to (1) teach the CoT output format, (2) imitate the CoT reasoning process, and (3) acquire

general knowledge from newly collected data. However, with modern base models that are already trained on massive corpora, the marginal benefit for (1) and (3) is limited. Moreover, collecting large-scale, high-quality CoT traces for (2) is expensive and often noisy.

<span id="page-24-2"></span>Table 17 Ablation on Cold-Start CoT SFT.

| Setting                    | VideoMME | MVBench | VideoMMMU |
|----------------------------|----------|---------|-----------|
| Qwen2.5-VL baseline        | 66.0     | 67.1    | 54.7      |
| SFT with Video-R1-CoT data | 60.1     | 64.0    | 53.8      |
| RL with thinking           | 66.1     | 71.2    | 56.4      |
| SFT → RL with thinking     | 61.7     | 64.3    | 53.5      |

In early experiments, SFT on Video-R1-CoT data [\(Feng et al.,](#page-13-2) [2025\)](#page-13-2), which has both the intermediate reasoning traces and final answer, not only failed to improve performance, but actually degraded the Qwen2.5- VL baseline, a phenomenon also observed in prior work [\(Li et al.,](#page-14-2) [2025e;](#page-14-2) [Chen et al.,](#page-13-16) [2025a\)](#page-13-16). Table [17](#page-24-2) summarizes this effect. Pure SFT substantially hurts performance across all three benchmarks. When we apply GRPO starting from the SFT checkpoint ("SFT → RL with thinking"), the final model remains significantly worse than RL applied directly on the base model.

These results suggest that low-quality CoT supervision can distort the behavior of a strong base model and create a poor initialization for RL. We therefore focus on directly incentivizing the base model's reasoning via GRPO-style reinforcement learning.

