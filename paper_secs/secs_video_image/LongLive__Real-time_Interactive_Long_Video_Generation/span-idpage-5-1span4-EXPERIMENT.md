# <span id="page-5-1"></span>4 EXPERIMENT

Implementation We build LONGLIVE on Wan2.1-T2V-1.3B [\(Wan et al.,](#page-11-0) [2025\)](#page-11-0), which produces 5s clips at 16 FPS and 832 × 480 resolution. We first adapt the pretrained model into a few-step causal-attention model using a self-forcing [\(Huang et al.,](#page-9-2) [2025\)](#page-9-2) DMD pipeline on VidProM [\(Wang](#page-11-7) [& Yang,](#page-11-7) [2024\)](#page-11-7) data, while enabling our short-window attention and the frame sink (we keep all tokens from the first frame chunk as sink tokens). We then perform streaming long tuning on a 60s sequence that contains a single prompt switch. To construct this switch-prompt dataset, we prompt Qwen2-72B-Instruct [\(Yang et al.,](#page-11-8) [2024a\)](#page-11-8) to generate follow-up prompts conditioned on each original VidProM prompt. During training, each iteration continues the model's own rollout by generating the next 5s video clip until a maximum length of 60s is reached; each batch includes exactly one prompt switch with the switch time sampled uniformly from 5s to 55s. When a switch occurs, we apply KV-recache. During streaming long tuning, we also keep the same short-window attention

<span id="page-6-0"></span>Table 1: **Comparison with relevant baselines.** We compare LONGLIVE with representative open-source video generation models of similar parameter sizes and resolutions. Evaluation scores are calculated on the standard prompt suite of VBench (Huang et al., 2024a). FPS - a single H100 GPU.

| Model                                         | #Params     | Resolution       | Throughput (FPS) ↑ | Evaluation scores ↑ |         |          |
|-----------------------------------------------|-------------|------------------|--------------------|---------------------|---------|----------|
|                                               | ni di di di | resolution       |                    | Total               | Quality | Semantic |
| Diffusion models                              |             |                  |                    |                     |         |          |
| LTX-Video (HaCohen et al., 2025)              | 1.9B        | $768 \times 512$ | 8.98               | 80.00               | 82.30   | 70.79    |
| Wan2.1 (Wan et al., 2025)                     | 1.3B        | $832{\times}480$ | 0.78               | 84.26               | 85.30   | 80.09    |
| Autoregressive models                         |             |                  |                    |                     |         |          |
| SkyReels-V2 (Chen et al., 2025a)              | 1.3B        | $960 \times 540$ | 0.49               | 82.67               | 84.70   | 74.53    |
| MAGI-1 (Teng et al., 2025)                    | 4.5B        | $832 \times 480$ | 0.19               | 79.18               | 82.04   | 67.74    |
| CausVid (Yin et al., 2025)                    | 1.3B        | $832 \times 480$ | 17.0               | 81.20               | 84.05   | 69.80    |
| NOVA (Deng et al., 2025)                      | 0.6B        | $768 \times 480$ | 0.88               | 80.12               | 80.39   | 79.05    |
| Pyramid Flow (Jin et al., 2025)               | 2B          | $640 \times 384$ | 6.7                | 81.72               | 84.74   | 69.62    |
| Self Forcing, chunk-wise (Huang et al., 2025) | 1.3B        | $832 \times 480$ | 17.0               | 84.31               | 85.07   | 81.28    |
| Self Forcing, frame-wise (Huang et al., 2025) | 1.3B        | $832{\times}480$ | 8.9                | 84.26               | 85.25   | 80.30    |
| LongLive                                      | 1.3B        | 832×480          | 20.7               | 84.87               | 86.97   | 76.47    |

<span id="page-6-1"></span>Table 2: Interactive long video evaluation: Quality scores are reported on the whole 60s sequence. CLIP scores are reported on 10s video segments with the same semantics ( $\uparrow$  higher is better).

| Method                            | Quality | CLIP Score ↑ |         |         |         |         |         |
|-----------------------------------|---------|--------------|---------|---------|---------|---------|---------|
|                                   | Score ↑ | 0–10 s       | 10-20 s | 20–30 s | 30–40 s | 40–50 s | 50–60 s |
| SkyReels-V2 (Chen et al., 2025a)  | 80.49   | 20.96        | 22.51   | 25.78   | 18.45   | 19.57   | 19.61   |
| Self-Forcing (Huang et al., 2025) | 82.46   | 28.46        | 24.89   | 23.53   | 22.96   | 23.07   | 23.19   |
| LongLive                          | 84.38   | 28.85        | 25.68   | 24.64   | 24.23   | 24.32   | 24.32   |

and frame-sink settings. This training procedure takes about 12 hours on 64 H100 GPUs. Notably, LONGLIVE supports any model capable of autoregressive rollout with a KV cache. We implement LONGLIVE on a linear-attention AR model, SANA-Video (Chen et al., 2025b), achieving further acceleration on long-video generation.

#### 4.1 SHORT VIDEO GENERATION

We first evaluate LONGLIVE's short-video generation on VBench using their official prompts, and compared it with relevant open-source video generation models of similar scale, including LTX-Video (HaCohen et al., 2025), Wan2.1 (Wan et al., 2025), SkyReels-V2 (Chen et al., 2025a), MAGI-1 (Teng et al., 2025), CausVid (Yin et al., 2025), NOVA (Deng et al., 2025), Pyramid Flow (Jin et al., 2025), and Self-forcing (Huang et al., 2025). All scores are normalized using the same numerical system with VBench. On 5-second clips, LONGLIVE matches the strongest baselines in total score, demonstrating excellent quality and stability, as shown in Table 1. Benefiting from the short window attention design, LONGLIVE is also the fastest among all the methods, reaching 20.7 FPS for real-time inference. It shows that LONGLIVE does not degrade the short-clip generation capability.

#### 4.2 Long Video Generation

We evaluate LONGLIVE's single-prompt long-video generation on VBench-Long (Huang et al., 2024b) using its official prompt set. For each prompt, we generate a 30-second video and split it into clips according to the VBench-Long official scripts. We compare against three representative open-source models: SkyReels-V2 (Chen et al., 2025a), FramePack (Zhang & Agrawala, 2025), and Self-Forcing (Huang et al., 2025). Because FramePack is an I2V model, we first synthesize an initial frame from the same text prompt and feed it to FramePack; other T2V models generate directly from the prompt. We report the standard VBench-Long metrics for long-horizon quality and consistency in Table 3. LONGLIVE achieve the state-of-the-art performance, while being the fastest.

<span id="page-7-1"></span>![](_page_7_Figure_0.jpeg)

0-10s: Medium close-up of the serene model in a white gown amid drifting sakura petals and soft pink smoke.

10-20s: She slowly lifts her hand, fingertips grazing a petal/soft pink smoke

20-30s: A gentle breeze rises. The petals float softly in the air, creating a dreamy and ethereal atmosphere.

30-40s: She softly closes her eyes while holding the same poised gesture.

40-50s: Her fingers barely touching the delicate pink smoke, while a small bird flits in.

50-60s: the bird now perches on her outstretched finger; pink and cyan-blue smoke thickens and envelops the scene.

Figure 6: Qualitative comparison for interactive long video generation. LONGLIVE exhibits strong prompt compliance, smooth transitions, and high long-range consistency while sustaining high throughput. Compared to ours, SkyReels-V2 shows weaker long-range consistency, and Self-Forcing faces quality degradation on longer videos.

<span id="page-7-0"></span>Table 3: Single-prompt 30s long video evaluation on VBench-Long.

| Model        | Total<br>Score ↑ | Quality<br>Score ↑ | Semantic<br>Score ↑ | Throughput<br>(FPS) ↑ |
|--------------|------------------|--------------------|---------------------|-----------------------|
| SkyReels-V2  | 75.29            | 80.77              | 53.37               | 0.49                  |
| FramePack    | 81.95            | 83.61              | 75.32               | 0.92                  |
| Self-Forcing | 81.59            | 83.82              | 72.70               | 17.0                  |
| LONGLIVE     | 83.52            | 85.44              | 75.82               | 20.7                  |

Table 4: Ablation study on KV recache. KV recache achieves the best consistency score and CLIP score.

| Method      | Background<br>Consistency ↑ | Subject<br>Consistency ↑ | CLIP<br>Score ↑ |  |
|-------------|-----------------------------|--------------------------|-----------------|--|
| No KV cache | 92.75                       | 89.59                    | 28.95           |  |
| KV cache    | 94.77                       | 93.69                    | 25.92           |  |
| KV recache  | 94.81                       | 94.04                    | 27.87           |  |

#### 4.3 INTERACTIVE LONG VIDEO GENERATION

For interactive long-form videos with multiple prompt switches, few existing methods support true streaming generation. We implemented this setting for two representative baselines: SkyReels-V2 and Self-Forcing. We then compare our approach against them. Because the standard VBench protocol is not directly applicable, we curated a custom set of 160 interactive 60-second videos, each comprising six successive 10-second prompts as the validation set. For long-horizon quality, we evaluate our 60s interactive videos on VBench-Long dimensions that support customized prompt videos, including subject consistency, background consistency, motion smoothness, aesthetic quality, and imaging quality. For semantic adherence, we segment each video at prompt boundaries and compute clip-wise semantic score using CLIP [\(Radford et al.,](#page-10-10) [2021\)](#page-10-10) scores. Qualitative and quantitative results are shown in Figure [6](#page-7-1) and Table [2,](#page-6-1) respectively. LONGLIVE exhibits strong prompt compliance, smooth transitions, and high long-range consistency while sustaining high throughput. In contrast, Self-Forcing degrades on longer horizons and, SkyReels-v2 shows weaker consistency. In terms of speed, LONGLIVE is more than 41× faster than SkyReels-v2 and slightly faster than Self-Forcing, even with KV re-cache, thanks to our short-window attention design. Please see our project page for more qualitative comparisons for interactive long video generation. Finally, a user study in which participants rated Overall Quality, Motion Quality, Instruction Following, and Visual Quality, *i.e.*, Figure [1](#page-0-0) (right) further supports the effectiveness of our approach.

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 7: Ablation study on short window size and frame sink. Smaller windows reduce consistency, while enabling frame sink mitigates the drop.

#### 4.4 KV RECACHE

In Table [4,](#page-7-0) we ablate KV caching strategies at prompt switches in a 10-second video setting with a single switch at the 5-second. We compare (i) No KV cache: clear the entire cache at the switch; (ii) KV-cache: retain the full cache unchanged; and (iii) KV-recache (ours): refresh the cache by recomputing key–value states conditioned on the preceding frames and the new prompt. We assess visual consistency with VBench Background Consistency and Subject Consistency, and measure semantic score with the CLIP model. Clearing the cache breaks long-range consistency, causing abrupt visual changes. Retaining the cache preserves continuity but induces prompt inertia: the model sticks to the previous prompt, yielding a lower semantic score on the switched prompt. Our KV recache maintains continuity while restoring compliance to the switched prompt. Please see Figure [3,](#page-3-0) Appendix Figure [D,](#page-20-0) and the demo page for more qualitative comparisons on KV recache.

#### 4.5 SHORT-WINDOW ATTENTION AND FRAME SINK

In Figure [7,](#page-8-0) we ablate short-window attention and the frame-sink under a 10-second generation setting. We vary the local-attention window from 3 to 27 latent frames, and additionally evaluate a configuration with 9 local latent frames plus 3 sink latent frames (effective window size 12). Long-range consistency is measured using VBench-Long [\(Huang et al.,](#page-10-9) [2024b\)](#page-10-9) (Background Consistency and Subject Consistency). Consistency improves as the attention window grows and saturates around a 24-frame window, revealing a clear quality–efficiency trade-off: larger windows retain more temporal context but increase latency and memory, while smaller windows are cheaper but less consistent. Our frame-sink mechanism mitigates this trade-off by recovering long-range context without attending to the full history: the 9-local + 3-sink setting achieves consistency close to a 21-frame window while preserving the speed and memory footprint of a short window.

#### 5 CONCLUSION

In this work, we introduce LONGLIVE, a frame-level AR framework for real-time and interactive long video generation. To maintain visual smoothness and semantic adherence during prompt switches in interactive settings, we propose a KV-recache technique. We present a streaming long tuning strategy that enables direct training on long videos, ensuring high-quality outputs. We further introduce short window attention and frame sink to accelerate long video generation while preserving visual consistency. Experimental results demonstrate that LONGLIVE can efficiently fine-tune a model for long-video AR generation in only 32 GPU-days. Moreover, tuning on long videos is essential not only for long video generation but also as a prerequisite for efficient inference (e.g., window attention with frame attention sink), substantially improving inference speed. During inference, it achieves 20.7 FPS inference on a single NVIDIA H100 GPU, and supports up to 240-second video generation while maintaining high fidelity and temporal coherence. Using INT8 quantization, LONGLIVE compresses from 2.7 GB to 1.4 GB, with minimal performance degradation. LONGLIVE also supports INT8-quantized inference, incurring only marginal quality loss. We provide further results, analyses, implementation details, and qualitative showcases in the Appendix.

