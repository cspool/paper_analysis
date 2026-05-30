# <span id="page-13-1"></span><span id="page-13-0"></span>A. MORE EXPERIMENTS

#### A.1. Supplementary Ablation Experiments

<span id="page-13-2"></span>Ablation Studies on the Scaling Factor δ for ATS. We conduct a series of experiments to further investigate the impact of the temporal scaling factor δ on the alignment between video and text representations. Accurate temporal alignment plays a vital role in enhancing the model's understanding of both semantic and sequential aspects of video-language data. To this end, we evaluate the model's performance across three representative video-language benchmarks—LongVideoBench, MLVU, and VideoMME—by varying the temporal scaling factor δ from 0.5 to 3.0. As shown in Table [6,](#page-13-2) we observe a consistent trend across all benchmarks: performance improves as δ increases, peaking at δ = 2 with an average score of 60.92. These results suggest that setting δ = 2 strikes the best balance between temporal resolution and semantic alignment, resulting in optimal overall performance.

Table 6. Performance under different scaling factors δ across multiple benchmarks.

| Scaling Factor δ | LongVideoBench | MLVU  | VideoMME | Avg   |
|------------------|----------------|-------|----------|-------|
| 0.5              | 50.83          | 59.87 | 58.33    | 56.34 |
| 1.0              | 54.11          | 63.54 | 59.67    | 59.11 |
| 2.0              | 55.50          | 65.59 | 61.67    | 60.92 |
| 3.0              | 53.83          | 63.38 | 60.33    | 59.18 |

Ablation Studies on x, y Allocation. To further investigate the impact of different allocation strategies, we conduct quantitative experiments on our proposed VideoRoPE, comparing sequential and interleaved allocations of x and y. The results, summarized in Table [7,](#page-13-3) indicate that interleaving x and y leads to superior performance.

<span id="page-13-3"></span>We hypothesize that this improvement arises because interleaving maintains the similarity between the x and y dimensions, whereas sequential allocation increases their disparity, thereby hindering model performance.

Table 7. Ablation Study on x, y Allocation. VideoRoPE (Sequential) represents the sequential allocation of x and y, following the pattern x, x, x, . . . , y, y, y, . . . (similar to M-RoPE [\(Wang et al.,](#page-11-0) [2024a\)](#page-11-0)). VideoRoPE (Interleaved) represents the interleaved allocation, following the pattern x, y, x, y, . . . (similar to [Agrawal et al.](#page-8-5) [\(2024\)](#page-8-5)).

| Method                  | LongVideoBench |       |       | MLVU  |       |       |       |       |
|-------------------------|----------------|-------|-------|-------|-------|-------|-------|-------|
|                         | 8k             | 16k   | 32k   | 64k   | 8k    | 16k   | 32k   | 64k   |
| VideoRoPE(Sequential)   | 53.73          | 53.52 | 54.97 | 54.77 | 62.75 | 63.31 | 62.75 | 63.08 |
| VideoRoPE (Interleaved) | 54.46          | 55.29 | 57.15 | 57.26 | 65.19 | 66.29 | 66.02 | 65.56 |

Ablation Studies on Diagonal Layout Validated on More Benchmarks To further substantiate our claim that the Diagonal Layout (DL) enhances a model's capability in video understanding tasks, we conduct additional ablation studies on four diverse and challenging benchmarks: MLVU, VideoHallucer, V-NIAH, and V-NIAH-D. These benchmarks cover a wide range of evaluation perspectives, from multi-level video question answering to hallucination detection and fine-grained temporal alignment. As shown in Table [8,](#page-14-1) incorporating the DL module consistently improves performance over the baseline model across all benchmarks. Specifically, we observe notable gains on MLVU and V-NIAH and V-NIAH-D, suggesting that DL effectively facilitates better temporal reasoning and semantic alignment. These results reinforce the generalizability and robustness of the proposed Diagonal Layout design in understanding across various tasks.

Table 8. Effect of Diagonal Layout (DL) across multiple benchmarks.

| Method   | MLVU  | VideoHallucer | V-NIAH | V-NIAH-D |
|----------|-------|---------------|--------|----------|
| baseline | 61.56 | 34.3          | 78.67  | 74.67    |
| + DL     | 63.03 | 34.8          | 80.44  | 76.44    |

<span id="page-14-1"></span>Ablation Studies on Different Frequency Allocation Strategies We compare three different frequency allocation strategies: the *M-RoPE* approach, which emphasizes high-frequency modeling of temporal information and follows a [t...x...y...] format; an interleaved and evenly distributed pattern such as [t t x y x y x y]; and our proposed *VideoRoPE* method, which prioritizes positional encoding followed by low-frequency temporal modeling, arranged in a [x y...t...] format.

We evaluate these approaches on the *LongVideoBench* benchmark under varying context lengths. This benchmark includes a diverse set of video scenarios, ranging from rapidly changing dynamic scenes to slowly evolving static content.

As shown in the results below, our low-frequency temporal allocation consistently outperforms the interleaved [t t x y x y x y] pattern on average. This suggests that our frequency design more effectively balances global temporal context modeling with local spatial dynamics, making it better suited to handle a wide variety of video conditions.

Table 9. Comparison of different frequency allocation strategies under various context lengths.

| Context | [txy] | [t t x y x y] | [xyt] (Ours) |
|---------|-------|---------------|--------------|
| 16k     | 60.05 | 59.95         | 62.03        |
| 32k     | 59.33 | 58.40         | 59.54        |
| 64k     | 58.71 | 57.73         | 59.12        |
| Avg     | 59.36 | 59.06         | 60.14        |

#### A.2. Extrapolation to 128k Experiments

To explore the extrapolation limits of our approach, we extend the visual context during inference to 128k. Specifically, we utilize the vLLM framework [\(Kwon et al.,](#page-9-10) [2023\)](#page-9-10) in Server-API processing mode to enable efficient 128k inference.

<span id="page-14-2"></span>Due to the prolonged evaluation time required for 128k processing, we focus on the LongVideoBench benchmark. As shown in Table [10,](#page-14-2) although all four methods exhibit performance degradation at 128k, our proposed VideoRoPE experiences the least drop, demonstrating its robustness under extreme extrapolation settings.

Table 10. Comparison of model performance at 64k and 128k context lengths for different methods.

| Method                         | LongVideoBench |       |  |  |
|--------------------------------|----------------|-------|--|--|
|                                | 64k            | 128k  |  |  |
| Vanilla RoPE (Su et al., 2024) | 54.04          | 48.01 |  |  |
| TAD-RoPE (Gao et al., 2024)    | 53.42          | 45.77 |  |  |
| M-RoPE (Wang et al., 2024a)    | 54.35          | 51.45 |  |  |
| VideoRoPE                      | 57.26          | 55.64 |  |  |

### <span id="page-14-0"></span>B. Additional Details on Evaluation Benchmarks

For long video understanding, we employ three benchmarks: (1) LongVideoBench highlights reasoning questions that depend on long frame sequences, which cannot be effectively addressed by a single frame or a few sparse frames, with durations ranging from 8 seconds to 1 hour. We retain only the questions that are free from subtitles. (2) MLVU provides a comprehensive benchmark tailored for assessing the performance of Multimodal Large Language Models in understanding long videos. The dataset features videos lasting between 3 minutes and 2 hours, with nine diverse evaluation tasks. For our analysis, we concentrate on seven multiple-choice tasks, including Topic Reasoning, Anomaly Recognition, Needle QA, Ego Reasoning, Plot QA, Action Order, and Action Count. (3) Video-MME stands out as a high-quality benchmark curated for broad scenario coverage, with videos drawn from six key visual domains and 30 subfields. Its dataset spans a wide temporal range, including short clips of 11 seconds and extended videos lasting up to 1 hour.

For long video retrieval, we adopt the following two benchmarks: (1) V-NIAH is specifically designed to identify highly specific moments within long videos, simulating real-world scenarios where only a small segment of a video is relevant within a vast corpus. The setup follows the same configuration as LongVA, where a "needle" image is inserted at a random position within a "haystack" of 3,000 frames. Each needle image corresponds to a particular question, which is unrelated to the content of the haystack. Each frame is encoded with 144 tokens, and the needle frame is inserted at 0.2 depth intervals. Validation begins at 100 frames, with checks every 200 frames up to 3,000. (2) Vision Needle-in-a-Haystack with Distractors (V-NIAH-D), our proposed method, builds upon V-NIAH by periodically inserting a distractor 200 frames away from the needle. This distractor is semantically similar to the needle, but it remains irrelevant to the specific question being asked. The insertion period for the distractor is calculated using 2 · π · 100000032/<sup>128</sup> ≈ 198.7. In our experiments, we directly use a period of 200 for distractor insertion. For additional examples, refer to Figure [8.](#page-17-0)

For the video hallucination, we use VideoHallucer for evaluation. VideoHallucer classifies hallucinations into two primary types: intrinsic and extrinsic. It further breaks these down into subcategories for detailed analysis, including object-relation, temporal, semantic detail, extrinsic factual, and extrinsic non-factual hallucinations. This framework assesses the model's ability to accurately answer both basic and hallucinated questions about the video content.

