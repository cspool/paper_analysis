# <span id="page-20-0"></span>F More Details about Experiment

### F.1 Detailed Experiment Settings

**Baseline setup.** For AKS [18], we adhered to the default configuration: candidate frames were sampled at 1 fps, and frame-question similarity was computed via BLIP [81]. Based on the algorithm's selection logic, we evaluated frame budgets of  $\{32, 64, 128, 192, 256\}$ . We excluded budgets of 8 and 16 as the algorithm occasionally yielded null returns at these low settings. For Q-Frame [64], we employed the default "fixed frame count" strategy. Since this method limits the initial candidate pool to 128 frames, our evaluation was restricted to budgets of  $\{8, 16, 32, 64, 128\}$ .

### F.2 Extended Experiments with DIG

To investigate the scalability of **DIG** in ultra-long context scenarios, we extended our experiments using Qwen3-VL-8B [76], an open-source LMM distinguished for its robust long-context processing capability. We test **DIG** against the uniform sampling baseline and AKS [18].

**Experiment settings.** For **DIG**, the query identification and CAFS configurations align with Section 5, with the exception that we employ Qwen3-VL-8B [76] as the unified backbone for both reward assignment and final inference. Similarly, AKS [18] setup mirrors Section 5 but utilizes Qwen3-VL-8B [76] as the base model. To rigorously test performance across varying context lengths, we scaled input frame counts from 8 to 768, with each frame encoded into approximately 150 tokens. The results are in Table 5.

**DIG delivers consistent performance gains.** As evidenced in Table 5, **DIG** yields substantial improvements across nearly all frame configurations. Notably, with 256 input frames, DIG achieves an 8.6% performance boost on MLVU [54] compared to uniform sampling. Crucially, DIG maintains robustness even at the extreme scale of 768 frames, surpassing the baseline by 4.7% on MLVU [54], 3.7% on LongVideoBench [55], and 3.5% on VideoMME-Medium [56]. In contrast, while AKS [18] remains competitive at lower frame counts ( $\leq 64$ ), it exhibits marked performance degradation as the context length increases, frequently falling below the uniform sampling baseline. Given that practical video understanding tasks necessitate maximizing input frames to capture comprehensive temporal details, AKS [18] demonstrates limited utility for real-world applications. Conversely, **DIG** exhibits superior scalability, effectively delivering sustained performance gains.

### F.3 Detailed Experiment Results & More Analysis

We present detailed performance breakdowns corresponding to the benchmarks discussed in Section 5. Comprehensive quantitative results are in Tables 6, 7, and 8.

**Uniform sampling suffices for global queries.** For global queries, specifically Anomaly Recognition and Topic Reasoning tasks within MLVU [54], all evaluated methods perform comparably to uniform sampling, regardless of the input frame count. This observation reaffirms our previous assertion: uniform sampling is the preferred strategy for global queries, as it achieves sufficient performance while maintaining high efficiency.

Inference for localized queries operates in two distinct stages: query-aware frame selection and subsequent reasoning based on the retrieved content. Without the initial selection stage, evaluating the model's fundamental performance

<span id="page-21-0"></span>**Table 5:** *Performance comparison between different frame selection methods. Base LMM is Qwen3-VL-8B [\[76\]](#page-15-7). Bold indicates best performance, while Red Box denote results inferior to uniform sampling.*

| Method     | #Frames | MLVU | LVB  | VideoMME |      |  |  |
|------------|---------|------|------|----------|------|--|--|
|            |         |      |      | Medium   | Long |  |  |
| UNI        | 8       | 53.4 | 50.3 | 48.4     | 49.2 |  |  |
| DIG (Ours) | 8       | 58.2 | 54.9 | 53.1     | 49.8 |  |  |
| UNI        | 16      | 53.9 | 50.9 | 51.3     | 48.0 |  |  |
| DIG (Ours) | 16      | 58.9 | 53.9 | 52.9     | 49.9 |  |  |
| UNI        | 32      | 53.7 | 51.0 | 49.4     | 48.6 |  |  |
| AKS [18]   | 32      | 57.3 | 54.4 | 52.3     | 50.1 |  |  |
| DIG (Ours) | 32      | 58.7 | 53.2 | 53.9     | 49.4 |  |  |
| UNI        | 64      | 54.7 | 51.2 | 49.9     | 48.9 |  |  |
| AKS [18]   | 64      | 56.3 | 52.7 | 50.9     | 51.6 |  |  |
| DIG (Ours) | 64      | 59.6 | 54.8 | 54.7     | 49.0 |  |  |
| UNI        | 128     | 57.2 | 54.4 | 55.1     | 51.3 |  |  |
| AKS [18]   | 128     | 58.9 | 53.0 | 54.4     | 51.1 |  |  |
| DIG (Ours) | 128     | 64.4 | 58.3 | 58.4     | 51.1 |  |  |
| UNI        | 192     | 58.9 | 57.1 | 57.6     | 51.6 |  |  |
| AKS [18]   | 192     | 61.1 | 54.5 | 61.0     | 53.8 |  |  |
| DIG (Ours) | 192     | 66.8 | 60.4 | 58.2     | 50.7 |  |  |
| UNI        | 256     | 60.4 | 57.6 | 57.8     | 53.4 |  |  |
| AKS [18]   | 256     | 63.8 | 55.6 | 59.2     | 53.2 |  |  |
| DIG (Ours) | 256     | 69.0 | 61.2 | 61.6     | 53.8 |  |  |
| UNI        | 512     | 65.4 | 60.2 | 61.4     | 55.0 |  |  |
| AKS [18]   | 512     | 65.5 | 57.6 | 60.7     | 56.0 |  |  |
| DIG (Ours) | 512     | 71.7 | 63.8 | 65.6     | 56.4 |  |  |
| UNI        | 768     | 67.5 | 60.9 | 64.3     | 56.6 |  |  |
| AKS [18]   | 768     | 65.3 | 58.3 | 62.3     | 57.3 |  |  |
| DIG (Ours) | 768     | 72.2 | 64.6 | 67.8     | 59.0 |  |  |

is challenging, as errors may stem from information-deficient inputs rather than inherent model limitations. By incorporating this stage to ensure the input contains relevant information, we can decouple data retrieval issues from reasoning capabilities. This allows for a more accurate assessment of the model's intrinsic proficiency across different tasks, yielding deeper insights.

**Query-aware selection uncovers intrinsic visual perception capabilities.** As shown in Table [6](#page-22-0) and [8,](#page-23-0) our method significantly and consistently outperforms uniform sampling on localized perception tasks (e.g., PlotQA, NeedleQA, and L1-Perception). Notably, these tasks primarily evaluate fundamental visual perception capabilities. Our findings suggest that LMMs are intrinsically capable of solving such tasks, provided the query-relevant information is effectively supplied. This explains the substantial performance gap: while uniform sampling often introduces significant noise by including irrelevant content, query-aware selection ensures the model is conditioned on relevant frames.

**Temporal reasoning remains a fundamental bottleneck.** Conversely, regarding tasks requiring temporal logic (e.g., Action Order and L2-Relation), performance remains stagnant across all methods. Even when provided with query-relevant visual information, model performance does not improve. This underscores a critical limitation: current LMMs struggle with temporal reasoning and sequencing, a deficiency that persists independently of the quality of visual information retrieval.

<span id="page-22-0"></span>**Table 6:** Performance comparison between different frame selection methods on MLVU. Base LMMs are Qwen2.5-VL-32B [16] (left) and Qwen2.5-VL-7B [16] (right). Bold indicates best performance. The tasks of MLVU [54] are PlotQA (PQA), NeedleQA (NQA), Action Count (AC), Action Order (AO), Ego Reasoning (ER), Anomaly Recognition (AR), Topic Reasoning (TR).

| Method       | <br>#Frames |      |      | M    | LVU [5 | 4]   |      |      |
|--------------|-------------|------|------|------|--------|------|------|------|
|              |             | PQA  | NQA  | AC   | AO     | ER   | AR   | TR   |
| UNI          | 8           | 55.8 | 58.6 | 18.5 | 51.4   | 50.6 | 66.5 | 85.6 |
| Q-Frame [64] | 8           | 51.4 | 63.9 | 18.4 | 60.2   | 50.3 | 70.5 | 76.8 |
| DIG (Ours)   | 8           | 62.3 | 73.0 | 27.2 | 58.3   | 56.2 | 66.0 | 84.0 |
| UNI          | 16          | 59.4 | 63.9 | 18.0 | 54.8   | 52.8 | 69.5 | 86.3 |
| Q-Frame [64] | 16          | 56.4 | 64.8 | 19.9 | 59.5   | 51.4 | 70.5 | 77.7 |
| DIG (Ours)   | 16          | 67.9 | 78.0 | 35.0 | 66.8   | 57.1 | 69.0 | 86.7 |
| UNI          | 32          | 61.8 | 67.9 | 18.5 | 58.7   | 57.4 | 76.0 | 86.7 |
| AKS [18]     | 32          | 66.8 | 73.0 | 40.3 | 56.0   | 59.9 | 74.5 | 90.1 |
| Q-Frame [64] | 32          | 61.4 | 67.9 | 18.9 | 63.8   | 53.1 | 71.5 | 83.3 |
| DIG (Ours)   | 32          | 72.4 | 79.2 | 48.1 | 75.7   | 59.1 | 74.0 | 87.5 |
| UNI          | 64          | 68.5 | 72.1 | 25.7 | 61.4   | 61.1 | 80.0 | 86.7 |
| AKS [18]     | 64          | 73.8 | 76.6 | 40.8 | 58.3   | 63.1 | 75.0 | 88.2 |
| Q-Frame [64] | 64          | 68.5 | 73.2 | 21.8 | 66.0   | 59.7 | 77.5 | 85.9 |
| DIG (Ours)   | 64          | 75.9 | 81.1 | 49.5 | 78.4   | 66.5 | 78.5 | 89.7 |
| UNI          | 128         | 73.5 | 76.3 | 30.6 | 68.7   | 64.2 | 79.0 | 89.4 |
| AKS [18]     | 128         | 78.3 | 80.3 | 42.2 | 61.8   | 69.0 | 77.0 | 87.8 |
| Q-Frame [64] | 128         | 73.1 | 76.1 | 30.1 | 69.1   | 64.5 | 79.5 | 88.6 |
| DIG (Ours)   | 128         | 79.8 | 80.0 | 52.4 | 79.2   | 65.6 | 78.5 | 89.7 |
| UNI          | 192         | 75.0 | 78.0 | 36.9 | 69.9   | 64.5 | 78.0 | 90.9 |
| AKS [18]     | 192         | 77.6 | 81.4 | 47.1 | 63.3   | 68.2 | 77.0 | 89.4 |
| DIG (Ours)   | 192         | 82.6 | 81.4 | 53.9 | 80.7   | 65.3 | 79.0 | 91.6 |
|              |             |      |      |      |        |      |      |      |

| Model        | #Frames   |      |      | M    | LVU [5 | 4]   |      |      |
|--------------|-----------|------|------|------|--------|------|------|------|
| Wodel        | #ITallies | PQA  | NQA  | AC   | AO     | ER   | AR   | TR   |
| UNI          | 8         | 52.1 | 62.5 | 19.4 | 44.0   | 48.6 | 66.5 | 82.9 |
| Q-Frame [64] | 8         | 50.6 | 67.0 | 19.9 | 48.6   | 49.7 | 68.0 | 73.8 |
| DIG (Ours)   | 8         | 57.1 | 73.2 | 31.6 | 48.6   | 51.1 | 66.5 | 82.9 |
| UNI          | 16        | 56.0 | 63.4 | 19.9 | 42.5   | 54.8 | 70.0 | 84.4 |
| Q-Frame [64] | 16        | 55.5 | 67.6 | 20.4 | 50.6   | 49.7 | 70.5 | 78.7 |
| DIG (Ours)   | 16        | 66.4 | 79.7 | 36.9 | 51.7   | 55.4 | 68.5 | 85.2 |
| UNI          | 32        | 59.7 | 69.0 | 22.8 | 51.4   | 54.0 | 74.5 | 84.4 |
| AKS [18]     | 32        | 69.6 | 76.3 | 42.2 | 50.2   | 56.5 | 72.0 | 85.2 |
| Q-Frame [64] | 32        | 60.1 | 67.9 | 23.9 | 54.1   | 54.3 | 70.0 | 83.7 |
| DIG (Ours)   | 32        | 70.3 | 80.6 | 42.2 | 54.4   | 59.1 | 73.0 | 87.5 |
| UNI          | 64        | 64.0 | 74.4 | 26.2 | 51.7   | 59.9 | 76.0 | 87.1 |
| AKS [18]     | 64        | 69.9 | 80.8 | 41.3 | 54.1   | 60.5 | 69.5 | 84.8 |
| Q-Frame [64] | 64        | 63.8 | 73.5 | 26.2 | 56.0   | 58.2 | 72.0 | 85.9 |
| DIG (Ours)   | 64        | 75.3 | 82.8 | 46.6 | 60.2   | 62.2 | 75.5 | 87.8 |
| UNI          | 128       | 71.4 | 79.2 | 34.5 | 58.7   | 61.4 | 73.0 | 86.7 |
| AKS [18]     | 128       | 71.6 | 83.7 | 48.5 | 57.1   | 60.8 | 72.0 | 84.0 |
| Q-Frame [64] | 128       | 71.4 | 79.2 | 34.5 | 58.7   | 61.4 | 73.0 | 86.7 |
| DIG (Ours)   | 128       | 78.3 | 82.3 | 45.6 | 62.5   | 63.6 | 72.0 | 87.8 |
| UNI          | 192       | 72.0 | 80.8 | 40.3 | 61.4   | 63.6 | 73.0 | 87.5 |
| AKS [18]     | 192       | 74.2 | 83.1 | 46.1 | 58.7   | 63.6 | 73.0 | 85.6 |
| DIG (Ours)   | 192       | 78.7 | 84.5 | 47.1 | 63.3   | 65.3 | 72.0 | 87.5 |
| UNI          | 256       | 73.5 | 80.0 | 41.3 | 61.4   | 61.1 | 73.0 | 89.0 |
| AKS [18]     | 256       | 75.3 | 84.2 | 47.3 | 59.5   | 66.2 | 75.5 | 87.8 |
| DIG (Ours)   | 256       | 78.1 | 84.5 | 49.0 | 62.2   | 65.1 | 73.0 | 89.0 |

<span id="page-22-1"></span>**Table 7:** Performance comparison between different frame selection methods on VideoMME. Base LMMs are Qwen2.5-VL-7B [16](left) and Qwen2.5-VL-32B [16](right). Bold indicates best performance. The tasks are Object Reasoning (ORA), Object Recognition (ORC), Action Reasoning (ARA), Information Synopsis (INS), Counting Problem (COP), Temporal Reasoning (TER), Temporal Perception (TEP), Spatial Perception (SPP), Spatial Reasoning (SPR), OCR, Attribute Perception (ATP), Action Recognition (ACR).

| Model        | #Frames |      |      |      |      | V    | 'ideoM | ME [56 | ]    |      |      |      |      |
|--------------|---------|------|------|------|------|------|--------|--------|------|------|------|------|------|
| Model        |         | ORA  | ORC  | ARA  | INS  | COP  | TER    | TEP    | SPR  | SPP  | OCR  | ATP  | ACR  |
| UNI          | 8       | 49.5 | 54.5 | 49.5 | 67.8 | 36.2 | 40.7   | 47.3   | 58.9 | 63.0 | 48.9 | 62.2 | 53.4 |
| Q-Frame [64] | 8       | 50.5 | 50.1 | 50.8 | 64.8 | 36.9 | 36.0   | 43.3   | 62.1 | 33.3 | 44.0 | 57.0 | 52.2 |
| DIG (Ours)   | 8       | 47.6 | 60.2 | 52.3 | 70.0 | 39.6 | 40.7   | 56.4   | 64.3 | 66.7 | 55.4 | 65.3 | 55.9 |
| UNI          | 16      | 53.5 | 58.8 | 51.2 | 74.6 | 37.7 | 43.5   | 63.6   | 64.3 | 72.2 | 54.7 | 67.1 | 56.9 |
| Q-Frame [64] | 16      | 52.4 | 52.2 | 52.2 | 66.0 | 36.8 | 36.1   | 56.9   | 62.1 | 45.9 | 48.8 | 59.0 | 49.0 |
| DIG (Ours)   | 16      | 57.9 | 61.0 | 54.4 | 74.6 | 41.4 | 43.5   | 56.4   | 66.1 | 75.9 | 59.0 | 71.2 | 56.2 |
| UNI          | 32      | 57.5 | 63.6 | 55.8 | 76.5 | 42.5 | 43.5   | 67.3   | 76.8 | 72.2 | 62.6 | 74.8 | 59.7 |
| AKS [18]     | 32      | 56.8 | 66.7 | 51.9 | 80.2 | 42.9 | 49.2   | 60.0   | 76.8 | 72.2 | 66.2 | 74.8 | 59.7 |
| Q-Frame [64] | 32      | 55.3 | 56.6 | 54.2 | 68.9 | 38.3 | 37.2   | 59.6   | 65.5 | 50.1 | 50.1 | 64.0 | 50.6 |
| DIG (Ours)   | 32      | 59.5 | 67.2 | 56.1 | 75.9 | 40.3 | 46.9   | 52.7   | 76.8 | 72.2 | 69.8 | 73.4 | 61.0 |
| UNI          | 64      | 58.1 | 66.7 | 54.0 | 76.8 | 41.4 | 43.5   | 69.1   | 76.8 | 68.5 | 67.6 | 74.8 | 63.9 |
| AKS [18]     | 64      | 58.4 | 67.5 | 56.1 | 79.3 | 44.4 | 52.0   | 72.7   | 76.8 | 72.2 | 72.7 | 74.8 | 61.0 |
| Q-Frame [64] | 64      | 56.0 | 62.4 | 56.0 | 72.6 | 46.5 | 45.2   | 56.8   | 69.1 | 54.2 | 51.3 | 72.0 | 51.2 |
| DIG (Ours)   | 64      | 60.6 | 68.4 | 57.5 | 78.0 | 46.3 | 49.2   | 58.2   | 73.2 | 75.9 | 73.4 | 73.4 | 63.3 |
| UNI          | 128     | 61.2 | 71.2 | 58.9 | 79.9 | 45.1 | 57.1   | 70.9   | 76.8 | 68.5 | 71.2 | 76.6 | 66.1 |
| AKS [18]     | 128     | 61.0 | 68.4 | 59.3 | 80.2 | 44.8 | 57.1   | 76.4   | 75.0 | 68.5 | 77.0 | 77.9 | 61.3 |
| Q-Frame [64] | 128     | 58.7 | 64.5 | 55.6 | 78.0 | 38.9 | 55.6   | 64.9   | 72.4 | 54.3 | 57.4 | 67.0 | 59.3 |
| DIG (Ours)   | 128     | 61.7 | 72.3 | 57.2 | 80.5 | 45.1 | 52.5   | 58.2   | 78.6 | 68.5 | 71.2 | 78.8 | 66.8 |
| UNI          | 192     | 62.3 | 72.0 | 60.7 | 79.9 | 48.5 | 54.2   | 72.7   | 76.8 | 70.4 | 71.9 | 77.5 | 68.1 |
| AKS [18]     | 192     | 60.8 | 69.2 | 57.5 | 79.3 | 45.5 | 57.6   | 81.8   | 76.8 | 72.2 | 76.3 | 77.9 | 63.9 |
| DIG (Ours)   | 192     | 64.5 | 73.7 | 60.0 | 80.2 | 46.3 | 54.2   | 65.5   | 78.6 | 68.5 | 74.1 | 79.3 | 65.8 |
| UNI          | 256     | 62.1 | 71.5 | 59.6 | 82.4 | 46.6 | 57.6   | 76.4   | 75.0 | 66.7 | 71.9 | 77.9 | 68.1 |
| AKS [18]     | 256     | 61.0 | 70.9 | 57.9 | 79.3 | 44.8 | 57.1   | 83.6   | 75.0 | 70.4 | 74.1 | 77.9 | 64.5 |
| DIG (Ours)   | 256     | 63.0 | 72.0 | 62.1 | 82.4 | 46.3 | 54.8   | 67.3   | 78.6 | 66.7 | 73.4 | 80.2 | 67.4 |

| Model        | #Frames |      |      |      |      | V    | ideoM | ME [56 | ]    |      |      |      |      |
|--------------|---------|------|------|------|------|------|-------|--------|------|------|------|------|------|
|              |         | ORA  | ORC  | ARA  | INS  | COP  | TER   | TEP    | SPR  | SPP  | OCR  | ATP  | ACR  |
| UNI          | 8       | 57.2 | 54.8 | 53.4 | 69.7 | 30.1 | 37.8  | 46.0   | 65.5 | 54.2 | 47.6 | 62.0 | 45.6 |
| Q-Frame [64] | 8       | 50.5 | 50.1 | 50.8 | 64.8 | 36.9 | 36.0  | 43.3   | 62.1 | 33.3 | 44.0 | 57.0 | 52.2 |
| DIG (Ours)   | 8       | 55.6 | 58.1 | 53.4 | 69.7 | 31.5 | 40.9  | 46.0   | 72.4 | 62.5 | 54.9 | 62.0 | 48.4 |
| UNI          | 16      | 55.4 | 56.5 | 52.5 | 70.5 | 32.2 | 47.6  | 51.4   | 75.9 | 62.5 | 42.7 | 65.0 | 50.6 |
| Q-Frame [64] | 16      | 52.4 | 52.2 | 52.2 | 66.0 | 36.8 | 36.1  | 56.9   | 62.1 | 45.9 | 48.8 | 59.0 | 49.0 |
| DIG (Ours)   | 16      | 58.3 | 59.7 | 53.4 | 71.0 | 37.8 | 47.0  | 56.8   | 75.9 | 62.5 | 45.1 | 65.0 | 47.3 |
| UNI          | 32      | 55.4 | 55.9 | 54.2 | 76.8 | 35.0 | 40.9  | 62.2   | 75.9 | 62.5 | 47.6 | 66.0 | 51.7 |
| AKS [18]     | 32      | 58.6 | 57.0 | 57.6 | 78.0 | 34.3 | 47.6  | 56.8   | 79.3 | 45.8 | 61.0 | 68.0 | 51.1 |
| Q-Frame [64] | 32      | 55.3 | 56.6 | 54.2 | 68.9 | 38.3 | 37.2  | 59.6   | 65.5 | 50.1 | 50.1 | 64.0 | 50.6 |
| DIG (Ours)   | 32      | 58.8 | 64.0 | 55.5 | 78.0 | 35.7 | 51.8  | 48.7   | 75.9 | 58.3 | 56.1 | 69.0 | 50.0 |
| UNI          | 64      | 61.8 | 62.9 | 58.0 | 76.8 | 34.3 | 44.5  | 64.9   | 79.3 | 62.5 | 58.5 | 69.0 | 59.3 |
| AKS [18]     | 64      | 61.0 | 63.4 | 59.7 | 80.1 | 37.8 | 54.3  | 62.2   | 79.3 | 54.2 | 59.8 | 73.0 | 56.6 |
| Q-Frame [64] | 64      | 56.0 | 62.4 | 56.0 | 72.6 | 46.5 | 45.2  | 56.8   | 69.1 | 54.2 | 51.3 | 72.0 | 51.2 |
| DIG (Ours)   | 64      | 62.6 | 65.6 | 57.6 | 76.4 | 32.9 | 51.8  | 59.5   | 75.9 | 58.3 | 67.1 | 73.0 | 58.2 |
| UNI          | 128     | 63.4 | 69.9 | 63.5 | 81.3 | 42.0 | 54.3  | 59.5   | 86.2 | 58.3 | 68.3 | 73.0 | 57.1 |
| AKS [18]     | 128     | 66.0 | 68.3 | 58.8 | 81.3 | 42.7 | 57.9  | 67.6   | 82.8 | 50.0 | 69.5 | 75.0 | 59.9 |
| Q-Frame [64] | 128     | 58.7 | 64.5 | 55.6 | 78.0 | 38.9 | 55.6  | 64.9   | 72.4 | 54.3 | 57.4 | 67.0 | 59.3 |
| DIG (Ours)   | 128     | 65.2 | 73.1 | 61.8 | 81.7 | 45.5 | 52.4  | 64.9   | 82.8 | 54.2 | 69.5 | 75.0 | 60.4 |
| UNI          | 192     | 64.7 | 69.9 | 63.0 | 81.3 | 44.8 | 55.5  | 73.0   | 86.2 | 62.5 | 68.3 | 75.0 | 62.1 |
| AKS [18]     | 192     | 65.5 | 69.4 | 59.7 | 81.3 | 42.7 | 58.5  | 75.7   | 79.3 | 54.2 | 73.2 | 78.0 | 58.8 |
| DIG (Ours)   | 192     | 66.8 | 74.2 | 62.6 | 81.7 | 42.0 | 57.3  | 64.9   | 79.3 | 58.3 | 73.2 | 80.0 | 61.0 |

<span id="page-23-0"></span>**Table 8: Performance Comparison between Different Frame Selection Methods on LongVideoBench.** Base LMMs are Qwen2.5-VL-7B [16](top) and Qwen2.5-VL-32B [16](bottom). **Bold** indicates best performance.

|              |         |      |      |      |      |         |      |      |      | LongV | ideoBer | nch [55] |      |      |       |        |      |      |      |      |
|--------------|---------|------|------|------|------|---------|------|------|------|-------|---------|----------|------|------|-------|--------|------|------|------|------|
| Model        | #Frames |      |      |      | L1-  | Percept | tion |      |      |       |         |          |      |      | L2-Re | lation |      |      |      |      |
|              |         | S2E  | S2A  | O2E  | T2O  | S2O     | T2E  | E2O  | T2A  | Avg   | TOS     | ЕЗЕ      | SAA  | ОЗО  | ТЗО   | ТЗЕ    | TAA  | SSS  | SOS  | Avg  |
| UNI          | 8       | 57.0 | 51.1 | 62.8 | 56.6 | 45.8    | 58.5 | 56.9 | 49.4 | 54.4  | 38.4    | 62.8     | 47.2 | 45.5 | 47.3  | 47.9   | 46.3 | 34.0 | 64.2 | 48.3 |
| Q-Frame [64] | 8       | 67.7 | 73.9 | 60.9 | 57.9 | 55.6    | 64.6 | 63.1 | 55.7 | 62.7  | 31.5    | 62.8     | 52.8 | 47.0 | 39.2  | 48.0   | 45.1 | 28.9 | 65.4 | 46.8 |
| DIG (Ours)   | 8       | 69.9 | 68.2 | 62.1 | 50.0 | 55.6    | 61.5 | 61.5 | 62.0 | 61.8  | 37.0    | 61.7     | 50.0 | 48.5 | 54.1  | 43.8   | 45.1 | 29.9 | 67.9 | 48.6 |
| UNI          | 16      | 65.6 | 64.8 | 62.8 | 53.9 | 48.6    | 61.5 | 61.5 | 51.9 | 59.0  | 37.0    | 62.8     | 50.0 | 47.0 | 56.8  | 45.2   | 51.9 | 36.1 | 63.0 | 49.3 |
| Q-Frame [64] | 16      | 66.7 | 70.5 | 65.5 | 63.2 | 61.1    | 64.6 | 69.2 | 63.3 | 65.6  | 34.3    | 59.6     | 56.9 | 54.6 | 43.2  | 49.3   | 50.0 | 38.1 | 65.4 | 50.1 |
| DIG (Ours)   | 16      | 72.0 | 71.6 | 59.8 | 65.8 | 54.2    | 69.2 | 69.2 | 63.3 | 65.4  | 37.0    | 57.4     | 52.8 | 43.9 | 56.8  | 52.1   | 43.9 | 36.1 | 74.1 | 50.4 |
| UNI          | 32      | 67.7 | 58.0 | 61.7 | 56.6 | 62.5    | 67.7 | 67.7 | 51.9 | 61.9  | 37.0    | 61.7     | 55.6 | 56.1 | 55.4  | 49.3   | 52.4 | 40.2 | 66.7 | 52.7 |
| AKS [18]     | 32      | 65.6 | 77.3 | 67.8 | 63.2 | 63.9    | 63.1 | 63.1 | 64.6 | 66.1  | 37.0    | 67.0     | 58.3 | 56.1 | 45.9  | 53.4   | 48.8 | 38.1 | 74.1 | 53.2 |
| Q-Frame [64] | 32      | 64.5 | 69.3 | 60.9 | 59.2 | 61.1    | 61.5 | 69.2 | 60.8 | 63.3  | 32.9    | 59.6     | 62.5 | 50.0 | 50.0  | 45.2   | 46.3 | 39.2 | 66.7 | 50.3 |
| DIG (Ours)   | 32      | 72.0 | 78.4 | 63.2 | 68.4 | 62.5    | 67.7 | 67.7 | 62.0 | 68.0  | 41.1    | 62.8     | 52.8 | 51.5 | 55.4  | 50.7   | 48.8 | 36.1 | 70.4 | 52.1 |
| UNI          | 64      | 73.1 | 67.0 | 62.8 | 59.2 | 56.9    | 63.1 | 66.2 | 62.0 | 64.6  | 34.2    | 62.8     | 58.3 | 59.1 | 60.8  | 47.9   | 51.2 | 43.3 | 67.9 | 53.9 |
| AKS [18]     | 64      | 69.9 | 77.3 | 71.3 | 65.8 | 59.7    | 60.0 | 64.6 | 64.6 | 67.2  | 41.1    | 70.2     | 61.1 | 56.1 | 47.3  | 49.3   | 47.6 | 40.2 | 76.5 | 54.5 |
| Q-Frame [64] | 64      | 63.4 | 70.5 | 63.2 | 57.9 | 61.1    | 63.1 | 64.6 | 65.8 | 63.8  | 34.3    | 67.0     | 58.3 | 53.0 | 52.7  | 46.6   | 51.2 | 37.1 | 66.7 | 52.0 |
| DIG (Ours)   | 64      | 69.9 | 78.4 | 66.7 | 69.7 | 55.6    | 67.7 | 72.3 | 68.4 | 68.8  | 38.4    | 66.0     | 58.3 | 62.1 | 59.5  | 53.4   | 46.3 | 42.3 | 69.1 | 54.9 |
| UNI          | 128     | 71.0 | 67.0 | 67.8 | 64.5 | 61.1    | 67.7 | 72.3 | 73.4 | 68.2  | 38.4    | 68.1     | 56.9 | 59.1 | 56.8  | 50.7   | 54.9 | 46.4 | 74.1 | 56.3 |
| AKS [18]     | 128     | 69.9 | 72.7 | 66.7 | 64.5 | 61.1    | 60.0 | 66.2 | 64.6 | 66.1  | 37.0    | 68.1     | 63.9 | 56.1 | 50.0  | 50.7   | 48.8 | 47.4 | 76.5 | 55.6 |
| Q-Frame [64] | 128     | 66.7 | 67.1 | 69.0 | 60.5 | 59.7    | 64.6 | 67.7 | 64.6 | 65.1  | 38.4    | 64.9     | 58.3 | 54.6 | 56.8  | 49.3   | 51.2 | 45.4 | 75.3 | 55.1 |
| DIG (Ours)   | 128     | 72.0 | 79.5 | 66.7 | 71.1 | 61.1    | 69.2 | 75.4 | 70.9 | 70.9  | 38.4    | 68.1     | 61.1 | 65.2 | 56.8  | 57.5   | 47.6 | 45.4 | 67.9 | 56.3 |
| UNI          | 192     | 74.2 | 72.7 | 66.0 | 68.4 | 59.7    | 67.7 | 70.8 | 63.3 | 68.2  | 35.6    | 66.0     | 59.7 | 59.1 | 58.1  | 58.9   | 56.1 | 46.4 | 67.9 | 56.5 |
| AKS [18]     | 192     | 69.9 | 76.1 | 67.8 | 63.2 | 61.1    | 63.1 | 67.7 | 65.8 | 67.2  | 35.6    | 68.1     | 62.5 | 56.1 | 56.8  | 52.1   | 48.8 | 46.4 | 72.8 | 55.6 |
| DIG (Ours)   | 192     | 74.2 | 84.1 | 66.7 | 67.1 | 59.7    | 69.2 | 73.8 | 78.5 | 72.0  | 35.6    | 68.1     | 61.1 | 66.7 | 56.8  | 61.6   | 48.8 | 49.5 | 70.4 | 57.6 |
| UNI          | 256     | 69.9 | 73.9 | 69.1 | 63.2 | 59.7    | 63.1 | 72.3 | 68.4 | 66.7  | 37.0    | 69.1     | 62.5 | 60.6 | 55.4  | 56.2   | 54.9 | 46.4 | 69.1 | 56.9 |
| AKS [18]     | 256     | 68.8 | 72.7 | 70.1 | 65.8 | 61.1    | 66.2 | 63.1 | 65.8 | 67.0  | 35.6    | 69.1     | 62.5 | 56.1 | 56.8  | 52.1   | 48.8 | 43.3 | 71.6 | 55.2 |
| DIG (Ours)   | 256     | 76.3 | 80.7 | 71.3 | 72.4 | 65.3    | 69.2 | 75.4 | 74.7 | 73.4  | 37.0    | 71.3     | 59.7 | 68.2 | 56.8  | 56.2   | 45.1 | 49.5 | 67.9 | 56.9 |

|              |         |      |      |      |      |         |      |      |      | LongVi | ideoBer | nch [55] | ]    |      |       |         |      |      |      |      |
|--------------|---------|------|------|------|------|---------|------|------|------|--------|---------|----------|------|------|-------|---------|------|------|------|------|
| Model        | #Frames |      |      |      | L1-  | Percept | ion  |      |      |        |         |          |      |      | L2-Re | elation |      |      |      |      |
|              |         | S2E  | S2A  | O2E  | T2O  | S2O     | T2E  | E2O  | T2A  | Avg    | TOS     | E3E      | SAA  | ОЗО  | ТЗО   | T3E     | TAA  | SSS  | SOS  | Avg  |
| UNI          | 8       | 62.4 | 61.4 | 65.5 | 54.0 | 51.4    | 58.5 | 58.5 | 51.9 | 58.2   | 30.1    | 63.8     | 55.6 | 42.4 | 47.3  | 49.3    | 45.1 | 46.4 | 58.0 | 49.2 |
| Q-Frame [64] | 8       | 62.4 | 81.8 | 62.1 | 55.3 | 55.6    | 61.5 | 66.2 | 53.2 | 62.6   | 30.1    | 57.5     | 56.9 | 42.4 | 43.2  | 46.6    | 41.5 | 36.1 | 59.3 | 46.1 |
| DIG (Ours)   | 8       | 63.4 | 75.0 | 60.9 | 59.2 | 48.6    | 60.0 | 66.2 | 60.8 | 61.8   | 37.0    | 60.6     | 61.1 | 50.0 | 52.7  | 49.3    | 47.6 | 44.3 | 65.4 | 52.0 |
| UNI          | 16      | 57.0 | 71.6 | 57.5 | 52.6 | 48.6    | 63.1 | 63.1 | 45.6 | 57.4   | 37.0    | 58.5     | 58.3 | 53.0 | 54.1  | 50.7    | 53.7 | 42.3 | 66.7 | 52.7 |
| Q-Frame [64] | 16      | 69.9 | 77.3 | 64.4 | 64.5 | 58.3    | 64.6 | 63.1 | 62.0 | 65.9   | 31.5    | 62.8     | 58.3 | 51.5 | 46.0  | 43.8    | 46.3 | 40.2 | 54.3 | 49.5 |
| DIG (Ours)   | 16      | 66.7 | 80.7 | 63.2 | 55.3 | 62.5    | 66.2 | 64.6 | 68.4 | 65.9   | 34.3    | 56.4     | 63.9 | 54.5 | 56.8  | 54.8    | 46.3 | 39.2 | 67.9 | 52.7 |
| UNI          | 32      | 68.8 | 68.2 | 64.4 | 57.9 | 54.2    | 63.1 | 61.5 | 53.2 | 61.8   | 32.9    | 66.0     | 56.9 | 53.0 | 52.7  | 58.9    | 50.0 | 47.4 | 70.4 | 54.5 |
| AKS [18]     | 32      | 65.6 | 77.3 | 69.0 | 61.8 | 65.3    | 61.5 | 66.2 | 64.6 | 66.7   | 34.3    | 69.2     | 59.7 | 50.0 | 41.9  | 48.0    | 51.2 | 44.3 | 72.8 | 52.8 |
| Q-Frame [64] | 32      | 69.9 | 77.3 | 63.2 | 59.2 | 65.3    | 61.5 | 60.0 | 54.4 | 61.4   | 37.0    | 59.6     | 54.2 | 48.5 | 48.6  | 48.0    | 51.2 | 50.5 | 60.5 | 51.3 |
| DIG (Ours)   | 32      | 71.0 | 75.0 | 65.5 | 57.9 | 65.3    | 72.3 | 67.7 | 60.8 | 66.9   | 35.6    | 70.2     | 66.7 | 62.1 | 56.8  | 56.2    | 43.9 | 51.5 | 71.6 | 57.2 |
| UNI          | 64      | 69.9 | 65.9 | 64.4 | 61.8 | 58.3    | 60.0 | 70.8 | 57.0 | 63.7   | 31.5    | 67.0     | 58.3 | 51.5 | 55.4  | 54.8    | 51.2 | 51.5 | 69.1 | 54.9 |
| AKS [18]     | 64      | 67.7 | 76.1 | 66.7 | 60.5 | 69.4    | 61.5 | 72.3 | 67.1 | 67.8   | 37.0    | 73.4     | 56.9 | 53.0 | 47.3  | 52.1    | 48.8 | 52.6 | 75.3 | 55.8 |
| Q-Frame [64] | 64      | 63.4 | 77.3 | 66.7 | 56.6 | 66.7    | 64.6 | 67.7 | 65.8 | 64.0   | 35.6    | 63.8     | 61.1 | 47.0 | 46.0  | 52.1    | 52.4 | 53.6 | 67.9 | 53.8 |
| DIG (Ours)   | 64      | 69.9 | 79.5 | 65.5 | 64.5 | 65.3    | 66.2 | 72.3 | 67.1 | 68.8   | 34.3    | 68.1     | 73.6 | 62.1 | 59.5  | 54.8    | 46.3 | 57.7 | 72.8 | 55.8 |
| UNI          | 128     | 71.0 | 70.5 | 62.1 | 61.8 | 68.1    | 63.1 | 67.7 | 60.8 | 65.7   | 35.6    | 70.2     | 62.5 | 51.5 | 55.4  | 57.5    | 53.7 | 60.8 | 71.6 | 58.3 |
| AKS [18]     | 128     | 69.9 | 72.7 | 67.8 | 63.2 | 65.3    | 61.5 | 70.8 | 63.3 | 67.0   | 37.0    | 74.5     | 58.3 | 56.1 | 51.4  | 53.4    | 52.4 | 53.6 | 76.5 | 58.6 |
| Q-Frame [64] | 128     | 65.6 | 69.3 | 60.9 | 59.2 | 65.3    | 61.5 | 67.7 | 59.5 | 63.7   | 35.6    | 71.3     | 61.1 | 48.5 | 51.4  | 53.4    | 52.4 | 60.8 | 70.4 | 56.9 |
| DIG (Ours)   | 128     | 76.3 | 83.0 | 65.5 | 65.8 | 70.8    | 67.7 | 80.0 | 63.3 | 71.6   | 34.3    | 72.3     | 68.1 | 68.2 | 66.2  | 57.5    | 45.1 | 55.7 | 74.1 | 60.2 |
| UNI          | 192     | 72.0 | 73.9 | 66.7 | 67.1 | 66.7    | 67.7 | 72.3 | 63.3 | 68.8   | 34.3    | 76.6     | 58.3 | 62.1 | 58.1  | 60.3    | 52.4 | 56.7 | 71.6 | 59.4 |
| AKS [18]     | 192     | 71.0 | 79.5 | 67.8 | 67.1 | 66.7    | 60.0 | 69.2 | 62.0 | 68.3   | 35.6    | 74.5     | 58.3 | 57.6 | 54.1  | 50.7    | 48.8 | 54.6 | 76.5 | 57.3 |
| DIG (Ours)   | 192     | 73.1 | 84.1 | 67.8 | 68.4 | 68.1    | 64.6 | 76.9 | 65.8 | 71.1   | 38.4    | 74.5     | 70.8 | 69.7 | 71.6  | 57.5    | 43.9 | 56.7 | 75.3 | 60.7 |

