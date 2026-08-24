# Q3: Can the eviction strategy alleviate attention bias? By evicting different types of $R_H$ at different layers, we have the following observations:

**Observations.** *i*): From Figure 5, we can find that KV cache eviction exacerbates the bias. However, parallel KV cache eviction can achieve a more stable distribution. ii): Evicting sink bias tokens in the early layers may exacerbate attention bias, but evicting them in the deeper layers can mitigate this attention bias. iii): Evicting recency bias tokens in the intermediate layers can mitigate attention bias, while evicting recency bias tokens in the deeper layers redistributes the attention scores obtained by the recency bias tokens to the intermediate tokens. iv): Simultaneously evicting sink bias and recency bias tokens can alleviate attention bias in the intermediate layers (Layer 16). v): As shown in Figure 6, evicting tokens with abnormally high attention scores appears to effectively mitigate attention bias within the model. However, the impact of this strategy on taskspecific performance remains uncertain. We will investigate this further in our experiments.

## 5. Experiment

## 5.1. Experimental Settings

Models, Baselines, and Tasks. We compare our method with existing length extrapolation approaches, including Position Interpolation (PI) (Chen et al., 2023b), NTK-Aware (LocalLLaMA, 2023b), ChunkLlama (An et al., 2024), AttenCalibration (Yu et al., 2024), APE (Yang et al., 2025b), StarAttention (Acharya et al., 2024), and

InfLLM (Xiao et al., 2024), on LongBench (Bai et al., 2023) and InfiniteBench (Zhang et al., 2024), evaluating them on Llama2-7B-chat-hf (Touvron et al., 2023), LLaMA3.1 (Grattafiori et al., 2024), Qwen2.5 (Yang et al., 2025a) and Llama-3-8B-Instruction (AI, 2024). We also compare our method with the following opensource and closed-source models trained on long-context data: ChatGLM-3-6B-128K (GLM et al., 2024), Kimi-Chat (AI, 2023), Yi-6B-200K (01.AI, 2023a), Yi-34B-200K (01.AI, 2023b), Claude-2 (Anthropic, 2023), Yarn-Mistral-7b-128k (Peng et al., 2023), and GPT-4 (Achiam et al., 2023). Since AttenCalibration only calibrates the attention distribution and lacks the capability for length extrapolation, we incorporate NTK-aware techniques to enable this functionality, resulting in AttenCalibration-NTK. Details of our hyperparameters are provided in Appendix C.

#### 5.2. Length Extrapolation Settings

**Main results.** We present our method in Table 1, showing the performance of several strong baselines on LongBench. We have the following main findings: i): Our method is the *only one* that surpasses FullKV (i.e., the baseline without any length extrapolation) across different backbones. ii): Section 4 reveals that parallel KV cache compression exacerbates attention bias. However, combining it with the eviction  $R_H$  method to calibrate the attention distribution, i.e., Ours-calibration-compression, can restore the performance to that of the original KV cache size. iii): Chunk-based length extrapolation methods, such as InfLLM and ChunkL-

<span id="page-7-0"></span>

|                              |        | Llama2   | -7B-chat-l | hf(4k) |        |            |         |
|------------------------------|--------|----------|------------|--------|--------|------------|---------|
| Methods                      | R.PK   | R.Num    | R.KV       | En.MC  | Math.F | Code.Debug | Average |
| Max Length                   | 125k   | 125k     | 175k       | 834k   | 120k   | 258k       | 273k    |
| FullKV                       | 1.36   | 1.86     | 0.4        | 0.44   | 17.43  | 21.57      | 7.18    |
| Dynamic-PI                   | 0.17   | 0.00     | 0.00       | 7.42   | 2.00   | 21.32      | 5.15    |
| NTK-Aware                    | 2.54   | 0.00     | 0.00       | 3.06   | 7.71   | 18.78      | 5.35    |
| ChunkLlama                   | 12.88  | 13.22    | 0.20       | 0.87   | 17.14  | 22.08      | 11.07   |
| InfLLM                       | 100.00 | 96.61    | 2.40       | 29.80  | 16.86  | 22.34      | 44.67   |
| AttenCalibration-NTK         | 0.00   | 0.00     | 0.00       | 1.06   | 5.71   | 19.24      | 4.34    |
| Ours                         | 100.00 | 97.63    | 20.60      | 33.62  | 19.71  | 25.13      | 49.45   |
| Ours-calibration             | 100.00 | 98.64    | 22.80      | 36.24  | 19.71  | 30.20      | 51.27   |
| Ours-compression             | 97.80  | 87.96    | 5.00       | 35.81  | 15.86  | 27.41      | 44.97   |
| Ours-calibration-compression | 97.97  | 90.14    | 10.80      | 35.46  | 15.86  | 28.21      | 46.41   |
|                              |        | Llama3   | 8B-instru  | ct(8k) |        |            |         |
| Methods                      | R.PK   | R.Num    | R.KV       | En.MC  | Math.F | Code.Debug | Averag  |
| FullKV                       | 6.10   | 6.27     | 4.80       | 42.79  | 38.57  | 22.34      | 20.15   |
| Dynamic-PI                   | 0.00   | 0.00     | 0.00       | 28.82  | 29.71  | 24.62      | 13.86   |
| NTK-Aware                    | 3.39   | 8.47     | 9.40       | 35.37  | 39.43  | 17.77      | 18.97   |
| ChunkLlama                   | 3.05   | 9.15     | 3.60       | 13.54  | 34.29  | 11.42      | 12.51   |
| AttenCalibration-NTK         | 4.58   | 8.47     | 12.40      | 34.28  | 36.57  | 22.68      | 19.83   |
| InfLLM                       | 100.00 | 99.00    | 5.00       | 43.70  | 23.70  | 22.08      | 48.91   |
| Ours                         | 100.00 | 99.83    | 92.80      | 54.59  | 40.00  | 22.84      | 68.34   |
| Ours-calibration             | 100.00 | 99.49    | 93.80      | 56.77  | 40.00  | 23.24      | 68.88   |
| Ours-compression             | 100.00 | 99.83    | 89.20      | 55.48  | 40.00  | 21.32      | 67.64   |
| Ours-calibration-compression | 100.00 | 99.83    | 91.00      | 56.77  | 40.00  | 22.20      | 68.30   |
|                              |        | Other pr | prietary   | models |        |            |         |
| Models                       | R.PK   | R.Num    | R.KV       | En.MC  | Math.F | Code.Debug | Averag  |
| GPT-4                        | 100.00 | 100.00   | 89.00      | 67.25  | 60.00  | 37.06      | 75.55   |
| Kimi-Chat                    | 98.14  | 95.42    | 53.60      | 72.49  | 12.57  | 17.14      | 58.23   |
| Claude-2                     | 97.8   | 98.14    | 65.40      | 62.88  | 32.29  | 17.77      | 62.38   |
|                              |        | Other op | en-source  | models |        |            |         |
| Models                       | R.PK   | R.Num    | R.KV       | En.MC  | Math.F | Code.Debug | Averag  |
| YaRN-Mistral-7B-128k         | 92.71  | 56.61    | < 5        | 27.95  | 17.14  | 60.00      | 42.82   |
| Yi-6B-200K                   | 100    | 94.92    | < 5        | 36.68  | < 5    | < 5        | 39.85   |
| Yi-34B-200K                  | 100    | 100      | < 5        | 38.43  | < 5    | 25.71      | 44.86   |
| ChatGLM-3-6B-128K            | 92.2   | 80.68    | < 5        | 10.48  | < 5    | 7.71       | 32.68   |

Table 2: The model's performance on the InfiniteBench dataset across different datasets.

<span id="page-7-1"></span>

|            |                        | L     | lama2-7B-  | chat-hf(4k) |            |            |            |  |  |  |  |  |  |  |  |
|------------|------------------------|-------|------------|-------------|------------|------------|------------|--|--|--|--|--|--|--|--|
| Methods    | 2k                     | 4k    | 8k         | 16k         | 32k        | 64k        | 128k       |  |  |  |  |  |  |  |  |
| Llama2-7b  | 7.03                   | 6.71  | $> 10^{2}$ | $> 10^{2}$  | $> 10^{2}$ | $> 10^{2}$ | $> 10^{2}$ |  |  |  |  |  |  |  |  |
| Dynamic-PI | 7.03                   | 6.71  | 7.02       | 11.62       | 59.31      | $> 10^{2}$ | $> 10^{2}$ |  |  |  |  |  |  |  |  |
| NTK-Aware  | 8.61                   | 8.41  | 8.29       | 7.19        | 40.71      | $> 10^{2}$ | $> 10^{2}$ |  |  |  |  |  |  |  |  |
| ChunkLlama | 7.03                   | 6.71  | 6.42       | 5.01        | 4.82       | 12.36      | 43.57      |  |  |  |  |  |  |  |  |
| InfLLM     | 23.24                  | 23.46 | 21.86      | 20.40       | 19.84      | 18.26      | 18.97      |  |  |  |  |  |  |  |  |
| Ours       | 8.01                   | 9.71  | 11.97      | 10.46       | 11.34      | 11.58      | 12.56      |  |  |  |  |  |  |  |  |
|            | Llama3-8B-instruct(8k) |       |            |             |            |            |            |  |  |  |  |  |  |  |  |
| Methods    | 2k                     | 4k    | 8k         | 16k         | 32k        | 64k        | 128k       |  |  |  |  |  |  |  |  |
| Llama3-8b  | 9.90                   | 9.15  | 7.94       | 63.13       | $> 10^{2}$ | $> 10^{2}$ | $> 10^{2}$ |  |  |  |  |  |  |  |  |
| Dynamic-PI | 9.90                   | 9.15  | 17.25      | 69.96       | $> 10^{2}$ | $> 10^{2}$ | $> 10^{2}$ |  |  |  |  |  |  |  |  |
| NTK-Aware  | 10.71                  | 9.66  | 8.16       | 6.74        | 8.06       | 77.63      | $> 10^{2}$ |  |  |  |  |  |  |  |  |
| ChunkLlama | 9.88                   | 9.14  | 7.92       | 6.57        | 6.13       | 5.33       | 5.40       |  |  |  |  |  |  |  |  |
| InfLLM     | 8.50                   | 9.30  | 8.72       | 9.47        | 8.98       | 9.66       | 9.10       |  |  |  |  |  |  |  |  |
| Ours       | 5.85                   | 6.75  | 6.65       | 6.30        | 5.61       | 5.13       | 5.72       |  |  |  |  |  |  |  |  |

Table 3: We test the perplexity on the NarrativeQA (Kočiskỳ et al., 2018) test set.

lama, generally perform better than position encoding-based methods such as Dynamic-PI and NTK-Aware. *iv)*: Directly calibrating the attention distribution in NTK-aware length extrapolation methods, such as *AttenCalibration-NTK*, leads to strong performance primarily on the longest datasets, including NtrvQA, GovReport, and RB-P. This suggests that the effect of attention distribution calibration becomes increasingly significant as input length grows.

**Extrapolating beyond 128K context lengths.** We evaluate the performance under extremely long contexts in Table 2, comparing it with several powerful open-source and closed-source models. These models are trained on context lengths exceeding 128K, and thus do not require additional extrapolation capabilities to handle ultra-long contexts. We have the following findings: *i*): Our method performs excep-

tionally well on needle-in-a-haystack retrieval tasks (R.PK, R.Num, R.KV), being the *only model* capable of achieving over 90% accuracy across all tasks, surpassing even the strongest closed-source model, GPT-4. *ii*): Position encoding-based length extrapolation methods, such as NTK-Aware, Dynamic-PI, generally struggle to achieve good performance on tasks with ultra-long contexts compared to chunk-based extrapolation approaches. *iii*): Our training-free extrapolation method, using an 8K window, is the *only approach* that surpasses the powerful closed-source models Kimi-Chat and Claude-2, achieving 91.17% of GPT-4's performance on ultra-long contexts with an 8B model.

**Language modeling.** To further compare the performance of our method in language modeling, we present the results of perplexity (PPL) calculations on the NarrativeQA test set in Table 3, which reflect the model's performance in long-context language modeling. For fair comparison, we typically calculate the PPL for the query chunk, as it corresponds to the model's decoding phase. i): Chunk-based position extrapolation methods (ChunkLlama, InfLLM, and Ours) achieve significantly lower PPL compared to position encoding-based methods (Dynamic-PI and NTK-Aware). ii): Position encoding-based methods start to collapse in performance for language modeling when the length exceeds 32k. iii): As the number of chunks increases (from 2K to 128K), our method still demonstrates consistent perplexity stability across different lengths. Surprisingly, ChunkLlama maintains high performance on Llama3-8B-instruct, outperforming other methods.

#### 5.3. Evaluation in Long-context Models

To demonstrate the effectiveness of our method for longcontext models, we evaluate it on two models trained with contexts up to 128K, which therefore do not require any extrapolation capabilities: LLaMA3.1 (Grattafiori et al., 2024) and Qwen2.5 (Yang et al., 2025a), used as base models. All evaluations are conducted under a consistent 24K KV cache size. While standard baselines directly utilize the full 24K position encodings, our method applies extrapolation techniques to reuse a 6K position encoding, thereby supporting longer input lengths without modifying the model architecture. To ensure fairness, we standardize the use of special prompt tokens across models (e.g., < | begin\_of\_text| > for LLaMA3.1), which we observe to have a significant impact on performance. All reported results correspond to configurations that include these prompt tokens.

As shown in Table 4 and Table 5, our PARALLELCOMP method consistently improves performance across a broad range of tasks. On LongBench, we observe modest but consistent gains compared to baselines. On InfiniteBench, which emphasizes ultral-long context understanding, our method demonstrates significant improvements, particularly

<span id="page-8-0"></span>

| Method                | NARR  | QAS   | MULT  | HOPT  | 2WKI  | MUS   | GOV   | QMS   | NEWS  | TREC  | TRIV  | SSM   | PCNNT | PREN   | LCC   | REP   | AVG   |
|-----------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|-------|-------|-------|
| Llama3.1              | 27.80 | 44.25 | 49.46 | 47.86 | 40.54 | 23.64 | 32.64 | 22.90 | 26.90 | 38.00 | 88.44 | 25.64 | 2.02  | 92.00  | 10.35 | 18.64 | 36.94 |
| ParallelComp-Llama3.1 | 29.45 | 45.98 | 50.67 | 48.36 | 46.56 | 23.32 | 32.60 | 24.29 | 27.34 | 38.50 | 86.72 | 25.93 | 0.05  | 95.00  | 14.15 | 21.42 | 38.15 |
| Qwen2.5               | 27.83 | 41.31 | 50.41 | 53.52 | 44.68 | 30.00 | 33.38 | 24.01 | 25.40 | 71.00 | 86.10 | 39.91 | 7.25  | 100.00 | 6.86  | 7.88  | 40.60 |
| ParallelComp-Qwen2.5  | 28.42 | 42.24 | 50.54 | 56.26 | 42.02 | 28.25 | 33.43 | 23.20 | 25.20 | 71.50 | 89.21 | 41.84 | 5.00  | 93.50  | 20.73 | 13.34 | 41.54 |

Table 4: Performance on LongBench benchmark. Models are evaluated under a 24K KV cache budget.

<span id="page-8-1"></span>

| Method                | PS     | NUM   | KV    | EN.MC | MATH  | CODE  | AVG   |
|-----------------------|--------|-------|-------|-------|-------|-------|-------|
| Llama3.1              | 5.59   | 26.25 | 18.60 | 32.86 | 31.52 | 22.56 | 26.36 |
| ParallelComp-Llama3.1 | 100.00 | 83.56 | 88.60 | 66.38 | 37.14 | 22.08 | 59.55 |
| Qwen2.5               | 59.32  | 58.31 | 33.80 | 61.39 | 85.71 | 23.76 | 53.72 |
| ParallelComp-Qwen2.5  | 100.00 | 76.27 | 63.40 | 66.86 | 92.57 | 24.75 | 70.64 |

Table 5: Performance on InfiniteBench with different models. Models are evaluated under a 24K KV cache budget.

for tasks such as PS, NUM, and KV. These results indicate that extrapolated position encodings can be effectively reused in parts of the model's position encodings to extend context length, achieving comparable or even improved performance relative to using full position encodings in long-context models.

<span id="page-8-2"></span>

|                              |              | Llama2        | -7B-chat-           | hf(4k)        |                |                    |                 |
|------------------------------|--------------|---------------|---------------------|---------------|----------------|--------------------|-----------------|
| Methods<br>Max Length        | R.PK<br>125k | R.Num<br>125k | <b>R.KV</b><br>175k | En.MC<br>834k | Math.F<br>120k | Code.Debug<br>258k | Average<br>273k |
| Ours                         | 100.00       | 97.63         | 20.60               | 33.62         | 19.71          | 25.13              | 49.45           |
| Ours-calibration             | 100.00       | 98.64         | 22.80               | 36.24         | 19.71          | 30.20              | 51.27           |
| Sink-eviction-layer-1-8      | 99.32        | 42.71         | 2.20                | 37.12         | 17.71          | 22.84              | 36.98           |
| Sink-eviction-layer-9-16     | 100.00       | 91.19         | 11.00               | 37.12         | 14.86          | 24.37              | 46.42           |
| Sink-eviction-layer-17-23    | 100.00       | 97.80         | 20.80               | 33.19         | 19.14          | 30.96              | 50.32           |
| Sink-eviction-layer-24-31    | 100.00       | 97.63         | 20.20               | 31.88         | 18.00          | 29.19              | 49.48           |
| Recency-eviction-layer-1-8   | 100.00       | 96.44         | 2.60                | 33.19         | 16.00          | 19.54              | 44.63           |
| Recency-eviction-layer-9-16  | 100.00       | 97.80         | 15.80               | 37.99         | 10.86          | 23.10              | 47.59           |
| Recency-eviction-layer-17-23 | 100.00       | 97.97         | 20.40               | 23.58         | 16.00          | 32.74              | 48.45           |
| Recency-eviction-layer-24-31 | 100.00       | 97.63         | 20.60               | 35.81         | 18.57          | 25.89              | 49.75           |
| Middle-eviction-layer-1-8    | 100.00       | 97.29         | 20.60               | 34.93         | 18.29          | 22.84              | 48.99           |
| Middle-eviction-layer-9-16   | 100.00       | 97.63         | 20.60               | 33.62         | 16.00          | 26.40              | 49.04           |
| Middle-eviction-layer-17-23  | 98.81        | 97.46         | 20.00               | 34.06         | 19.14          | 30.20              | 49.95           |
| Middle-eviction-layer-24-31  | 100.00       | 97.46         | 19.80               | 30.13         | 19.43          | 28.93              | 49.29           |

Table 6: Ablation of Llama2-7B-chat-hf on InfiniteBench. Ours-calibration refers to the approach where layers 9-16 adopt the recency bias token eviction method, while layers 25-32 evict sink bias tokens, and layers 1-8 evict middle bias tokens. Other methods follow the naming format [Evicted Token Type]-eviction-layer-[Evicted Layer Range].

## 5.4. Ablation of Attention Bias

We present in Table 6 the impact of evicting different bias tokens at various layers on different tasks. We have the following observations: i): The  $R_s$  in the shallow layers (1-8) is crucial for retrieval tasks. Without these tokens, the model's performance will be significantly impaired. ii): The  $R_r$  in the deeper layers (layers 9-16) plays a crucial role in the model's reasoning abilities. Evicting these tokens results in a decline in performance on coding and math tasks. iii): Shallow  $R_m$  (layers 1-8) damages the model's understanding ability, and evicting them can improve the model's performance. Deep  $R_m$  (layers 24-31) contributes to the model's ability in reading comprehension tasks (En.MC), and evicting them harms the model's performance. iv):  $R_r$  in the early layers (layers 1-8) is important for the model's in-context learning ability. For a detailed analysis of this

phenomenon, please refer to Appendix B.

<span id="page-8-3"></span>

| Chunk Number | Ours                                |                      | Ours-Compre                         | ession               |
|--------------|-------------------------------------|----------------------|-------------------------------------|----------------------|
| Chunk Number | Prefill (s) / Generation (ms/token) | Max Memory Used (MB) | Prefill (s) / Generation (ms/token) | Max Memory Used (MB) |
| 1            | 1317.72 / 24.30                     | 19394                | 1317.72 / 22.16                     | 16994                |
| 4            | 321.40 / 43.69                      | 35518                | 321.40 / 23.28                      | 24734                |
| 8            | 160.70 / 72.53                      | 47758                | 160.70 / 31.21                      | 36396                |
| 12           | 111.67 / 102.94                     | 65980                | 111.67 / 39.61                      | 48458                |
| 16           | N/A                                 | Out-of-Memory        | 82.36 / 49.25                       | 59140                |
| 20           | N/A                                 | Out-of-Memory        | 65.23 / 56.25                       | 71302                |
| 23           | N/A                                 | Out-of-Memory        | 56.07 / 57.19                       | 79742                |
| 24           | N/A                                 | Out-of-Memory        | N/A                                 | Out-of-Memory        |

Table 7: Throughput analysis. We evaluate on Llama2-7B-chat-hf and compare the improvement in chunk throughput with the use of parallel KV cache compression. Time tests were performed on the NarrativeQA dataset. Experiments are conducted on an AMD Instinct MI210 64GB GPU.

## 5.5. Throughput Analysis

We mainly focus on the throughput of chunks during context parallelism. Therefore, we compare the maximum number of parallel chunks and the memory usage before and after parallel KV cache compression. Table 7 presents the memory usage of the model using the parallel KV cache eviction strategy. On a single GPU, by compressing the KV cache size of each chunk to half of its original size, we achieve a 1.76x improvement in chunk throughput, thereby achieving a 23.50x acceleration in the prefill stage with negligible performance loss.

#### 6. Conclusion

In this paper, we propose PARALLELCOMP, a training-free and parallel long-context compression framework that significantly enhances the extrapolation capability of large language models (LLMs) for ultra-long contexts. PAR-ALLELCOMP overcomes the critical memory bottlenecks in length extrapolation and systematically analyzes the attention bias that arises in such settings. Specifically, our method allows 8B LLMs to extend inference length from 4K to 128K tokens on a single A100 80GB GPU without retraining or significant degradation in performance. By leveraging chunk-based parallel attention, dynamic KV cache eviction, and an attention calibration strategy, our approach alleviates both excessive memory usage and the attention sink phenomenon. Extensive theoretical and empirical results demonstrate that PARALLELCOMP effectively mitigates attention bias and enables robust, end-to-end inference. Notably, our method achieves 91.17% of GPT-4's performance on ultra-long context tasks using an 8B model, outperforming various state-of-the-art closed-source models. These findings pave the way for more scalable and efficient long-context inference.

