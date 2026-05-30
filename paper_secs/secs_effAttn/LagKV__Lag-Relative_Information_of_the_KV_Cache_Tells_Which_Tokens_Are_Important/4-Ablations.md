# 4 Ablations

In this section, we fix the sink size to S = 16 and vary the lag size L and retention ratio r. The values of L will be L = 128, 512 and 1024. The values of r will be 2×, 4×, 6× and 8× which correspond to r = 0.5, 0.25, 0.167, and 0.125 respectively. Aslo, we will alter the prefilling method to prove the stability of our approach and scoring method for the validity of lag information.

Datasets. We use the facility in [\(Yuan et al.,](#page-9-15) [2024\)](#page-9-15) to extensively test our method. It contains two benchmarks: LongBench [\(Bai et al.,](#page-8-9) [2024\)](#page-8-9) and Needle-in-a-HaystackTest with Passkey-Retrieval in a Paul Graham Essays background [\(Kamradt,](#page-8-10) [2023;](#page-8-10) [Mohtashami and Jaggi,](#page-9-16) [2023\)](#page-9-16). We only test the 64-digit passkey retrieval task which is much more challenging. And because we are using a recursive and evicting compression strategy, it's easier to illustrate some insights with the partial match score other than the exact one in their report. Therefore, the default needle score will be the partial score throughout the work unless otherwise specified. The main result of this ablation is Table [3.](#page-5-0)

Prefill stage. By default, like many other compression methods, compression begins after prefill completes for each layer. This is an efficient and accurate approach—preserving both the KV cache values and the first generated token (FGT) while reducing KV cache size. However, since we lack a reliable benchmark for long-context and longgeneration scenarios, we will extend the passkey retrieval task by enabling chunk-by-chunk compression during prefill. This will help us evaluate how compression impacts long-generation performance, especially the FGT. Also, this chunked prefilling method will be useful for extreme long context processing.

#### 4.1 LongBench

For the LongBench dataset, the method performs very well across different ratios and lag sizes. When L = 1024, r = 8×, the method still retains approximate 90% of the baseline performance. Since the compression ratio will increase when L decreases, the worse case is L = 128, r = 8× for both models but the method maintains at least 85% of the baseline performance.

#### 4.2 Passkey Retrieval

The 64-digit passkey retrieval task is a challenging one for most token eviction strategies. As discussed in [\(Yuan et al.,](#page-9-15) [2024\)](#page-9-15), the most successful eviction strategy H2O [\(Zhang et al.,](#page-9-2) [2024\)](#page-9-2) performs well in 7-digit task (scoring 100% for all compression ratios) but degrades a lot in the 64-digit one (for 4× in Llama-3, exact match score is 35% and partial match score is 70.8%). It happens because the strategy applies its compression after the prefill is done which means the FGT is not affected by the compression and the 7-digit passkey usually takes only 2 or 3 tokens. When the passkey size increases to 64, much more generated tokens are impacted by the compression. Many token-evict algorithms are struggling to maintain their performance in this case. In contrast, our method performs very well when the product of r and L is sufficient large enough (for L = 1024, r = 4× in Llama model, exact math score is 89% and partial match score is 96.57%).

Our recursive compression strategy will not perform well for the setups with small rL due to the fact that when the recursive window size is compressed to be close to or less than the length of

Table 1: RULER-16K Results of Llama-3.1-8B-Instruct

<span id="page-4-0"></span>

| Comp. Ratio | Method                          | SK1                          | SK2                         | SK3                         | MK1                         | MK2                         | MK3                         | MV                          | MQ                          | VT                           | CWE                        | FWE                         | QA1                                | QA2                         | AVERAGE                     |
|-------------|---------------------------------|------------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|------------------------------|----------------------------|-----------------------------|------------------------------------|-----------------------------|-----------------------------|
| 0.0         | FullKV                          | 100.0                        | 100.0                       | 100.0                       | 97.4                        | 100.0                       | 100.0                       | 100.0                       | 98.2                        | 100.0                        | 90.2                       | 87.5                        | 75.7                               | 54.7                        | 92.6                        |
| 0.25        | SnapKV<br>StreamingLLM<br>LagKV | 100.0<br>72.5<br>100.0       | 100.0<br>74.7<br>100.0      | 33.3<br>72.5<br><b>95.7</b> | <b>98.7</b> 79.2 97.4       | 83.3<br>86.7<br><b>96.7</b> | 63.9<br><b>66.7</b><br>56.9 | 97.9<br>72.7<br><b>99.4</b> | 98.2<br>75.0<br><b>98.5</b> | 94.8<br>90.5<br><b>100.0</b> | 85.3<br>0.1<br><b>88.4</b> | <b>90.2</b><br>87.1<br>89.0 | 64.9<br><b>75.7</b><br>74.3        | 46.9<br>43.8<br><b>50.0</b> | 81.3<br>69.0<br><b>88.2</b> |
| 0.5         | SnapKV<br>StreamingLLM<br>LagKV | 100.0<br>47.2<br>100.0       | 94.2<br>46.0<br><b>98.8</b> | 15.9<br>46.4<br><b>88.4</b> | 93.5<br>53.2<br><b>98.7</b> | 48.3<br>50.0<br><b>81.7</b> | 15.3<br><b>44.4</b><br>13.9 | 77.9<br>48.5<br><b>97.9</b> | 87.8<br>52.4<br><b>98.5</b> | 94.8<br>69.5<br><b>98.7</b>  | <b>72.3</b> 1.6 65.7       | 85.5<br>83.1<br><b>86.3</b> | 44.6<br><b>75.7</b><br>66.2        | 37.5<br>35.9<br><b>45.3</b> | 66.8<br>50.3<br><b>80.0</b> |
| 0.75        | SnapKV<br>StreamingLLM<br>LagKV | 93.4<br>28.6<br><b>100.0</b> | 79.3<br>21.8<br><b>98.8</b> | 4.3<br>20.3<br><b>46.4</b>  | 52.0<br>33.8<br><b>90.9</b> | 26.7<br>26.7<br><b>33.3</b> | 1.4<br><b>23.6</b><br>1.4   | 33.5<br>23.2<br><b>86.2</b> | 35.7<br>27.1<br><b>92.4</b> | 83.0<br>43.6<br><b>96.1</b>  | 17.1<br>0.9<br>10.9        | 77.2<br><b>80.4</b><br>73.3 | 28.4<br>33.8<br><b>46.0</b>        | 26.6<br>29.7<br><b>42.2</b> | 43.0<br>30.3<br><b>62.9</b> |
| 0.875       | SnapKV<br>StreamingLLM<br>LagKV | 85.7<br>12.1<br><b>95.6</b>  | 43.7<br>13.8<br><b>77.0</b> | 4.3<br><b>11.6</b><br>5.8   | 26.0<br>26.0<br><b>75.3</b> | 15.0<br><b>16.7</b><br>8.3  | 1.4<br><b>15.3</b><br>1.4   | 17.1<br>12.3<br><b>70.6</b> | 14.3<br>13.4<br><b>80.8</b> | 61.3<br>20.7<br><b>89.8</b>  | 1.9<br>0.9<br>1.5          | 66.3<br><b>75.3</b><br>62.0 | 18.9<br><b>29.7</b><br><b>29.7</b> | 26.6<br>29.7<br><b>37.5</b> | 29.4<br>21.3<br><b>48.9</b> |

Table 2: RULER-16K Results of Qwen2.5-7B-Instruct

<span id="page-4-1"></span>

| Comp. Ratio | Method       | SK1   | SK2   | SK3   | MK1  | MK2  | MK3  | MV   | MQ    | VT   | CWE  | FWE  | QA1  | QA2  | AVERAGE |
|-------------|--------------|-------|-------|-------|------|------|------|------|-------|------|------|------|------|------|---------|
| 0.0         | FullKV       | 100.0 | 100.0 | 100.0 | 99.2 | 99.1 | 94.2 | 94.3 | 100.0 | 99.0 | 79.9 | 93.2 | 72.6 | 48.2 | 90.8    |
| 0.25        | SnapKV       | 88.2  | 90.6  | 4.5   | 44.5 | 57.1 | 50.0 | 39.8 | 44.9  | 92.2 | 80.1 | 92.8 | 62.9 | 42.0 | 60.7    |
|             | StreamingLLM | 76.3  | 72.5  | 75.9  | 78.9 | 79.5 | 67.5 | 71.1 | 74.4  | 76.8 | 74.1 | 89.3 | 69.3 | 36.6 | 72.5    |
|             | LagKV        | 100.0 | 99.3  | 86.6  | 98.4 | 88.4 | 24.2 | 93.9 | 99.4  | 99.0 | 79.3 | 92.1 | 66.1 | 45.5 | 82.5    |
| 0.5         | SnapKV       | 86.8  | 64.5  | 3.6   | 24.2 | 28.6 | 9.2  | 21.9 | 23.0  | 90.8 | 77.8 | 92.3 | 40.3 | 35.7 | 46.1    |
|             | StreamingLLM | 50.7  | 42.8  | 52.7  | 52.3 | 45.5 | 45.8 | 48.8 | 51.2  | 61.5 | 72.3 | 88.3 | 72.6 | 33.0 | 55.2    |
|             | LagKV        | 100.0 | 97.8  | 48.2  | 98.4 | 54.5 | 3.3  | 93.9 | 95.3  | 98.6 | 74.4 | 89.5 | 58.1 | 42.9 | 73.5    |
|             | SnapKV       | 82.2  | 18.8  | 3.6   | 13.3 | 10.7 | 4.2  | 13.2 | 12.2  | 79.7 | 64.4 | 89.3 | 27.4 | 27.7 | 34.4    |
| 0.75        | StreamingLLM | 25.0  | 20.3  | 22.3  | 28.9 | 25.9 | 20.0 | 24.4 | 25.8  | 35.8 | 66.9 | 82.3 | 32.3 | 24.1 | 33.4    |
|             | LagKV        | 99.3  | 87.7  | 8.9   | 85.2 | 6.2  | 0.8  | 86.2 | 83.5  | 95.6 | 43.8 | 69.5 | 39.5 | 29.5 | 56.6    |
| 0.875       | SnapKV       | 69.7  | 8.0   | 3.6   | 14.1 | 6.2  | 0.8  | 11.6 | 11.4  | 55.1 | 47.7 | 79.2 | 19.4 | 22.3 | 26.9    |
|             | StreamingLLM | 11.2  | 13.8  | 14.3  | 18.8 | 15.2 | 12.5 | 13.4 | 14.0  | 20.7 | 56.1 | 78.1 | 21.0 | 18.8 | 23.7    |
|             | LagKV        | 99.3  | 62.3  | 3.6   | 56.2 | 0.0  | 0.8  | 60.2 | 50.6  | 93.0 | 18.8 | 55.2 | 29.0 | 22.3 | 42.4    |

the queried content, it's highly possible that only a small portion of the wanted information will be kept. In the task of 64-digit passkey retrieval, because digits usually require more tokens to be represented than the same length words, the number of expected tokens is much larger than the similar tasks in LongBench sub tasks like Document QA, that leads to its results are more sensitive to small rL. As shown in Fig. 2, the Qwen model which uses one token for one digit degenerates faster than the Llama model which represents three digits by one token with smaller rL. It hints us that we must choose the compression ratio and the lag size carefully in considering the length of the expected content and the tokenizer of the LLM. A.1 shows all the details of the needle results.

