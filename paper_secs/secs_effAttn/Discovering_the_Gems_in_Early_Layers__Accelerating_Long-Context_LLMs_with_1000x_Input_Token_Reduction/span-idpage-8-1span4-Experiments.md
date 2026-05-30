# <span id="page-8-1"></span>4 Experiments

Model and Datasets. We evaluated our approach using three popular long-context models: LLaMA 3.1 8B Instruct[2](#page-8-4) [\[DJP](#page-13-4)+24], Mistral Nemo 12B Instruct[3](#page-8-5) [\[JSM](#page-14-3)+23], and Phi 3.5 Mini 3.8B Instruct[4](#page-8-6) [\[AJA](#page-13-5)+24], all of which support an input token length of 128K. We compared our method, GemFilter, against standard attention and two state-of-the-art methods, SnapKV [\[LHY](#page-14-4)+24] and H2O [\[ZSZ](#page-15-1)+23] [5](#page-8-7) . For our experiments, we used two popular datasets: Needle in a Haystack [\[Kam24\]](#page-14-10) (Section [4.1\)](#page-8-2) and LongBench [\[BLZ](#page-13-7)+23] (Section [4.2\)](#page-10-0). More implementation details are provided in Appendix [C.2.](#page-17-2)

Filter Layer. Except Section [4.3,](#page-11-0) for context selection, we always use the index of 13 out of 32, 19 out of 40, and 19 out of 32 layers as the input filter for LLaMA 3.1, Mistral Nemo and Phi 3.5, respectively. In Section [4.3,](#page-11-0) we provide an ablation study for the filter layer choice.

### <span id="page-8-2"></span>4.1 Needle in a Haystack

The Needle in a Haystack [\[Kam24\]](#page-14-10) benchmark serves as a pressure test, challenging LLMs to retrieve accurate information from a specific sentence (the 'needle') hidden within an extensive document

<span id="page-8-3"></span><sup>1</sup>RoPE is the rotary positional embedding [\[SAL](#page-14-11)<sup>+</sup>24], encoding the positional information of tokens.

<span id="page-8-4"></span><sup>2</sup> <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct>

<span id="page-8-5"></span><sup>3</sup> <https://huggingface.co/mistralai/Mistral-Nemo-Base-2407>

<span id="page-8-7"></span><span id="page-8-6"></span><sup>4</sup> <https://huggingface.co/microsoft/Phi-3.5-mini-instruct>

<sup>5</sup>While there are many other generation acceleration methods, they may not be directly comparable to ours as they use orthogonal techniques. We refer the reader to Section [2](#page-4-0) for further details.

<span id="page-9-0"></span>![](_page_9_Figure_0.jpeg)

(c) GemFilter-1024. Mistral Nemo average score: 0.838; LLaMA 3.1 average score: 0.887.

Figure 4: Needle in a Haystack performance comparison of different methods using the Mistral Nemo 12B Instruct model (left column) and the LLaMA 3.1 8B Instruct model (right column). Results for the Phi 3.5 Mini 3.8B Instruct model are provided in Appendix C.3. The x-axis represents the length of the input tokens, while the y-axis shows the position depth percentage of the 'needle' information (e.g., 0% indicates the beginning, and 100% indicates the end). A higher score reflects better performance, meaning more effective retrieval of the 'needle' information. GemFilter significantly outperforms both standard attention (full KV cache) and SnapKV.

(the 'haystack'), where the sentence can appear at any arbitrary location. The difficulty increases as the length of the haystack grows. We use input lengths of 60K for Mistral Nemo 12B Instruct and 120K for LLaMA 3.1 8B Instruct, as these are the maximum lengths for standard attention on two A100-40GB GPUs. The KV cache size is set to 1024 for both SnapKV and GemFilter. In Figure 4, we see that GemFilter significantly outperforms both All KV (standard attention) and SnapKV with Mistral Nemo and LLaMA 3.1.6 The Needle in a Haystack results suggest that our method, GemFilter, achieves superior retrieval performance for long input contexts compared to

<span id="page-9-1"></span><sup>&</sup>lt;sup>6</sup>H2O cannot be implemented with FlashAttention due to its cumulative attention score strategy and is therefore unable to handle super long input contexts, which is why we exclude it here, following [LHY<sup>+</sup>24, XJD<sup>+</sup>24].

<span id="page-10-1"></span>Table 1: Performance comparison on LongBench across various LLMs and methods. A larger number means better performance. The best score is **boldfaced**.

|                               | Single-Document QA    |                       |                       |                       | Multi-Document QA     |                       |                       | Summarization      |                       |                  | Few-shot Learning     |                       |                  | hetic                 |                    |
|-------------------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|--------------------|-----------------------|------------------|-----------------------|-----------------------|------------------|-----------------------|--------------------|
| Method                        | Hrty O.A              | Oasper                | MF-en                 | HotpotQA              | 2WikiMQA              | Musique               | GovReport             | OMSum.             | MultiNews             | TREC             | TriviaOA              | SAMSun                | PCount           | PRe                   | Average            |
|                               |                       |                       |                       |                       |                       | LaMA 3                | 1 8B Ir               |                    |                       |                  |                       |                       |                  |                       |                    |
| All KV<br>H2O-4096            | 32.02 $22.94$         | 13.04 $12.61$         | 27.34 $26.48$         | 16.23<br>16.63        | 16.05<br>15.81        | 11.22<br>10.14        | <b>34.52</b><br>33.51 | 23.41<br>23.47     | <b>26.89</b> 26.81    | <b>73.0</b> 69.0 | 91.64<br>91.15        | 43.8<br><b>43.97</b>  | 7.16<br>6.66     | 97.73<br>71.67        | <b>36.72</b> 33.63 |
| SnapKV-1024<br>GemFilter-1024 | 31.98<br>20.71        | 11.17<br>11.0         | 25.33<br><b>29.28</b> | 14.81<br>19.12        | 15.73<br>17.01        | 10.69<br>13.01        | 26.95<br>30.37        | 22.89<br>21.75     | 25.86<br>25.17        | 67.5<br>63.0     | 91.89<br>90.7         | 42.85<br>42.5         | <b>7.67</b> 7.15 | 98.16<br>92.22        | 35.25<br>34.50     |
| SnapKV-2048<br>GemFilter-2048 | 31.45<br>24.36        | 11.94<br>12.63        | 26.24<br>25.39        | 15.73<br><b>19.58</b> | 16.03<br><b>17.03</b> | 11.66<br><b>14.11</b> | 29.64<br>33.15        | 23.24<br>22.31     | 26.44<br>26.49        | 69.5<br>69.5     | 91.48<br>91.59        | 42.68<br>42.64        | 7.21<br>4.61     | 98.03<br><b>98.75</b> | 35.80<br>35.87     |
| SnapKV-4096<br>GemFilter-4096 |                       | <b>13.12</b> 12.95    | 27.38<br>27.38        | 16.11<br>17.76        | 16.08<br>15.6         | 11.6<br>12.02         | 32.39<br>34.17        | 23.47<br>23.25     | 26.76<br>26.87        | 71.5<br>70.0     | 91.64<br><b>92.36</b> | 43.46<br>43.34        | 7.33<br>5.96     | 97.24<br>98.0         | 36.44<br>36.09     |
|                               |                       |                       |                       |                       | Mis                   | tral Nen              | no 12B                | Instruct           | ŧ                     |                  |                       |                       |                  |                       |                    |
| All KV<br>H2O-4096            | 28.91<br><b>31.61</b> | 40.74 $39.52$         | $54.65 \\ 54.75$      | 52.15 $47.83$         | 48.36<br>48.09        | 30.28<br>27.0         | <b>30.66</b> 30.44    | <b>23.53</b> 23.21 | 26.31<br>26.42        | $75.0 \\ 72.5$   | 89.66<br>89.76        | $44.32 \\ 44.47$      | 4.5<br>3.0       | $100.0 \\ 73.0$       | $46.36 \\ 43.69$   |
| SnapKV-1024<br>GemFilter-1024 | 26.42<br>27.53        | 38.49<br>40.68        | 52.96<br>53.86        | 51.21<br>55.51        | 47.86<br><b>55.43</b> | 27.06<br>34.11        | 24.32<br>27.25        | 22.66<br>21.16     | 25.52<br>25.56        | 73.0<br>69.0     | 89.82<br>87.32        | 43.16<br>42.49        | 3.5<br>4.0       | 100.0<br>88.06        | 44.71<br>45.14     |
| SnapKV-2048<br>GemFilter-2048 | 25.85<br>29.27        | 40.69<br><b>41.53</b> | 54.48<br><b>54.91</b> | 51.96<br>57.62        | 49.06<br>54.97        | 26.95<br><b>35.09</b> | 26.29<br>29.34        | 23.17<br>22.58     | 25.9<br>26.19         | 74.5<br>72.0     | 89.66<br>89.65        | 43.89<br><b>44.93</b> | 4.0<br>4.0       | 99.5<br>97.5          | 45.42<br>47.11     |
| SnapKV-4096<br>GemFilter-4096 | 27.92<br>30.29        | 40.9<br>39.9          | 54.75<br>56.48        | 51.69<br><b>58.78</b> | 48.16<br>51.48        | 29.19<br>32.81        | 29.17<br>30.32        | 23.36<br>23.21     | 26.35<br><b>26.48</b> | 75.0<br>71.5     | 89.66<br><b>90.24</b> | 43.93<br>42.13        | 4.5<br>2.0       | 100.0<br>99.5         | 46.04<br>46.79     |
|                               |                       |                       |                       |                       | Phi                   | 3.5 Mir               | i 3.8B                | Instruct           |                       |                  |                       |                       |                  |                       |                    |
| All KV<br>H2O-4096            |                       | 17.23 $16.23$         | $35.63 \\ 34.17$      | $21.7 \\ 21.02$       | 25.7<br>23.05         | 11.68<br>10.49        | 34.14<br>33.42        | 23.17<br>21.95     | 24.95<br>24.95        | <b>71.5</b> 67.5 | $87.37 \\ 86.13$      | $13.08 \\ 16.71$      | <b>7.17</b> 1.55 | 83.85<br>47.46        | <b>34.62</b> 30.31 |
| SnapKV-1024<br>GemFilter-1024 | 24.31<br>16.57        | 16.03<br>18.29        | 34.93<br>35.91        | 20.72<br>24.22        | 26.02<br>26.1         | 13.74<br>9.7          | 28.27<br>30.29        | 22.03<br>18.96     | 24.02<br>23.64        | 67.5<br>64.5     | <b>87.71</b> 85.85    | 14.57<br><b>23.02</b> | 6.08<br>0.2      | <b>85.6</b> 81.12     | 33.68<br>32.74     |
| SnapKV-2048<br>GemFilter-2048 | 26.41<br>19.63        | 16.59<br>14.84        | <b>36.99</b> 35.99    | 21.8<br>21.38         | 26.07<br>19.72        | 12.57<br>10.13        | 30.88<br>32.39        | 22.37<br>21.24     | 24.51<br>24.71        | 69.5<br>65.0     | 87.54<br>86.49        | 13.13<br>20.47        | 6.57<br>2.17     | 83.92<br>69.5         | 34.20<br>31.69     |
| SnapKV-4096<br>GemFilter-4096 | 27.25<br>20.95        | 17.42<br><b>19.98</b> | $36.9 \\ 35.22$       | 21.37<br><b>28.82</b> | 25.42<br><b>28.21</b> | 12.55 <b>13.98</b>    | 32.9<br><b>34.2</b>   | $22.6 \\ 22.45$    | 24.87<br><b>25.08</b> | 70.5<br>64.5     | 87.45<br>85.86        | 13.28<br>18.68        | 6.81<br>3.43     | 84.04<br>65.56        | 34.53<br>33.35     |

SnapKV and standard attention. Additional results are provided in Appendix C.3.

#### <span id="page-10-0"></span>4.2 LongBench

LongBench [BLZ<sup>+</sup>23] is a multi-task benchmark designed to rigorously evaluate long-context understanding capabilities across various datasets, including single- and multi-document Question Answering (QA), summarization, few-shot learning, and synthetic tasks. We evaluate on the English-only dataset, following [LHY<sup>+</sup>24, XJD<sup>+</sup>24].

For each LLM, we evaluate GemFilter and SnapKV with selected tokens/KV caches of 1024, 2048, and 4096. We also evaluated standard attention (all KV cache) and H2O with a KV cache size of 4096 on the LongBench dataset to further demonstrate the performance of GemFilter, following [LHY<sup>+</sup>24]. Table 1 shows a negligible performance drop in LLMs using GemFilter compared to standard attention, even with only 1024 selected tokens. In some cases, GemFilter even outperforms standard attention, such as GemFilter-2048 for Mistral Nemo 12B Instruct. It demonstrates significantly better performance than H2O and comparable performance with SnapKV. Furthermore, GemFilter effectively filters key information in long contexts, provides interpretable summaries,

<span id="page-11-1"></span>![](_page_11_Figure_0.jpeg)

Figure 5: Distance between the needle position and selected token index position across three LLMs. The position depth percentage of the "needle" information is 50%. The x-axis means the layer index of different LLMs. The y-axis means min(topk\_index - niddle\_index). When y = 0, it means the needle information is covered by the selected token. The needle information has been successfully discovered in the early layers of all three LLMs.

and compresses the input context effectively, e.g., it reduces input tokens to an average of 8% when using 1024 tokens, and 32% when using 4096, with negligible accuracy drops.

#### <span id="page-11-0"></span>4.3 Filter Layer Choice

In this section, we explore which layer should be chosen as the input filter. First, we aim to determine which layer of the LLM can best identify the position of the needle information. In Figure 5, we plot the distance between the needle's position and the selected token index across all layers in the LLM. The results reveal three stages in the prompt computation of LLMs. In the first stage, the initial layers preprocess the input context and search for the 'needle'. In the second stage, some early to middle layers identify the needle information. Finally, in the third stage, the LLM prepares to generate the output based on the selected tokens.

<span id="page-11-2"></span>Table 2: Performance of our method on LongBench using different layers as an input filter. A larger number means better performance. The best score is **boldfaced**.

|              | Single    | -Docum | ent QA | Multi-   | Documen  | t QA    | Sur       | nmarizat | tion      | Few  | -shot Le | arning | Synt   | hetic |         |
|--------------|-----------|--------|--------|----------|----------|---------|-----------|----------|-----------|------|----------|--------|--------|-------|---------|
| Filter layer | Arty O.A. | Qasper | Mren   | HotpotQA | ZWikiMQA | Musique | GovReport | OMSum    | MultiNews | TREC | TriviaOA | SAMSum | PCount | PRe   | Average |
|              |           |        |        |          | LLaM     | A 3.1 8 | B Instru  | uct (32  | layers)   |      |          |        |        |       |         |
| layer-1      | 16.32     | 7.38   | 13.86  | 13.9     | 13.21    | 5.22    | 25.61     | 20.09    | 24.51     | 47.0 | 76.59    | 39.78  | 2.55   | 23.01 | 23.50   |
| layer-7      | 16.89     | 6.83   | 13.47  | 13.78    | 12.23    | 9.67    | 26.56     | 19.49    | 24.55     | 58.0 | 84.87    | 41.07  | 6.5    | 50.69 | 27.47   |
| layer-12     | 15.53     | 7.73   | 16.53  | 17.08    | 13.33    | 9.88    | 28.94     | 20.32    | 25.01     | 58.0 | 88.16    | 40.42  | 8.36   | 43.06 | 28.03   |
| layer-13     | 20.71     | 11.0   | 29.28  | 19.12    | 17.01    | 13.01   | 30.37     | 21.75    | 25.17     | 63.0 | 90.7     | 42.5   | 7.15   | 92.22 | 34.50   |
| layer-14     | 21.14     | 13.06  | 25.45  | 20.89    | 17.32    | 12.9    | 29.85     | 22.06    | 24.91     | 62.0 | 89.88    | 42.33  | 6.17   | 92.17 | 34.30   |
| layer-19     | 19.06     | 11.69  | 27.12  | 20.98    | 16.98    | 14.04   | 29.17     | 21.88    | 25.18     | 58.0 | 89.65    | 40.4   | 8.75   | 94.84 | 34.12   |
| layer-25     | 24.74     | 12.33  | 26.18  | 18.56    | 16.3     | 12.54   | 28.66     | 21.75    | 25.14     | 61.5 | 88.78    | 39.47  | 8.67   | 90.59 | 33.94   |
| layer-31     | 20.62     | 9.13   | 17.51  | 19.13    | 13.76    | 10.07   | 28.21     | 21.11    | 25.16     | 58.0 | 88.4     | 42.37  | 8.23   | 58.8  | 30.04   |

We then use the first layer that accurately identifies the needle's position as the input filter. In our experiments, we find that this layer remains consistent across different inputs. As shown in Table 2, performance first increases and then decreases as we select the input filter layer from the beginning to the end. The peak performance is observed at the 13th layer, which supports our layer

selection strategy. Performance remains robust between layers 13 and 25, providing flexibility in layer selection. Exploring the distinct functions of different layers presents an interesting direction for future research.

### <span id="page-12-0"></span>4.4 Running Time and GPU Memory Consumption

In this section, we compare the running time and GPU memory consumption of different methods with FlashAttention [DFE<sup>+</sup>22, Dao23, SBZ<sup>+</sup>24] support.<sup>7</sup> As shown in Figure 3, our method, GemFilter, achieves a 2.4× speedup compared to SnapKV and standard attention, with 30% and 70% reductions in GPU memory usage, respectively. It saves both running time and GPU memory by processing the long input context only during the first stage, as described in Section 4.3. For the latter two stages, the LLMs only need to handle compressed inputs. In Figure 6, we present a comparison of running time and GPU memory consumption for Mistral Nemo 12B Instruct and Phi 3.5 Mini 3.8B Instruct using various methods. GemFilter runs faster and uses less GPU memory than the state-of-the-art methods, as discussed above. Additionally, Figure 3 and Figure 6 further support our Theorem 3.3 in Section 3.3.

<span id="page-12-2"></span>![](_page_12_Figure_3.jpeg)

Figure 6: Comparison of time and GPU memory usage across different methods on Mistral Nemo 12B Instruct and Phi 3.5 Mini 3.8B Instruct. GemFilter uses the 19th layer as an input filter for both LLMs. It achieves a  $2.4\times$  speedup and reduces GPU memory usage by 30% compared to SnapKV.

<span id="page-12-1"></span> $<sup>^{7}</sup>$ We exclude H2O as it does not support FlashAttention and thus requires more GPU memory and running time than standard attention during prompt computation.

