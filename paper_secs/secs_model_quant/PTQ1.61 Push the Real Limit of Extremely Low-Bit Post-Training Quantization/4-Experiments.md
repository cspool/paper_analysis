# 4 Experiments

In this section, we conduct extensive experiments to validate our novel extremely low-bit PTQ method PTQ*1.61* on various benchmarks and LLMs with existing methods to demonstrate that our approach achieves outstanding performance under extremely challenging quantization.

#### 4.1 Experimental Setup

Baseline Since our PTQ*1.61* is an extremely low-bit PTQ method, we primarily choose PB-LLM (10% 8-bit) [\(Shang et al.,](#page-10-6) [2023\)](#page-10-6) and BiLLM [\(Huang et al.,](#page-9-4) [2024\)](#page-9-4), which claim to be extremely low-bit methods but actually have an equivalent bit-width larger than 2-bit, as baselines. Additionally, several state-of-the-art PTQ methods (2-bit) such as OmniQuant [\(Shao et al.,](#page-10-10) [2023\)](#page-10-10), AWQ [\(Lin](#page-9-9) [et al.,](#page-9-9) [2023\)](#page-9-9), QuIP [\(Chee et al.,](#page-9-15) [2024\)](#page-9-15), and GPTQ [\(Frantar et al.,](#page-9-8) [2022\)](#page-9-8) are also be evaluated.

Models We evaluate our method mainly on LLaMA [\(Touvron et al.,](#page-10-0) [2023a\)](#page-10-0), LLaMA-2 [\(Tou](#page-10-1)[vron et al.,](#page-10-1) [2023b\)](#page-10-1) and LLaMA-3 [\(Dubey et al.,](#page-9-16) [2024\)](#page-9-16), for LLaMA-families are currently the most popular and widely applied among LLMs. Considering the comprehensiveness, experiments on OPT families [\(Zhang et al.,](#page-11-11) [2022\)](#page-11-11) are in Appendix [D.](#page-14-0)

Training Details We initialize learnable scaling factors with α<sup>w</sup> = ∥w∥<sup>1</sup> nw and AdamW optimizer [\(Loshchilov and Hutter,](#page-10-13) [2017\)](#page-10-13) with zero weight decay is utilized to update them with learning rate 5e-4 and 1e-3. For PTQ, our calibration set sampled from WikiText2 [\(Merity et al.,](#page-10-14) [2016\)](#page-10-14) consists

<span id="page-6-0"></span>

| LLaMA | Methods   | Bits    | PIQA  | ARC-e | HellaS | Wing  | Race  | ARC-c | LAMB-o | LAMB-s | Avg.  |
|-------|-----------|---------|-------|-------|--------|-------|-------|-------|--------|--------|-------|
|       | FP        | 16      | 78.67 | 75.29 | 56.99  | 70.01 | 40.29 | 41.81 | 73.57  | 67.82  | 63.06 |
|       | GPTQ      | 2       | 53.64 | 26.09 | 25.87  | 47.75 | 22.68 | 22.44 | 0.0    | 0.0    | 24.81 |
|       | QuIP      | 2       | 60.23 | 38.26 | 34.83  | 51.22 | 28.90 | 22.10 | 15.10  | 8.52   | 32.40 |
| 1-8   | OmniQuant | 2       | 58.22 | 39.94 | 32.45  | 52.49 | 32.25 | 22.10 | 23.66  | 12.77  | 34.24 |
|       | PB-LLM    | 1.7(+1) | 55.71 | 29.12 | 28.31  | 48.86 | 26.41 | 19.80 | 10.42  | 10.09  | 28.59 |
|       | BiLLM     | 1(+1.1) | 61.10 | 40.99 | 31.80  | 53.67 | 30.14 | 20.64 | 23.15  | 16.48  | 36.00 |
|       | PTQ1.61   | 1.61    | 63.71 | 49.62 | 35.73  | 56.75 | 32.54 | 25.26 | 38.93  | 26.57  | 41.14 |
|       | FP        | 16      | 79.16 | 77.36 | 59.92  | 72.77 | 39.62 | 46.42 | 76.15  | 71.08  | 65.31 |
|       | GPTQ      | 2       | 51.90 | 25.84 | 26.08  | 49.72 | 24.11 | 22.18 | 0.0    | 0.0    | 24.98 |
| 1-13  | OmniQuant | 2       | 67.14 | 51.43 | 41.28  | 56.20 | 32.73 | 29.52 | 23.40  | 17.85  | 39.94 |
|       | PB-LLM    | 1.7(+1) | 60.45 | 37.46 | 30.79  | 51.07 | 30.24 | 18.69 | 27.71  | 21.33  | 34.72 |
|       | BiLLM     | 1(+1.1) | 67.90 | 50.84 | 39.02  | 62.19 | 34.07 | 26.11 | 49.93  | 33.75  | 44.98 |
|       | PTQ1.61   | 1.61    | 68.17 | 58.59 | 40.02  | 58.33 | 34.26 | 27.22 | 45.95  | 35.94  | 46.56 |
|       | FP        | 16      | 80.96 | 80.39 | 63.34  | 75.69 | 40.57 | 52.82 | 77.59  | 73.34  | 68.09 |
|       | GPTQ      | 2       | 52.50 | 20.39 | 25.88  | 51.38 | 23.25 | 21.33 | 0.06   | 0.0    | 24.35 |
| 1-30  | QuIP      | 2       | 72.42 | 58.80 | 45.64  | 63.30 | 35.69 | 30.20 | 52.45  | 36.64  | 49.39 |
|       | OmniQuant | 2       | 70.35 | 58.03 | 44.82  | 58.17 | 34.93 | 31.74 | 41.88  | 31.75  | 46.46 |
|       | PB-LLM    | 1.7(+1) | 63.76 | 40.11 | 33.32  | 61.17 | 30.91 | 21.33 | 43.78  | 33.09  | 40.93 |
|       | PTQ1.61   | 1.61    | 70.24 | 63.64 | 46.82  | 63.61 | 37.13 | 32.17 | 55.95  | 44.61  | 51.77 |
|       | FP        | 16      | 78.07 | 76.30 | 57.14  | 69.06 | 39.52 | 43.34 | 73.86  | 68.23  | 63.19 |
|       | QuIP      | 2       | 56.53 | 28.70 | 27.52  | 48.78 | 24.40 | 18.94 | 3.18   | 2.06   | 26.26 |
| 2-7   | OmniQuant | 2       | 57.34 | 38.80 | 30.11  | 51.78 | 27.37 | 20.73 | 3.98   | 1.47   | 30.20 |
|       | PB-LLM    | 1.7(+1) | 54.46 | 28.20 | 27.03  | 49.09 | 26.70 | 19.11 | 7.08   | 5.71   | 27.17 |
|       | BiLLM     | 1(+1.1) | 60.39 | 39.94 | 30.74  | 51.93 | 29.57 | 21.16 | 18.44  | 14.05  | 33.28 |
|       | PTQ1.61   | 1.61    | 63.22 | 47.18 | 35.78  | 52.25 | 29.86 | 22.27 | 37.38  | 25.65  | 39.20 |
|       | FP        | 16      | 79.11 | 79.46 | 60.04  | 72.14 | 40.57 | 48.46 | 76.77  | 70.33  | 65.86 |
|       | QuIP      | 2       | 65.45 | 51.56 | 39.65  | 55.72 | 31.58 | 25.85 | 33.86  | 22.67  | 40.79 |
| 2-13  | OmniQuant | 2       | 62.62 | 44.27 | 40.16  | 52.17 | 30.81 | 24.66 | 20.07  | 10.17  | 35.62 |
|       | PB-LLM    | 1.7(+1) | 54.46 | 27.95 | 26.74  | 49.96 | 26.03 | 19.54 | 3.14   | 2.50   | 26.29 |
|       | BiLLM     | 1(+1.1) | 63.55 | 49.83 | 34.36  | 58.17 | 32.34 | 23.81 | 40.81  | 25.15  | 41.00 |
|       | PTQ1.61   | 1.61    | 66.54 | 56.86 | 40.32  | 55.88 | 33.30 | 26.45 | 47.23  | 31.21  | 44.72 |
|       | FP        | 16      | 79.54 | 80.09 | 60.14  | 73.24 | 40.29 | 50.17 | 75.65  | 68.72  | 65.98 |
|       | GPTQ      | 2       | 52.39 | 26.14 | 25.74  | 51.54 | 20.19 | 20.65 | 0.0    | 0.0    | 24.58 |
| 3-8   | QuIP      | 2       | 52.72 | 26.43 | 29.32  | 50.67 | 27.75 | 20.39 | 4.93   | 3.01   | 26.90 |
|       | OmniQuant | 2       | 54.13 | 28.87 | 26.50  | 50.12 | 22.68 | 20.48 | 0.02   | 0.02   | 25.35 |
|       | PB-LLM    | 1.7(+1) | 56.91 | 32.37 | 28.43  | 49.25 | 27.66 | 17.41 | 16.63  | 12.44  | 30.14 |
|       | BiLLM     | 1(+1.1) | 60.01 | 38.26 | 31.48  | 53.75 | 30.14 | 19.45 | 26.30  | 15.51  | 34.36 |
|       | PTQ1.61   | 1.61    | 63.22 | 46.17 | 34.71  | 52.80 | 29.09 | 23.04 | 30.27  | 18.26  | 37.20 |

Table 2: Reasoning accuracies comparison on LLaMA family. More tasks are listed in Table [11](#page-18-0) and [12.](#page-18-1)

of 128 random 2048 token-segments and the blockwise training process includes 20 epochs with a batch size of 1. For quantization preprocessing, the number of steps and ranks in lightweight restorative LoRA is 20K and 64 respectively. The entire process is deployed on 2 Nvidia A800 GPUs.

Datasets On language generation tasks which are the core objectives of LLMs, our test set comes from WikiText2 and C4 [\(Raffel et al.,](#page-10-15) [2020\)](#page-10-15). We also select several reasoning benchmarks, *i.e.*, PIQA [\(Bisk et al.,](#page-8-2) [2020\)](#page-8-2), ARC [\(Clark et al.,](#page-9-17) [2018\)](#page-9-17), HellaSwag [\(Zellers et al.,](#page-11-12) [2019\)](#page-11-12), Winogrande [\(Sak](#page-10-16)[aguchi et al.,](#page-10-16) [2021\)](#page-10-16), Race [\(Lai et al.,](#page-9-18) [2017\)](#page-9-18) and LAMBADA [\(Paperno et al.,](#page-10-17) [2016\)](#page-10-17), using the open-

sourced toolkit lm-evaluation-harness [\(Gao et al.,](#page-9-19) [2023\)](#page-9-19). We also assess the evaluation on MMLU [\(Hendrycks et al.,](#page-9-20) [2021\)](#page-9-20), GSM8K [\(Cobbe et al.,](#page-9-21) [2021\)](#page-9-21) and LongBench [\(Bai et al.,](#page-8-3) [2024\)](#page-8-3), please refer to Appendix [E.](#page-17-1) In addition, RedPajama [\(Com](#page-9-14)[puter,](#page-9-14) [2023\)](#page-9-14) is used for quantization preprocessing.

## 4.2 Experiments on Language Generation Tasks

The fundamental prowess of LLMs lies in their language generation capabilities. Consequently, evaluating such capabilities of a quantized model via perplexity serves as the core metric of a quantization method. As presented in Table [1,](#page-5-0) we compare

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Figure 5: Our novel quantization preprocessing scheme on other existing PTQ methods. Results on more LLMs and common sense reasoning tasks can be found in Figure 8.

the perplexities between our **PTQ1.61** and other baselines to valid the effectiveness on extremely low-bit PTQ tasks, from where we can see that our method achieves promising performance. In comparison to the two extremely low-bit methods PB-LLM and BiLLM, our performance significantly surpasses theirs while not introducing intolerable bit-width for each weight. In terms of OmniQuant, which performs best on LLaMA families among baselines, our method still surpasses it by a significant margin. For instance, on LLaMA-2-7B we achieves 12.70 while OmniQuant is 37.37 in Wiki-Text2. Significantly, our method performs better on LLaMA3, which is known to be harder to quantize.

#### 4.3 Experiments on Reasoning Benchmarks

Reasoning capability is becoming a crucial metric for evaluating PTQ approaches. The comparison results are indicated in Table 2, where our method exhibits superiority in most benchmarks. For instance, compared with the second best baseline, BiLLM, our method showcases an average performance increase of  $1.58\% \sim 5.92\%$ . Particularly, on LAMBADA most baselines suffer from significant performance degradation while we still maintain an outstanding level. Considering the comparison results and the minimal 1.61-bit, we conclude that our method represents the state-of-the-art extremely low-bit PTQ scheme for LLMs.

#### 4.4 Ablation Study

After demonstrating the advancement of our **PTQ1.61**, we conduct ablation study on LLaMA-13B to further validate the effectiveness of our each innovation as indicated in Table 3. As the first

<span id="page-7-1"></span>

| Structured<br>Mask | Learnable<br>Scalar | Preprocess   | WikiText2 | C4     |
|--------------------|---------------------|--------------|-----------|--------|
| -                  | -                   | -            | 14664     | 11377  |
| $\checkmark$       | -                   | -            | 1370.4    | 772.83 |
| -                  | -                   | $\checkmark$ | 569.81    | 702.44 |
| $\checkmark$       | $\checkmark$        | -            | 14.22     | 20.78  |
| $\checkmark$       | $\checkmark$        | $\checkmark$ | 9.67      | 13.51  |

Table 3: Ablation study (PPL) on LLaMA-13B.

row, without any additional improvements, directly using the derived analytically scaling factors for binarization would almost entirely compromise the text generation capability of LLMs. When utilizing our structured masks to retain salient weights as the second row, there is a significant improvement, indicating the importance of salient weights and the effectiveness of our masks, but there remains considerable room for enhancement. Furthermore, our novel block-wise strategy for non-salient weights binarization lifts the performance to a excellent level as demonstrated by the forth row. Ultimately, the last row illustrates that the row-wise pattern obtained by our quantization preprocessing brings a remarkable enhancement. More detailed ablation results are available in Appendix B.

#### **4.5** Quantization Preprocessing on Baselines

In addition to our **PTQ1.61**, we also employ the proposed quantization preprocessing scheme on other baselines to validate its scalability and the results can be found in Figure 5, which demonstrate that significant improvements appear in all baselines, especially for 2-bit GPTQ which completely collapses without preprocessing. With the effectiveness of our preprocessing scheme, future

research can take a fresh perspective to focus on finding a more appropriately pretrained model.

At present, despite its strong performance, our preprocessing scheme still suffers from drawbacks such as longer runtime (as indicated in Table [8\)](#page-14-3). Therefore, we must clarify that our preprocessing scheme is viewed as an optional component of our PTQ*1.61*. As indicated in Table [6,](#page-14-1) even without preprocessing, our PTQ*1.61* still achieves state-ofthe-art performance under extremely low-bit setting.

## 5 Conclusion

In this paper, we explore the real limit of posttraining quantization and propose an extremely low-bit PTQ approach namely PTQ*1.61*, which is truly the first PTQ method enables sub 2-bit quantization for LLMs. Firstly, one-dimensional structured mask with negligibly additional 0.0002-bit per weight is introduced to preserve salient weights. For non-salient weights binarization, we devise an efficient block-wise optimization strategy to learn scaling factors considering row correlations and angular biases. In addition to above contributions, we further propose a quantization preprocessing paradigm to transform the salient weights into a row-wise pattern which is able to alleviate the difficulty in per-channel quantization. Extensive experiments indicate that PTQ*1.61* becomes state-ofthe-art extremely low-bit PTQ method for LLMs.

