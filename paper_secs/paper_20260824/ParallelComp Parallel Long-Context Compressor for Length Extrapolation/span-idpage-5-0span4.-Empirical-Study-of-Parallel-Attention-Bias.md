# <span id="page-5-0"></span>4. Empirical Study of Parallel Attention Bias

In this section, we investigate the attention sink phenomenon in parallel attention and compare its similarities and differences with the regular attention sink phenomenon. Specifically, we explore the following question:

**Q1:** What types of attention patterns can be summarized? In summary, three main types of attention patterns emerge, as illustrated in Figure 3: U-shape, Mountain-shape, and Uniform-shape.

**Observations.** These attention distributions give rise to three corresponding biases: i) Attention sink, where focus is concentrated on the initial few tokens. ii) Recency bias, where attention is more strongly concentrated at the tail. iii) Middle bias, where attention is disproportionately focused on a few tokens in the middle of a sequence. iv) These biases manifest in a wavelike pattern, with  $R_H$  containing three token types  $(R_s, R_m, R_r)$  corresponding to these biases.

**Q2:** Is there any difference between the attention bias in parallel attention and the attention bias in classical attention? In this part, we provide a detailed analysis of bias in parallel attention. We observe in Figure 4 that there are relatively more peaks within the contexts compared to the classic attention mechanism.

<span id="page-5-2"></span>> **[图片提取文字 (无描述)]:**
> Threshold 0.1 Layer ID Threshold 0.3 32 **Before Calibration** After Calibration 32 Head ID
![](_page_5_Figure_9.jpeg)

Figure 6: The distribution of tokens with abnormally high attention scores. Blue represents outliers.

**Observations.** *i)* Similar to the blue local attention, the yellow curve shows the U-shaped attention sink repeatedly appearing in global attention. *ii)* Parallel attention and local attention both exhibit severe recency bias, but the bias is significantly mitigated in parallel attention compared to local attention. *iii)* When computing global attention  $A_{\mathfrak{g}}$ , the model suffers from a more severe recency bias compared to the attention sink, though it is still less pronounced than within  $A_{\mathfrak{l}}^c$  (blue line). *iv)* Compared to the classical attention distribution, i.e., the local attention, the peaks of  $A_{\mathfrak{g}}$  within the chunk are significantly weakened, indicating that global attention can significantly mitigate recency bias. *In other words, the parallel attention itself can mitigate attention bias.* 

<span id="page-6-0"></span>

|                              | Single | e-Docume | nt QA | Mul     | ti-Docume     | nt QA         | Su       | mmariza    | tion       | Few   | -shot Lea | rning  | Synt   | hetic | Co    | ode   |       |
|------------------------------|--------|----------|-------|---------|---------------|---------------|----------|------------|------------|-------|-----------|--------|--------|-------|-------|-------|-------|
| Methods                      | NITYOA | Qasper   | MFen  | HotpotQ | A<br>2WikiMQ! | Musique       | GovRepor | QMSum      | MultiNews  | TREC  | TriviaQA  | SAMSum | PCount | pRe   | Lec   | RB-P  | AVG   |
| Max Length                   | 84123  | 24204    | 17727 | 20325   | 19001         | 20520         | 60515    | 34477      | 16271      | 13049 | 26756     | 21884  | 32699  | 17158 | 37628 | 58822 | 30657 |
|                              |        |          |       |         |               | Lla           | ama2-7B- | chat-hf(4l | <b>(</b> ) |       |           |        |        |       |       |       |       |
| FullKV(4k)                   | 18.62  | 19.53    | 35.49 | 31.07   | 26.15         | 9.91          | 25.52    | 20.87      | 26.28      | 62.00 | 82.68     | 40.86  | 5.50   | 10.50 | 61.04 | 55.30 | 33.21 |
| Dynamic-PI                   | 9.69   | 20.05    | 33.10 | 16.40   | 23.83         | 3.62          | 27.83    | 18.75      | 16.53      | 62.00 | 67.00     | 40.37  | 1.58   | 5.14  | 55.30 | 55.49 | 28.54 |
| NTK-Aware                    | 13.02  | 14.25    | 31.51 | 29.55   | 30.64         | 11.83         | 28.78    | 16.96      | 26.30      | 62.50 | 74.88     | 39.35  | 4.08   | 4.50  | 49.74 | 49.39 | 30.46 |
| ChunkLlama                   | 22.97  | 20.52    | 33.71 | 28.91   | 26.14         | 13.84         | 14.84    | 21.62      | 18.13      | 62.50 | 77.15     | 40.83  | 2.03   | 4.00  | 59.81 | 54.33 | 31.33 |
| InfLLM                       | 18.14  | 22.11    | 29.86 | 30.99   | 30.74         | 9.41          | 26.33    | 20.63      | 26.18      | 62.50 | 84.24     | 39.92  | 3.36   | 6.00  | 60.15 | 55.99 | 32.91 |
| AttenCalibration-NTK         | 14.05  | 12.49    | 32.52 | 30.61   | 31.22         | 12.84         | 29.72    | 18.24      | 24.40      | 61.50 | 72.88     | 39.54  | 2.33   | 3.00  | 48.86 | 50.36 | 30.29 |
| Ours                         | 23.20  | 17.50    | 37.07 | 38.67   | 32.68         | 20.22         | 25.00    | 22.79      | 25.84      | 64.00 | 84.63     | 40.67  | 4.00   | 31.50 | 59.37 | 58.53 | 36.60 |
| Ours-calibration             | 24.95  | 19.07    | 38.16 | 39.53   | 32.62         | 22.64         | 25.42    | 22.82      | 26.01      | 63.00 | 85.41     | 40.36  | 5.00   | 32.50 | 59.04 | 58.84 | 37.21 |
| Ours-compression             | 23.32  | 16.97    | 35.25 | 39.49   | 32.47         | 20.17         | 24.33    | 21.97      | 25.68      | 63.50 | 84.46     | 40.81  | 4.00   | 31.50 | 59.43 | 58.54 | 36.37 |
| Ours-calibration-compression | 24.04  | 18.39    | 38.03 | 39.89   | 35.38         | 22.15         | 24.26    | 22.46      | 24.51      | 63.50 | 84.83     | 40.73  | 4.00   | 31.50 | 57.67 | 58.48 | 36.86 |
|                              |        |          |       |         |               | Lla           | ma3-8B-i | nstruct(81 | <b>(</b> ) |       |           |        |        |       |       |       |       |
| FullKV(8k)                   | 24.31  | 38.13    | 39.69 | 44.16   | 35.66         | 21.00         | 28.35    | 23.06      | 26.96      | 73.00 | 90.13     | 42.46  | 4.61   | 68.50 | 60.46 | 56.11 | 42.29 |
| Dynamic-PI                   | 21.71  | 36.66    | 38.24 | 33.70   | 35.48         | 14.28         | 29.41    | 22.04      | 25.55      | 74.50 | 82.61     | 42.62  | 2.33   | 85.59 | 58.22 | 47.16 | 40.63 |
| NTK-Aware                    | 25.92  | 37.54    | 42.23 | 48.32   | 36.96         | 27.51         | 33.74    | 24.13      | 26.35      | 50.50 | 88.84     | 42.53  | 7.24   | 95.61 | 34.84 | 39.04 | 41.33 |
| ChunkLlama                   | 25.01  | 37.39    | 43.52 | 49.37   | 37.56         | 3 <b>0.95</b> | 17.57    | 23.51      | 19.72      | 76.00 | 90.38     | 42.14  | 4.71   | 67.95 | 61.10 | 52.57 | 42.47 |
| InfLLM                       | 19.93  | 43.52    | 40.58 | 48.31   | 35.99         | 23.25         | 30.49    | 21.60      | 26.53      | 74.00 | 90.93     | 42.30  | 8.00   | 74.00 | 58.98 | 52.46 | 43.18 |
| AttenCalibration-NTK         | 26.54  | 37.52    | 41.13 | 47.56   | 38.98         | 26.51         | 34.21    | 23.35      | 25.64      | 45.50 | 89.23     | 42.21  | 4.81   | 93.51 | 36.86 | 42.82 | 41.02 |
| Ours                         | 26.67  | 39.05    | 42.66 | 49.58   | 40.02         | 26.23         | 29.10    | 24.18      | 26.74      | 69.00 | 91.03     | 42.07  | 7.81   | 92.38 | 58.84 | 53.54 | 44.93 |
| Ours-calibration             | 26.89  | 39.46    | 42.01 | 49.88   | 41.41         | 26.68         | 29.17    | 24.55      | 26.77      | 72.50 | 90.53     | 42.13  | 8.02   | 92.75 | 58.06 | 53.97 | 45.21 |
| Ours-compression             | 26.18  | 36.56    | 39.72 | 47.10   | 34.89         | 24.96         | 27.03    | 23.86      | 24.52      | 67.00 | 89.55     | 41.20  | 7.37   | 92.29 | 58.51 | 52.15 | 43.31 |
| Ours-calibration-compression | 26.46  | 37.49    | 41.28 | 48.28   | 36.29         | 26.68         | 26.79    | 24.98      | 25.18      | 69.00 | 90.37     | 40.72  | 7.34   | 91.29 | 57.30 | 53.97 | 44.31 |

Table 1: Length Extrapolation Performance Comparison across Different Tasks. Ours-calibration and Ours-compression both represent parallel KV Cache Eviction, where the former evicts tokens of  $R_h$ , and the latter evicts tokens of  $R_l$ . Ours-calibration-compression represents the simultaneous adoption of both eviction strategies. FullKV refers to truncating the context to 4k or 8k lengths (without extrapolation) for generation.

