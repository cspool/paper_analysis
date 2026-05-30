# **G.1 Detailed Runtime Profiling**

We evaluate the computational efficiency of **DIG** compared to distinct baselines, AKS [\[18\]](#page-11-9) and Q-Frame [\[64\]](#page-14-10). The total runtime of each method can be divided into two stages:

- *Key Frame Selection*, where the method identifies optimal indices from raw video.
- *Inference*, where the LMM processes the selected frames to generate a response.

All experiments were conducted on a node equipped with 8 NVIDIA A100 GPUs. To provide a comprehensive analysis, we report the standard LMM inference latency across varying input frame counts in Table [9](#page-24-1) and detail the selection overhead introduced by specific methods in Table [10.](#page-24-1)

**DIG achieves a favorable efficiency-performance trade-off.** As evidenced in Table [10,](#page-24-1) **DIG** offers a significant efficiency advantage over AKS [\[18\]](#page-11-9), reducing computational overhead by an order of magnitude while maintaining superior downstream performance (see Section [5\)](#page-5-1). While **DIG** incurs a marginal increase in processing time compared to Q-Frame [\[64\]](#page-14-10), this cost is justified by substantial robustness gains; specifically, Q-Frame [\[64\]](#page-14-10) fails to outperform uniform sampling as frame counts exceed 32, whereas **DIG** consistently surpasses baselines across all settings. Furthermore, comparing the selection overhead (Table [10\)](#page-24-1) against standard inference latency (Table [9\)](#page-24-1), the additional cost remains within a reasonable range. This confirms that **DIG** effectively balances efficiency and accuracy, serving as a practical, plug-and-play module for enhanced long-form video understanding.

<span id="page-24-1"></span>**Table 9:** *Inference latency analysis. The inference time (in minutes) of the base LMM (Qwen2.5-VL-7B [\[16\]](#page-11-2)) across different input frame counts using standard uniform sampling.*

| Dataset             | # Frames |     |     |      |      |      |      |
|---------------------|----------|-----|-----|------|------|------|------|
|                     | 8        | 16  | 32  | 64   | 128  | 192  | 256  |
| MLVU [54]           | 3.2      | 5.0 | 9.3 | 17.6 | 29.1 | 37.3 | 43.4 |
| LongVideoBench [55] | 1.4      | 2.2 | 4.3 | 8.3  | 14.0 | 19.9 | 25.6 |
| VideoMME [56]       | 3.1      | 4.7 | 8.7 | 15.8 | 26.1 | 36.7 | 46.3 |

**Table 10:** *Comparison of frame selection overhead. The time cost (in minutes) required by different methods to process videos and select key frames. For DIG, we break down the cost into Query Identification (QI), Content-Aware Frame Selection (CAFS), Reward Assignment (RA), and Video Refinement (VR).*

| Dataset             | AKS [18] | Q-Frame [64] | DIG (Ours) |      |       |     |       |
|---------------------|----------|--------------|------------|------|-------|-----|-------|
|                     |          |              | QI         | CAFS | RA    | VR  | Sum   |
| MLVU [54]           | ≥ 720    | 122.1        | 11.3       | 25.9 | 218.9 | 0.2 | 256.3 |
| LongVideoBench [55] | ≥ 720    | 34.5         | 7.6        | 20.8 | 110.4 | 0.1 | 138.9 |
| VideoMME [56]       | ≥ 720    | 94.2         | 11.6       | 31.2 | 264.8 | 0.3 | 307.9 |

