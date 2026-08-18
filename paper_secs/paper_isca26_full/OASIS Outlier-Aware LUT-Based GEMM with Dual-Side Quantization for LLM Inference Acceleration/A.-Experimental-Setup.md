# *A. Experimental Setup*

value in the min tree.

Models and Tasks We examine the algorithm performance of OASIS on a spectrum of LLMs and tasks. Specifically, the models include OPT-6.7B/13B/30B [61], LLaMA-7B/13B/30B [52], LLaMA-2-7B/13B/70B [53], LLaMA-3- 8B [15], and Mistral-7B [20], which are implemented with Transformers [56] and PyTorch [44]. These LLMs are evaluated on two tasks: (i) the next-word prediction task using the WikiText-2 [36] dataset, measured by the perplexity (PPL) metric, and (ii) the zero-shot accuracy task across six common sense datasets: PIQA [4], ARC-easy (ARC-E) [7], ARCchallenge (ARC-C) [7], BoolQ [6], HellaSwag [59], and Wino-Grande [47]. The zero-shot performance evaluation utilizes the Language Model Evaluation Harness [13] framework. All algorithm performance experiments are conducted on an NVIDIA A100-80GB GPU.

OASIS' NU-WAQ Implementation Details In OASIS, both weights and activations are quantized using K-Means clustering. Weights undergo 4-bit per-output-channel quantization without outlier protection, while activations are quantized per-token with 3/4-bit precision. We first perform post-training K-Means quantization on the LLM weights to obtain the weight centroids and indices. Next, the activation centroids are trained offline using 16 calibration samples from the C4 dataset [9], and the indices are computed online. We incorporate a weighted-K-Means algorithm to obtain the activation centroids, where the weights are determined by Fisher information matrices [45] of the activations. To handle outliers, the top 0.5% largest and bottom 0.5% smallest activation values are preserved in FP16 format, while the inliers are quantized. During inference, OASIS dynamically identifies outliers using the *Orizuru* units, while OASIS-S reuses the thresholds from the offline training process on the calibration dataset.

Baseline LLM Quantization Methods We compare OASIS with INT-WAQ baselines round-to-nearest (RTN), SmoothQuant [58], QuaRot [3], and Atom [62]. Except for Atom, which uses group-wise quantization for both weights and activations with a group size of 128, all other baseline algorithms employ per-output-channel quantization for weights and per-token quantization for activations.

Architecture Modeling and Comparison The hardware performance of the OASIS architecture is modeled using a cycle-accurate simulator modified from DnnWeaver [48]. The area and power metrics of the core logic units in the OASIS accelerator, such as the Concat Unit, MAC Tree, Index Counter, and *Orizuru*, are derived from synthesis results using the TSMC 28 nm standard cell library. Table II shows the detailed configurations of the hardware components on-chip.

We use Cacti [27] and DRAMSim3 [26] to simulate the overhead of on-chip SRAM and off-chip HBM, respectively. We denote W4A3 OASIS as OASIS-A3 and W4A4 OASIS as OASIS-A4, and similarly for OASIS-S. We compare the hardware performance of OASIS with a series of baseline hardware accelerators, including the GPU-based platforms of NVIDIA A100-80GB GPU [40] and QuaRot [3] and an ASIC accelerator of FIGLUT [42]. Unless otherwise specified, the hardware performance of OASIS and the baseline accelerators are evaluated on the next-word-prediction task with an output sequence length of 2048.

#### *B. Algorithm Performance Analysis*

Table III shows the WikiText-2 PPL results for OASIS and baseline INT-WAQ methods across various models with a sequence length of 2048. OASIS consistently achieves the lowest PPL for both W4A4 and W4A3 precisions, outperforming the INT-WAQ methods. This demonstrates the effectiveness of OASIS's NU-WAQ and outlier protection methods in reducing quantization errors and enhancing model performance. For

| Precision | Method            | OPT   |       | LLaMA |       |       |       | LLaMA-2 |       | II aMA 2 PD | M:-41 7D   |            |
|-----------|-------------------|-------|-------|-------|-------|-------|-------|---------|-------|-------------|------------|------------|
|           |                   | 6.7B  | 13B   | 30B   | 7B    | 13B   | 30B   | 7B      | 13B   | 70B         | LLaMA-3-8B | Mistral-7B |
| FP16      | -                 | 10.86 | 10.12 | 9.56  | 5.68  | 5.09  | 4.10  | 5.47    | 4.88  | 3.32        | 6.14       | 5.25       |
|           | RTN               | 6e3   | 3e4   | 7e3   | 8e3   | 1e4   | 3e5   | 2e3     | 7e3   | 2e5         | 2e3        | 6e3        |
|           | SmoothQuant       | 2e4   | 7e3   | 1e4   | 4e2   | 67.20 | 32.51 | 7e2     | 56.61 | 10.54       | 1e3        | 5e2        |
| W4A4      | QuaRot            | 12.21 | 11.20 | 10.92 | 6.34  | 5.58  | 4.64  | 6.19    | 5.45  | 3.83        | 8.16       | 5.77       |
| W4A4      | Atom <sup>†</sup> | 12.05 | 10.99 | 10.74 | 6.25  | 5.52  | 4.61  | 6.12    | 5.31  | 3.73        | 8.10       | 5.76       |
|           | OASIS-S           | 11.77 | 10.93 | 10.31 | 6.08  | 5.38  | 4.40  | 6.00    | 5.21  | 3.60        | 7.02       | 5.84       |
|           | OASIS             | 11.62 | 10.75 | 10.21 | 6.04  | 5.37  | 4.38  | 5.90    | 5.19  | 3.55        | 7.11       | 5.75       |
| W4A3      | RTN               | 3e4   | 2e4   | 2e4   | 2e4   | 2e4   | 1e4   | 6e5     | 5e5   | 6e5         | 1e5        | 1e4        |
|           | SmoothQuant       | 7e4   | 7e4   | 6e4   | 5e4   | 2e4   | 2e4   | 8e3     | 1e4   | 1e4         | 8e3        | 9e3        |
|           | QuaRot            | 2e2   | 2e2   | 1e2   | 29.75 | 19.02 | 13.50 | 2e2     | 2e2   | 85.28       | 3e2        | 2e2        |
|           | Atom <sup>†</sup> | 20.51 | 15.61 | 14.48 | 9.62  | 7.36  | 6.18  | 11.40   | 8.00  | 5.05        | 13.11      | 10.83      |
|           | OASIS-S           | 15.12 | 13.49 | 12.14 | 7.60  | 6.28  | 5.31  | 7.91    | 6.99  | 4.13        | 8.96       | 7.42       |
|           | OASIS             | 14.12 | 12.84 | 11.78 | 7.17  | 6.21  | 5.10  | 7.49    | 6.43  | 4.05        | 8.18       | 7.27       |

† Atom applies group quantization to weights and activations, with the group size of 128.

LLaMA-2-7B at W4A4, OASIS achieves a PPL of 5.90, with only a 0.43 degradation from the FP16 model, which is 34% lower than Atom's degradation. Additionally, OASIS reduces PPL by 0.05 at W4A4 and 0.27 at W4A3 compared to OASIS-S, highlighting the benefits of dynamic outlier detection. For the LLaMA-3-8B, which is known to be more quantization-sensitive, OASIS achieves a PPL of 7.11 at W4A4, which reduces the PPL degradation by 49% compared to Atom. We notice that for LLaMA-2-7B and 13B, W4A3 quantization yields higher PPL than their counterparts in the LLaMA-7B and 13B models, because the more extensively trained LLaMA-2 models are harder to post-training quantize at low precisions [22].

On average, OASIS introduces only a 2.05% and 5.90% accuracy drop at W4A4 and W4A3 precision levels, respectively, compared to the FP16 baseline, while significantly outperforming state-of-the-art INT-WAQ methods. In the W4A4 setting, OASIS improves accuracy by 6.44% and 6.92% compared to Atom and QuaRot, respectively. Under the W4A3 configuration, OASIS achieves accuracy improvements of 8.79% over Atom and 30.44% over QuaRot.

