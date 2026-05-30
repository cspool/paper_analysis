# <span id="page-17-0"></span>D Experiments on more metrics

Following prior work [\[62\]](#page-13-5), we evaluate different methods on EvalCrafter [\[34\]](#page-12-13) benchmark for multiaspects metrics evaluation. We select CLIPSIM, CLIP-Temp, DOVER [\[53\]](#page-13-14) video quality assessment (VQA) metrics to evaluate the generation quality, and Flow-score to evaluate the temporal consistency. We conduct experiments on CogVideoX-2B, CogVideoX-5B, and HunyuanVideo-13B under W4A6 quantization setting. We present the evaluation results in Tab. [7.](#page-17-2)

<span id="page-17-2"></span>Table 7: Performance of 4-bit weight and 6-bit activation quantization on text-to-video generation under EvalCrafter benchmark. Higher (↑) metrics represent better performance.

|              |              |         |           | VQA       | VQA       | FLOW   |
|--------------|--------------|---------|-----------|-----------|-----------|--------|
| Model        | Method       | CLIPSIM | CLIP-Temp | Aesthetic | Technical | Score. |
|              | FP           | 0.1844  | 0.9978    | 76.64     | 85.02     | 3.452  |
|              | Q-DiT        | 0.1787  | 0.9978    | 63.15     | 67.37     | 2.331  |
|              | PTQ4DiT      | 0.1772  | 0.9985    | 58.76     | 52.60     | 1.837  |
| CogVideoX-2B | SmoothQuant  | 0.1762  | 0.9981    | 55.18     | 53.87     | 1.378  |
|              | Quarot       | 0.1808  | 0.9975    | 51.83     | 56.79     | 2.867  |
|              | ViDiT-Q      | 0.1812  | 0.9976    | 53.09     | 59.84     | 3.040  |
|              | 2Q-VDiT<br>S | 0.1838  | 0.9979    | 70.50     | 73.31     | 3.122  |
|              | FP           | 0.1814  | 0.9982    | 78.87     | 73.17     | 4.536  |
|              | Q-DiT        | 0.1835  | 0.9976    | 47.96     | 46.72     | 2.967  |
|              | PTQ4DiT      | 0.1789  | 0.9984    | 22.93     | 44.07     | 2.230  |
| CogVideoX-5B | SmoothQuant  | 0.1742  | 0.9976    | 3.05      | 14.13     | 1.026  |
|              | Quarot       | 0.1805  | 0.9983    | 33.10     | 43.67     | 3.040  |
|              | ViDiT-Q      | 0.1795  | 0.9980    | 42.01     | 48.59     | 1.850  |
|              | 2Q-VDiT<br>S | 0.1819  | 0.9987    | 73.45     | 74.41     | 3.688  |
|              | FP           | 0.1910  | 0.9985    | 80.66     | 63.51     | 1.674  |
|              | Q-DiT        | 0.1871  | 0.9987    | 56.45     | 43.17     | 1.482  |
|              | PTQ4DiT      | 0.1786  | 0.9973    | 42.17     | 33.69     | 1.089  |
| HunyuanVideo | SmoothQuant  | 0.1782  | 0.9978    | 7.24      | 0.42      | 0.111  |
|              | Quarot       | 0.1873  | 0.9977    | 66.49     | 52.81     | 0.899  |
|              | ViDiT-Q      | 0.1895  | 0.9978    | 66.23     | 51.35     | 0.897  |
|              | 2Q-VDiT<br>S | 0.1902  | 0.9985    | 77.80     | 66.38     | 1.562  |

It can be seen that under the EvalCrafter [\[34\]](#page-12-13) benchmark, our S <sup>2</sup>Q-VDiT still achieved almost lossless performance and showed significant performance improvement compared to all comparison methods. Especially in terms of VQA-Technical metrics, our S <sup>2</sup>Q-VDiT even outperforms the full precision model on CogVideoX-5B and HunyuanVideo, while other methods show notable performance degradation. For CogVideoX-5B, S <sup>2</sup>Q-VDiT achieves 74.41 in VQA-Technical which outperforms the full precision model of 73.17, while current methods achieve the best of 48.59.

### <span id="page-17-1"></span>E Integration with Existing PTQ Methods

The techniques that we proposed Hessian-aware Salient Data Selection (SDS) and Attention-guided Sparse Token Distillation (STD) can also be applied to existing block-wise optimization-based post-training quantization methods. To verify the generality of these two techniques, we combined them with the existing baseline method PTQ4DiT [\[54\]](#page-13-4) and reported the performance improvement of these techniques on W4A6 CogVideoX-2B under VBench [\[19\]](#page-11-14) benchmark in Tab. [8.](#page-18-1) By using the calibration constructed by SDS, we further improved the performance of PTQ4DiT and increased Aesthetic Quality by 1.4. This demonstrates the improvement of SDS-constructed datasets under different optimization frameworks. From optimization perspective, we further improved the Aesthetic Quality to 47.27 by using sparse distillation STD. This also demonstrates the effectiveness and generalization of our attention-based optimization method.

<span id="page-18-1"></span>Table 8: Performance of 4-bit weight and 6-bit activation quantization on CogVideoX-2B under VBench evaluation benchmark suite

| Method  | Imaging | Aesthetic | Motion  | Dynamic | BG       | Subject  | Scene    | Overall  |
|---------|---------|-----------|---------|---------|----------|----------|----------|----------|
|         | Quality | Quality   | Smooth. | Degree  | Consist. | Consist. | Consist. | Consist. |
| FP      | 58.69   | 55.25     | 97.95   | 50.00   | 96.40    | 94.30    | 33.79    | 25.91    |
| PTQ4DiT | 42.91   | 45.49     | 98.48   | 5.56    | 95.65    | 92.85    | 17.88    | 21.15    |
| +SDS    | 43.06   | 46.89     | 98.64   | 11.11   | 95.79    | 93.33    | 18.10    | 22.27    |
| +STD    | 43.08   | 47.27     | 98.78   | 9.72    | 95.97    | 93.68    | 19.04    | 22.09    |

