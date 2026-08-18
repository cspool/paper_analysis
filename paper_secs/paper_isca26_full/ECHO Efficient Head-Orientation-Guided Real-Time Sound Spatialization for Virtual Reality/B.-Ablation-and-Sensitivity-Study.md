# *B. Ablation and Sensitivity Study*

In Hybrid mode, ECHO interleaves MI and IO steps, with IO estimation conditioned on MI outputs. As a result, Hybrid performance is closely tied to MI mode performance. To isolate the contribution of each component, we conduct ablation and sensitivity studies in MI mode and report results averaged over the AEA and TUM VI datasets.

We first evaluate the individual impact of two key modules in our design: low-precision quantization and point filtering. Table [IV](#page-8-1) reports the results, where *FP16* and *FP32* replace the INT4–FP8 low-precision module with higherprecision variants. *No F* indicates the variant in which the point filtering module is completely removed. In contrast, *QAF* keeps only the quantization-aware point filtering while turning off both the stability check and the selective sampling mechanisms. Replacing the low-precision module with *FP16* or *FP32* produces only minor RRE gains with negligible impact on ATE, confirming that the low-precision module in Section [III-B](#page-3-2) introduces minimal accuracy loss. Removing filtering (*No F*) allows excessive noisy correspondences into the optimization, causing instability and ultimately no valid pose output. The *QAF* variant, which retains only quantizationaware point filtering, avoids complete failure but still degrades ATE and RRE, as the lack of stability check and selective sampling allows residual outliers to accumulate and cause drift. These results show that quantization-aware point filtering is essential to avoid divergence, while stability check and selective sampling further enhance robustness and accuracy, making the full filtering module necessary for pose estimation.

We further conduct a sensitivity analysis of the point filtering hyperparameters (α, β, r1, r2) introduced in Section [III-C](#page-4-3)

<span id="page-8-2"></span>TABLE V: Sensitivity analysis of hyperparameters α, β, r1, and r2. ("A" indicates ATE (m), "R" indicates RRE (◦ ), and "–" indicates tracking failure.)

(a) Sensitivity analysis of thresholds α and β.

|   | 1     | 0.5   | α<br>0.1 | 0.05  | 0.01 | 180   | 150   | β<br>120 | 90    | 60    |
|---|-------|-------|----------|-------|------|-------|-------|----------|-------|-------|
| A | 0.039 | 0.038 | 0.030    | 0.059 | –    | 0.035 | 0.036 | 0.030    | 0.034 | 0.037 |
| R | 1.207 | 1.230 | 1.153    | 1.232 | –    | 1.181 | 1.165 | 1.153    | 1.159 | 1.195 |

(b) Sensitivity analysis of ratios r<sup>1</sup> and r2.

|   | r1    |       |       |       |     |       | r2    |       |  |  |
|---|-------|-------|-------|-------|-----|-------|-------|-------|--|--|
|   | 20%   | 10%   | 5%    | 2.5%  | 80% | 60%   | 40%   | 20%   |  |  |
| A | 0.082 | 0.039 | 0.030 | 0.030 | –   | 0.051 | 0.030 | 0.029 |  |  |
| R | 1.262 | 1.194 | 1.153 | 1.112 | –   | 1.237 | 1.153 | 1.131 |  |  |

<span id="page-8-3"></span>TABLE VI: Cross-dataset generalization study results.

| Metric  | ECHO   | Combined | AEA-only | TUM-only |
|---------|--------|----------|----------|----------|
| ATE (m) | 0.0326 | 0.0332   | 0.0338   | 0.0337   |
| RRE (◦) | 1.0140 | 1.0194   | 1.0237   | 1.0206   |

to quantify their impact on pose accuracy. Table [V](#page-8-2) (a) reports the sensitivity results for thresholds α and β. Since map points with E<sup>1</sup> > α are discarded before optimization, setting α too small (e.g., 0.01) removes most correspondences and leaves insufficient geometric constraints, resulting in tracking failure. In contrast, a large α (e.g., 1) retains many map points with substantial quantization error, which propagates into pose optimization and degrades accuracy (ATE/RRE increases to 0.039/1.207). β governs the stability check based on the quantized reprojection error E<sup>2</sup> q . A small β is overly strict and rejects many otherwise usable correspondences (0.037/1.195 at β = 60), while a large β becomes too permissive and allows unstable pairs to enter optimization (0.035/1.181 at β = 180). The default setting (α=0.1, β=120) balances quantization robustness and reprojection stability, achieving the lowest ATE and RRE.

Table [V](#page-8-2) (b) further evaluates the impact of the selective sampling ratios r<sup>1</sup> and r2. Increasing r<sup>1</sup> makes selective sampling easier to trigger, which can prematurely enter the reducedwork regime and discard too many map points and their correspondences, degrading accuracy (ATE/RRE increases to 0.082/1.262 at r1=20%). In contrast, smaller values remain stable around the default (0.030/1.153 at r1=5%). r<sup>2</sup> controls pruning strength once selective sampling is triggered: an overly aggressive r<sup>2</sup> removes too many correspondences and destabilizes tracking (tracking fails at r2=80%), whereas moderate settings preserve accuracy (0.051/1.237 at r2=60% and 0.030/1.153 at r2=40%). Overall, the default setting (r1=5%, r2=40%) follows a conservative, accuracy-first heuristic and provides a good trade-off, reducing workload while maintaining near-best pose accuracy.

