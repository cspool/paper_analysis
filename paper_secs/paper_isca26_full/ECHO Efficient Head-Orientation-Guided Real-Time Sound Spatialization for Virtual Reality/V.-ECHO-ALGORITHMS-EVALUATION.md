# V. ECHO ALGORITHMS EVALUATION

<span id="page-7-2"></span>To evaluate pose estimation accuracy, we compare ECHO with four state-of-the-art SLAM systems: ORB-SLAM3 [14], VINS-Fusion [80], HybVIO [93], and OKVIS [54]. All methods are tested on four randomly selected sequences from the AEA dataset and all six indoor sequences of the TUM VI dataset. The AEA sequences capture diverse motion patterns, while the TUM VI sequences serve as a standard benchmark for AR and VR style environments [14], [80].

ECHO is evaluated under two settings: standalone MI mode performs SLAM-based pose estimation by fusing image and IMU data at the camera frame rate, as shown in Figure 6 (b), incorporating all the optimizations described in Sections III-B and III-C. For point filtering, we fix  $\alpha = 0.1$ ,  $\beta = 120$ , and apply selective sampling by discarding  $r_2 = 40\%$  of points when the filtering ratio falls below  $r_1 = 5\%$ . Hybrid mode builds on MI mode by incorporating IO mode, adding RNNbased pose estimation between consecutive MI operations. The RNN is trained separately for each dataset using quantizationaware training. For AEA, the four chosen sequences are used for testing, with ten additional random sequences for training and validation. For TUM VI, a contiguous 20% segment from each sequence is randomly chosen for testing, with the remaining 80% reserved for training and validation. For fairness, other baselines are also evaluated under the IO mode, using an IMU integration module that propagates pose estimates and velocity at 100 Hz. Evaluation is performed with evo toolkit [37], reporting the Absolute Translation Error (ATE) in meters for global trajectory accuracy and the Relative Rotation Error (RRE) over 100 ms intervals in degrees for local drift. Both ATE and RRE are presented as root-mean-square errors (RMSE), following standard evaluation protocols [80].

## A. Accuracy Evaluation Results of ECHO

Table II shows ECHO's performance in Hybrid mode on the four AEA (AEA 1–4) and six TUM VI (TUM 1–6) sequences. We observe that ECHO achieves an average ATE of 0.033 m,

<span id="page-8-0"></span>TABLE III: MI mode average results in ATE (m)\RRE (◦ ).

| ATE\RRE | ECHO        | ORB-SLAM3 VINS-Fusion |             | HybVIO | OKVIS                   |
|---------|-------------|-----------------------|-------------|--------|-------------------------|
| Average | 0.030\1.153 | 0.029\1.129           | 0.539\1.385 |        | 0.392\1.920 0.099\1.234 |

<span id="page-8-1"></span>TABLE IV: Ablation results of quantization and point filtering. ("-" indicates tracking failure.)

| Metric  | ECHO  | Precision<br>FP16 | No F  | Filtering<br>QAF |       |
|---------|-------|-------------------|-------|------------------|-------|
| ATE (m) | 0.030 | 0.030             | 0.029 | –                | 0.042 |
| RRE (◦) | 1.153 | 1.133             | 1.133 | –                | 1.264 |

on par with ORB-SLAM3 (0.033 m), while also delivering a lower RRE (1.014° vs. 1.194°). The reduction in RRE is particularly important, as orientation errors directly undermine the MAA by constraining perceptual clustering. These results confirm the effectiveness of the lightweight RNN-based pose estimation module. Compared with VINS-Fusion, HybVIO, and OKVIS, ECHO attains substantially higher accuracy, highlighting the robustness of this hybrid design. Additionally, Table [III](#page-8-0) presents average results across the same sequences under MI mode, where ECHO attains an ATE of 0.030 m and an RRE of 1.153°, closely matching ORB-SLAM3 (0.029 m, 1.129°). Together with the Hybrid mode evaluation, these results demonstrate that ECHO consistently maintains high pose accuracy across both settings.

