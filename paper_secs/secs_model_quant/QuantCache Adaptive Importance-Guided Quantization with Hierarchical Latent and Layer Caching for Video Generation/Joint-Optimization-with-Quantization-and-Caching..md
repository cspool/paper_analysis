# Joint Optimization with Quantization and Caching.

The integration of Structural Redundancy-Aware Pruning (SRAP) with our existing quantization and caching strategies creates a *three-tier compute optimization frame-work*: (1) Hierarchical Latent Caching eliminates redundant computations across timesteps. (2) Adaptive Importance-Guided Quantization dynamically reduces numerical precision based on feature sensitivity. (3) Structural Redundancy-Aware Pruning selectively prunes layers within a timestep to prevent unnecessary overhead.

By holistically optimizing compute allocation across layers and timesteps, our method significantly accelerates DiT inference while preserving generative fidelity. This marks a substantial step forward in efficient video diffusion model and deployment for real-world applications.

<span id="page-6-2"></span><span id="page-6-0"></span>

| Method           | Bit-width (W/A) | Motion<br>Smooth. | BG.<br>Consist. | Subject<br>Consist. | Aesthetic<br>Quality | Imaging<br>Quality | Dynamic<br>Degree | Scene<br>Consist. | Overall<br>Consist. |
|------------------|-----------------|-------------------|-----------------|---------------------|----------------------|--------------------|-------------------|-------------------|---------------------|
| Open-Sora [50]   | 16/16           | 98.42             | 96.44           | 95.20               | 60.07                | 59.66              | 33.33             | 41.72             | 26.89               |
| Q-diffusion [23] | 8/8             | 96.54             | 94.47           | 92.52               | 58.00                | 56.57              | 38.88             | 38.57             | 26.33               |
| Q-DiT [2]        | 8/8             | 95.72             | 95.01           | 91.68               | 58.68                | 56.54              | 38.88             | 34.06             | 26.77               |
| PTQ4DiT [43]     | 8/8             | 98.02             | 96.33           | 96.23               | 58.40                | 53.29              | 37.50             | 36.36             | 25.98               |
| SmoothQuant [44] | 8/8             | 98.09             | 94.47           | 92.49               | 58.79                | 58.29              | 38.88             | 38.61             | 26.33               |
| Quarot [1]       | 8/8             | 97.09             | 95.34           | 90.00               | 55.96                | 56.34              | 37.50             | 37.55             | 26.09               |
| ViDiT-Q [48]     | 8/8             | 98.28             | 96.15           | 95.16               | 59.89                | 59.47              | 34.72             | 40.26             | 26.74               |
| QuantCache       | 8/8             | 98.52             | 96.12           | 94.62               | 58.57                | 55.94              | 31.94             | 36.92             | 26.97               |
| Q-DiT [2]        | 4/8             | 99.88             | 97.33           | 96.50               | 31.14                | 21.83              | 2.77              | 0.00              | 5.11                |
| PTQ4DiT [43]     | 4/8             | 94.62             | 98.50           | 98.69               | 32.76                | 35.57              | 5.56              | 3.75              | 11.76               |
| SmoothQuant [44] | 4/8             | 96.69             | 94.66           | 97.85               | 46.67                | 44.01              | 12.50             | 27.82             | 18.72               |
| Quarot [1]       | 4/8             | 94.63             | 94.55           | 99.70               | 46.04                | 41.46              | 37.50             | 29.94             | 18.91               |
| ViDiT-Q [48]     | 4/8             | 97.82             | 95.54           | 93.55               | 58.23                | 57.21              | 33.33             | 38.12             | 26.61               |
| QuantCache       | 4/6             | 98.57             | 96.34           | 94.56               | 58.63                | 55.94              | 34.72             | 39.39             | 26.77               |

Table 1. Performance comparison of various methods on VBench [12, 13]. The bit-width "16" refers to FP16 without quantization, while QuantCache-4/6 represents the version with adaptive importance-guided quantization. Due to failure to generate readable content, Q-diffusion for W4A8 is omitted. Notably, QuantCache-4/6 shows negligible loss in quality metrics compared to the baseline Open-Sora.

<span id="page-6-1"></span>

| Method           | Bit-width (W/A) | CLIPSIM | CLIP-<br>Temp | VQA-<br>Aesthetic | VQA-<br>Technical |
|------------------|-----------------|---------|---------------|-------------------|-------------------|
| Open-Sora [50]   | 16/16           | 0.1842  | 0.9983        | 62.58             | 50.18             |
| Q-DiT [2]        | 8/8             | 0.1833  | 0.9972        | 60.24             | 34.78             |
| PTQ4DiT [43]     | 8/8             | 0.1882  | 0.9986        | 53.85             | 53.03             |
| SmoothQuant [44] | 8/8             | 0.2000  | 0.9981        | 59.01             | 51.24             |
| Quarot [1]       | 8/8             | 0.1990  | 0.9971        | 57.97             | 51.99             |
| ViDiT-Q [48]     | 8/8             | 0.1999  | 0.9986        | 59.91             | 54.34             |
| QuantCache       | 8/8             | 0.1925  | 0.9989        | 60.19             | 52.39             |
| Q-DiT [2]        | 4/8             | 0.1729  | 0.9828        | 0.01              | 0.02              |
| PTQ4DiT [43]     | 4/8             | 0.1778  | 0.9968        | 2.18              | 0.32              |
| SmoothQuant [44] | 4/8             | 0.1878  | 0.9978        | 90.77             | 22.72             |
| Quarot [1]       | 4/8             | 0.1863  | 0.9960        | 46.75             | 32.95             |
| ViDiT-Q [48]     | 4/8             | 0.1854  | 0.9984        | 59.84             | 49.11             |
| QuantCache       | 4/6             | 0.1904  | 0.9981        | 59.92             | 49.14             |

Table 2. Performance comparison of various methods on CLIP and Dover. The bit-width "16" refers to FP16 without quantization, while QuantCache-4/6 represents the version with adaptive importance-guided quantization.

