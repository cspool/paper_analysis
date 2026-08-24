# B. Ratio Trajectory Analysis for the T2I Task

In Sec. 3.3, we visualized the ratio trajectory for inpainting tasks trained with our proposed layer-wise DiffCR. Here, we also provide the training trajectory of compression ratios for all layers during fine-tuning of a PixArt-Σ model on a T2I task, as shown in Fig. [8](#page-11-0) (a-c). The visualization consistently reveals that: (1) Each layer learns its unique compression ratio, with redundant layers achieving higher compression and critical layers remaining less or entirely uncompressed; (2) The average ratio across layers gradually converges to the target ratio. In this example, with a target of 20%, the final achieved average ratio is approximately 19%, indicating a minor gap. Notably, a trade-off exists between convergence speed and generation quality: a higher MSE loss coefficient for the ratio accelerates convergence but may degrade quality due to overly rapid compression, while a smaller coefficient promotes gradual convergence

<span id="page-11-0"></span>> **[图片提取文字 (无描述)]:**
> (a) (b) 0.200 - Layer 15 Layer 1 0.200 0.175 0.150 Avg. Ratio Layer 2 Layer 16 Layer 17 Compression Ratios Layer 18 Layer 19 Layer 5 Layer 20 Compression Layer 21 Layer 22 0.125 Layer 8 Layer 23 Layer 9 Layer 24 Layer 10 0.100 Layer 11 Layer 25 Layer 12 Layer 26 0.075 Layer 13 Layer 27 Layer 14 Layer 28 0.050 0.025 0.000 0.0 0 10 20 30 40 50 60 70 80 90 100 10 20 30 40 50 60 70 80 90 100 Training Iterations (k) Training Iterations (k) Compression Ratios 0.0 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 8 9 Layers
![](_page_11_Figure_6.jpeg)

Figure 8. Visualization of the compression ratio trajectory during fine-tuning for a T2I task: (a) Trajectories for each of the 28 layers in the PixArt-Σ model; (b) Average ratio trajectory across all layers; and (c) The final learned ratio distribution across 28 layers.

and maintains quality, albeit with slower training. In practice, we set the initial coefficient to 0.3 and dynamically adjust it during training to balance speed and quality effectively; (3) The middle layers exhibit greater redundancy, while the later layers generally have lower redundancy and often cannot be compressed. The early layers show variable redundancy levels.

Note that to prevent the model from learning 0% compression ratios across all layers, we balance diffusion loss (favoring lower ratios for higher quality) and MSE loss (driving the target average ratio) using a coefficient, without additional regularization or penalties. A higher coefficient speeds up convergence but may compromise quality, while a smaller one ensures gradual convergence and preserves quality. Some layers naturally learn 0% ratios, underscoring their importance.

