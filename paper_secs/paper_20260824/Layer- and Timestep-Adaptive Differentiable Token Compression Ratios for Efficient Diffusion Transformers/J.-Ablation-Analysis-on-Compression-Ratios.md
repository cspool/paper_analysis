# J. Ablation Analysis on Compression Ratios

In this work, we target lower latency as a step toward edge deployment. To analyze the effect of varying compression ratios, we conducted an ablation study using the PixArt- $\Sigma$  model on the MS-COCO-30K dataset [18]. Notably, 1/3 of the timesteps were allocated to full-model inference to preserve accuracy. The results in the table below show that our method scales effectively to larger compression ratios, with only a slight increase in FID (<1). A 30% compression ratio was previously selected for challenging generation tasks to maintain accuracy while building upon existing state-of-the-art efficient methods.

#### K. Is MSE Loss Alone Sufficient?

We found that simply using the MSE loss effectively guides ratios toward the target without additional regularization, so we fixed it to MSE loss, but other loss functions may also work well. In addition, although we did not enforce binary prediction, the routers tend to learn a polarized distribution in some layers, separating important tokens from unimportant ones, with the learned ratios aligning accordingly, as shown in Fig. 10.