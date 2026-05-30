# <span id="page-0-0"></span>**Binarized Diffusion Model for Image Super-Resolution**

Zheng Chen<sup>1</sup>, Haotong Qin<sup>2</sup>\*, Yong Guo<sup>3</sup>, Xiongfei Su<sup>4</sup>, Xin Yuan<sup>4</sup>, Linghe Kong<sup>1</sup>, Yulun Zhang<sup>1</sup>\*

<sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>ETH Zürich,

<sup>3</sup>Max Planck Institute for Informatics, <sup>4</sup>Westlake University

### **Abstract**

Advanced diffusion models (DMs) perform impressively in image super-resolution (SR), but the high memory and computational costs hinder their deployment. Binarization, an ultra-compression algorithm, offers the potential for effectively accelerating DMs. Nonetheless, due to the model structure and the multi-step iterative attribute of DMs, existing binarization methods result in significant performance degradation. In this paper, we introduce a novel binarized diffusion model, BI-DiffSR, for image SR. First, for the model structure, we design a UNet architecture optimized for binarization. We propose the consistent-pixel-downsample (CP-Down) and consistent-pixel-upsample (CP-Up) to maintain dimension consistent and facilitate the full-precision information transfer. Meanwhile, we design the channel-shuffle-fusion (CS-Fusion) to enhance feature fusion in skip connection. Second, for the activation difference across timestep, we design the timestep-aware redistribution (TaR) and activation function (TaA). The TaR and TaA dynamically adjust the distribution of activations based on different timesteps, improving the flexibility and representation alability of the binarized module. Comprehensive experiments demonstrate that our BI-DiffSR outperforms existing binarization methods. Code is released at: https://github.com/zhengchen1999/BI-DiffSR.

