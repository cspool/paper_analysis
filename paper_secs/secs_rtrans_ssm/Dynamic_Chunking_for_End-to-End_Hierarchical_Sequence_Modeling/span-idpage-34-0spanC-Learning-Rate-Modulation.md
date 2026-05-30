# <span id="page-34-0"></span>C Learning Rate Modulation

As discussed in Section 2.3, we employ modulated learning rates for each stage by multiplying a scalar  $\lambda^s$  to the base learning rate. Empirically, we find that a reasonable set of multipliers (*e.g.*,  $\lambda^0 = 2.0$ ,  $\lambda^1 = 1.5$ ,  $\lambda^2 = 1.0$ ) works well in general. To provide a more systematic experimental results across different architectural configurations, we follow previous works and set learning rates to be proportionally to the (1) square root of batch size (Malladi et al. 2022; Merrill et al. 2025), and (2) inverse square root of hidden dimension (Vaswani et al. 2017; Yang and Hu 2020). Concretely, without heavy manual tuning, we define  $\lambda^s$  as follows:

$$\lambda^{s} = \sqrt{N^{\text{GPT}} \cdot \frac{\prod_{i=s}^{S} N^{i}}{\prod_{i=0}^{S} N^{i}} \cdot \frac{D^{S}}{D^{s}}}, \qquad N^{S} = 1.0$$

$$(11)$$

<span id="page-35-2"></span>![](_page_35_Figure_0.jpeg)

Figure 13: **Compression Methods in chunking layer.** Default: H-Net's Downsample operation **(left-a)**. Max/Mean: Channel-wise max and mean pooling within boundaries **(left-b)**. XAttn: Cross-attention pooling within boundaries **(left-c)**. +Res: Adds boundary vector residuals to compressed outputs.

where  $N^{\text{GPT}}$  is the average number of bytes per token of training dataset, which is 4.6 for the GPT-2 tokenizer on FineWeb-Edu.

We note that such principles for optimizing signal propagation as neural network hyperparameters change is an active area of research, and our scaling factors are just heuristics that can likely be improved.

