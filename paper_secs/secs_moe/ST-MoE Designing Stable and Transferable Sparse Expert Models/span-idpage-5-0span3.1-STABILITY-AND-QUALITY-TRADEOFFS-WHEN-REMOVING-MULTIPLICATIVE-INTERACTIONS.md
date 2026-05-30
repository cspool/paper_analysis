# <span id="page-5-0"></span>3.1 STABILITY AND QUALITY TRADEOFFS WHEN REMOVING MULTIPLICATIVE INTERACTIONS

Some architectural improvements involve more multiplications than additions or do not sum many items at once. For example, a matrix multiplication has one multiplication for each addition and hence we do not refer to it as a "multiplicative" operation. We present and analyze the impact of two instances of multiplicative interactions in Transformers here.

**GELU Gated Linear Units (GEGLU).** Our first example is the Gated Linear Unit (Dauphin et al., 2017) which is a component-wise product of two linear projections, one of which is first passed through a sigmoid function. Shazeer (2020) extends this to other variants and presents a GELU-Linear (Hendrycks and Gimpel, 2016) FFN layer as a replacement the usual ReLU (Nair and Hinton, 2010) FFN in Transformer.

$$FFN_{GEGLU}(x, W, V, b, c) = GELU(xW + b) \odot (xV + c)$$
(3)

This quality gain was corroborated in later work (Narang et al., 2021).

Root Mean Square Scale Parameters. Our second example is the scale parameter in root mean square (RMS) normalization (Zhang and Sennrich, 2019). Within the Transformer, rather than calling layers back-to-back, there is an internal structure (referred to as sublayer calls) which improve gradient propagation and training dynamics. Our sublayer calls match that of Raffel et al. (2019) and consist of: (1) RMS normalization, (2) layer call (e.g. Self Attention), (3) dropout (Srivastava et al., 2014), (4) add residual (He et al., 2015). RMS normalization scales the input vector  $x \in \mathbb{R}^d$  element-wise per the root-mean-square. It then rescales the output element-wise by multiplying with a learned scale parameter g.

$$y_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}} \cdot g_i \tag{4}$$

Table 2 shows that both removing GEGLU layers or the RMS scale parameter improves stability, but at a significant loss to model quality. We note that these scale parameters (g) have a disproportionate gain to model quality versus parameters elsewhere (e.g. FFN). In line with our findings, Shleifer et al. (2021) found adding a learned multiplicative scalar to the residual connection in Transformers made them much more unstable.

In Appendix C, we further study the quality impact of adding new multiplicative interactions in expert layers. We find that this operation yields quality improvements with virtually no slow-down in model step time.

#### <span id="page-5-1"></span>3.2 STABILITY AND QUALITY TRADEOFFS WHEN ADDING NOISE

We next explore a hypothesis that adding noise into the model can improve training stability (Nee-lakantan et al., 2015). Taleb (2012) argues that certain systems exhibit the property of anti-fragility, where they *improve* through noise. Inspired by the concept and by our observation that fine-tuning

<span id="page-6-1"></span>

| Method                       | Fraction Stable | Quality (†)         |
|------------------------------|-----------------|---------------------|
| Baseline                     | 4/6             | <b>-1.755</b> ±0.02 |
| Remove GEGLU                 | 3/3             | $-1.849 \pm 0.02$   |
| Remove RMS Norm. Scale Param | 3/3             | $-2.020 \pm 0.06$   |

Table 2: **Removing operations with more multiplicative interactions**. Multiplicative interactions improve quality, but can destabilize training. Individually removing two sources of multiplicative components improves the stability, but worsens quality significantly. When we remove the GEGLU layer, we replace it with with an equivalent Dense-ReLU-Dense layer to match the FLOPs and parameters.

(which injects noise via dropout) was rarely unstable, we examined whether training noise might improve the stability of sparse models. Table 3 shows a stability improvement versus the baseline, but at the expense of lower quality. We also find that input-jitter, introduced by Fedus et al. (2021), diminishes quality at XL-scale, hence we ablate it in our models. Input-jitter multiplies the input logits to the router by a uniform random variable between  $[1-10^{-2},1+10^{-2}]$ . Dropout in our ablation is applied throughout the Transformer. As seen previously, improvements in small-scale settings may fail to generalize when scaled up and therefore trends should always be monitored and re-assessed at increasing scale (Kaplan et al., 2020).

<span id="page-6-2"></span>

| Method                   | Fraction Stable | Quality (†)              |
|--------------------------|-----------------|--------------------------|
| Baseline                 | 4/6             | <b>-1.755</b> $\pm 0.02$ |
| Input jitter $(10^{-2})$ | 3/3             | $-1.777 \pm 0.03$        |
| Dropout (0.1)            | 3/3             | $-1.822 \pm 0.11$        |

Table 3: **Injecting noise during training**. Both input-jitter and dropout improve stability, but lead to a significant loss of model quality. There is a clear tradeoff with most methods: when one improves stability, it then typically decreases model quality. Our work aims to find methods that fix stability without hurting quality.

