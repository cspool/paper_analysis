# LOGART: PUSHING THE LIMIT OF EFFICIENT LOGA-RITHMIC POST-TRAINING QUANTIZATION

Jiawei Xu<sup>1</sup> , Yi Zheng<sup>2</sup> , Chenghe Sun<sup>1</sup> , Taiyu Zhou<sup>1</sup> , Zuqi Zhang<sup>1</sup> , Jie Li<sup>3</sup> , Lirong Zheng<sup>2</sup> , Zhuo Zou<sup>2</sup>

<sup>1</sup>University of Macau, <sup>2</sup>Fudan University, <sup>3</sup>South China University of Technology jiaweixu@um.edu.mo, zhuo@fudan.edu.cn

## ABSTRACT

Efficient deployment of deep neural networks increasingly relies on Post-Training Quantization (PTQ). Logarithmic PTQ, in particular, promises multiplier-free hardware efficiency, but its performance is often limited by the nonlinear and symmetric quantization grid and standard rounding-to-nearest (RTN) approach. While learnable rounding has significantly advanced linear PTQ, its application to the non-linear and often discrete nature of logarithmic domain remains unexplored. This paper introduces learnable Logarithmic Adaptive Rounding Techniques (Log-ART) that pioneer task-aware learnable rounding specifically for the logarithmic domain. LogART further extends the learnable rounding strategy to flexibly support outlier-aware, asymmetric, and hardware-friendly dynamic logarithmic bases, determined in a distribution-aware manner using an efficient search strategy. Extensive experiments demonstrate that LogART achieves state-of-the-art accuracy while maintaining efficiency in quantizing models across various architectures and ultra-low bitwidths, outperforming existing logarithmic PTQ methods and paving the way for more effective hardware deployment. The code is available at <https://github.com/logart-lab/logart>.

