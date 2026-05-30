# <span id="page-3-0"></span>3.3. Quantization via Efficient Selective Finetuning

In this section, we introduce QuEST, an efficient finetuning method for diffusion models that can significantly boost low-bit performance with less time and memory usage. We also present the two unique properties in quantized diffusion models, which serve as the foundation for the design of our method. Fig. [1\(](#page-2-0)c) illustrates our approach.

#### 3.3.1. Data-free Efficient Network-wise Training.

We first present the general training pipeline of our method. To alleviate the need for substantial training data, we construct the calibration set in a data-free manner. By feeding random Gaussian noises x<sup>T</sup> into the full-precision model and sampling over different time steps, we can obtain the calibration data needed for finetuning the quantized model. In practice, we only have to infer the full-precision model a few times to gather the needed number of calibration samples, totaling 128 or 256 samples per time step.

As depicted in Fig. [1\(](#page-2-0)c), to overcome the quantization challenge efficiently, we update partial model weights (WTE and WA) that only account for a small subset of parameters related to the time step t. The remaining weight parameters W<sup>F</sup> are kept frozen during optimization. We also fix the weight quantization parameters during training, reducing the amount of parameters that need to be optimized. For instance, in LDM-4 [\[30\]](#page-8-2), no more than 7% of the parameters are adjusted. The choices for the weights to be finetuned will be discussed in the following sections.

The activation quantization parameters can be viewed as additional model parameters. Therefore, we further propose a network-wise training strategy. Different from quantization methods using layer-wise or block-wise reconstruction [\[19,](#page-8-4) [32\]](#page-9-1) that bind quantization parameters with their corresponding layers or blocks, we optimize all activation scaling factors together with the partial weight parameters. Additionally, while layer/block-wise optimization methods can only reconstruct sequentially, we update the required parameters at once. In this way, we significantly save the time and memory needed for quantization.

