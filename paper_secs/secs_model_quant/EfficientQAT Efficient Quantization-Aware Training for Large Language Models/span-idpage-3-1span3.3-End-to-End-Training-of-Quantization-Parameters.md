# <span id="page-3-1"></span>3.3 End-to-End Training of Quantization Parameters

We further introduce the End-to-End Training of Quantization Parameters (E2E-QP), aimed at efficiently training the entire quantized model on target datasets.

End-to-End Training of step sizes. Unlike traditional Quantization-Aware Training (QAT) methods [\(Liu et al.,](#page-10-12) [2023e;](#page-10-12) [Ma et al.,](#page-10-1) [2024\)](#page-10-1) that train full-precision weights, E2E-QP begins with W<sup>q</sup> initialized via Block-AP and focuses solely on the training of quantization parameters (s and z). Our findings indicate that training s, z, or both yields similar performance (see Table [6](#page-7-1) for details). However, since training z involves converting it from

a low-bits format to full-precision, we typically train only s by default unless specified otherwise to avoid additional memory overhead.

Additionally, within E2E-QP, there is no quantization process as per Equation [\(1\)](#page-2-1); only the dequantization process occurs as described in Equation [\(2\)](#page-2-2). Thus, the gradient of the trainable parameter s is computed as ∂w<sup>b</sup> ∂s = w<sup>q</sup> − z.

Overall, the memory usage for training in E2E-QP is drastically reduced due to the reduced trainable parameter count. Detailed memory footprints for various model sizes and bits under E2E-QP are listed in Table [7.](#page-8-0) For instance, the Llama-2-70B model can complete 2-bit QAT through E2E-QP using only 34.2GB of memory. Equipped with E2E-QP, EfficientQAT is adaptable to different scenarios by simply changing the training datasets, which includes applications such as continual pre-training and instruction-tuning [\(Taori et al.,](#page-11-13) [2023\)](#page-11-13).

