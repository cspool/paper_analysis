# <span id="page-27-0"></span>D.7 INFERENCE OVERHEAD OF ADAPTERS

We investigate the inference throughput and memory usage of QWHA and CLoQ and present the results in Table [19](#page-27-1) and Table [20,](#page-27-2) respectively. The evaluation uses a prefill length of 2048 and a generation length of 64, with batch size 128. We compare the FP16 pre-trained model, a quantized model with LoRA (corresponding to CLoQ), a sparse adapter (SHiRA), and FT-based adapters including WHA (QWHA), DCA (LoCA), and DHA (SSH).

As stated in SHiRA, sparse adapters provide a slight speedup over LoRA due to their simple scatter operations, compared to the low-rank matrix multiplications in LoRA. WHA introduces an additional WHT operation, but due to the fast Hadamard kernel, its overhead remains small: only a 1.9% decrease in throughput compared to LoRA. In contrast, conventional FT-based adapters such as DCA and DHA incur substantial overhead, showing a 50.9% throughput drop. Therefore, although QWHA applies an inverse WHT during inference, its overhead is marginal compared to LoRA, whereas other FT-based adapters experience significant efficiency degradation.

Regarding the inference memory usage, QWHA reduces overall memory usage by 13.0% compared to CLoQ, while both methods use 3.04 GB for model weights. This improvement arises from the use of a sparse adapter with efficient scatter operations. In addition, because the inverse WHT in QWHA is implemented without heavy matrix multiplications, it incurs no additional memory overhead and in fact results in lower peak memory usage. Consequently, QWHA achieves more than a 10% reduction in total memory consumption.

<span id="page-27-1"></span>Table 19: Inference throughput (tokens/sec) of pretrained and quantized models of each adapters.

<span id="page-27-2"></span>

| Method                  | Pre-trained | LoRA  | Sparse | WHA   | DCA/DHA |
|-------------------------|-------------|-------|--------|-------|---------|
| Throughput (tokens/sec) | 66.7        | 188.1 | 191.9  | 184.6 | 92.4    |

Table 20: Peak memory usage (GB) for each method.

| Method            | QWHA  | CLoQ  |
|-------------------|-------|-------|
| Memory Usage (GB) | 52.68 | 59.53 |

### D.8 TRAINING CURVE

We analyzed the training loss, gradient norms, and convergence behavior throughout fine-tuning, and present the training curves and adapter gradient norms in Figure [10.](#page-27-3) We observe similar convergence behavior of QWHA and CLoQ. The gradient norms remain on a comparable scale despite QWHA's large nominal scaling factor. This is because the effective scaling factor of both QWHA and CLoQ is close to 1.0. Moreover, under the same effective scaling value, low-rank adapters involve matrix multiplications during backpropagation, which naturally downscales the gradient norms applied to their parameters. In contrast, the sparse adapter in QWHA does not undergo this process, resulting in gradient norms that remain stable and consistently about twice as large throughout training.

<span id="page-27-3"></span>![](_page_27_Figure_10.jpeg)

Figure 10: Training loss (left) and gradient norm (right) of each methods during Alpaca fine-tuning in LLaMA-3.2-3B 4-bit model.