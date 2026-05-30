# <span id="page-16-0"></span>A4 TRAINING TIME

As shown in Table [A12,](#page-16-4) we report the training time of the proposed OmniQuant within the LLaMA family. Note that for LLaMA, we only activate learnable weight clipping for weight-only quantization. Therefore, the training time for weight-only quantization is shorter relative to weight-activation quantization, given the fewer learnable parameters involved. While our proposed method necessitates a training time that is approximately 5× greater than GPTQ, it remains markedly faster than QAT methods, which demand hundreds of GPU hours.

