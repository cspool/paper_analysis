# <span id="page-19-0"></span>G. Statistics of Training Cost

Table [10](#page-19-2) presents additional statistics for the training costs when using RoSTE and other benchmark algorithms. We observe that while achieving better performance, RoSTE requires only similar amount of computation costs compared to benchmarked algorithms.

<span id="page-19-2"></span>Table 10. Training time and peak GPU memory consumption for obtaining a quantized fine-tuned Qwen2.5 7B from its pre-trained checkpoint on a server of 8 × A100.

| Bit-width | Method          | Training Time (hours) | Peak Memory (GB) |
|-----------|-----------------|-----------------------|------------------|
| FP16      | SFT             | 2.1                   | 300              |
|           | LoRA (r = 64)   | 0.55                  | 173              |
|           | SFT → GPTQ      | 2.1 → 0               | 300 → 0          |
|           | SFT → QuaRot    | 2.1 → 0               | 300 → 0          |
|           | SFT → SpinQuant | 2.1 → 1.3             | 300 → 263        |
| W4A4KV4   | QLoRA (r = 64)  | 0.83                  | 98               |
|           | STE             | 2.4                   | 317              |
|           | RoSTE           | 2.8                   | 318              |