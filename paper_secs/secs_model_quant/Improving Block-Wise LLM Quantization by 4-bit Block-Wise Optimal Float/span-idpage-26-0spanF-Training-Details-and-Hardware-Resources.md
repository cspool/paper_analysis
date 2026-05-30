# <span id="page-26-0"></span>F Training Details and Hardware Resources

Our hyperparameter choices align closely with those used by Dettmers et al. during the original evaluation of the QLoRA method [\[5\]](#page-9-2). We use the AdamW optimizer [\[27\]](#page-10-5) with a constant learning rate of 4 · 10<sup>−</sup><sup>5</sup> , configured with the exponential decay rates β<sup>1</sup> = 0.9 and β<sup>2</sup> = 0.999. We perform supervised fine-tuning for 1875 steps using batch size 16. Furthermore, we use gradient clipping with a max\_grad\_norm parameter of 0.3. A dropout with a 10% dropout rate is applied to the LoRA layers. In contrast to Dettmers et al. [\[5\]](#page-9-2), we do not perform double qunaitzation, i.e., the quantization constants are not further quantized.

All fine-tuning runs were conducted on a single A100 40GB GPU. Each run finished in less than 8 hours. For perplexity and accuracy evaluations, either an NVIDIA RTX 3080 with 10GB of memory or an A100 40GB was used.

