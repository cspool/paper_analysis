# <span id="page-21-1"></span>I Generative examples

Finally, we showcase several examples of how SpQR quantization affects the generated samples. For this evaluation, we take several prompts and use the compressed language model to continue generating text from these prompts. We compare the original LLaMA-65B and two quantized versions: SpQR and RTN-4bit. More specifically, we use the SpQR configuration that corresponds to near-lossless compression from Table [1.](#page-8-0) We use greedy autoregressive inference for all generated samples to ensure reproducibility. The examples in Figure [9](#page-22-0) show that all models produce a valid text, but SpQR matches the 16-bit model more frequently. The near-lossless algorithm also seems to produce more semantically similar texts.

