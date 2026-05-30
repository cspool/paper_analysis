# Limitations

As APB is specifically optimized to minimize the prefill time for extremely long inputs, it is less effective for processing shorter inputs, particularly those under 32K tokens. When applying APB to short inputs, the optimal distributed setting is to run the inference on a single host, as the computation can be effectively parallelized within a single GPU. In such cases, APB falls back to vanilla FLASHATTN.

