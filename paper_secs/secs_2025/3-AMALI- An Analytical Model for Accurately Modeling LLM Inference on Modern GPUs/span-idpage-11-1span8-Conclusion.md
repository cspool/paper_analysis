# <span id="page-11-1"></span>8 Conclusion

We have successfully constructed a GPU analytical model named AMALI to accurately predict the performance of a CUDA kernel in the context of LLM inference applications on modern GPUs. AMALI meticulously models the tensor cores and constant/instruction cache of modern GPUs when they execute LLM inference applications. Moreover, AMALI builds a multi-warp model to reflect LLM inference's unique characteristics. These techniques make AMALI a convincible as well as convenient tool to fast explore the GPU architecture design space for LLM inferences with deep insights.

