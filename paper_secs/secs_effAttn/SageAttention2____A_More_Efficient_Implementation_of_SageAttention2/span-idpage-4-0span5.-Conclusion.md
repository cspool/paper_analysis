# <span id="page-4-0"></span>5. Conclusion

We introduce SageAttention2++ to further accelerate SageAttention2. We propose to utilize the faster instruction of FP8 Matmul accumulated in FP16 for the matrix multiplication of P V . Experiments show that SageAttention2++ achieves a 3.9× speedup over FlashAttention while maintaining the same attention accuracy as SageAttention2. This means SageAttention2++ can accelerate various models, including those for language, image, and video generation, with negligible end-to-end metrics loss.

