# C Appendix / Future Work

Here are three main points we believe that should be further improved for IFMoE.

- The first point addresses the GroupedGEMM kernel implementation in cuBLAS. Due to version conflicts between PyTorch and CUDA, it is currently challenging to utilize the GroupedGEMM kernel provided by cuBLAS. However, with the future introduction of PyTorch supporting the CUDA 12.5 library, the application of this implementation is expected to significantly accelerate MoE inference performance.
- The second point pertains to the token acceptance process in the draft model. Currently, IFMoE accepts all tokens generated from the draft model and readjusts the KV-cache accordingly. However, in certain high-demand tasks such as code generation, not all the tokens may be acceptable. Thus, leveraging the logits during the verification and readjustment phase is critical to determine whether the model needs correction. By introducing a rollback mechanism, IFMoE should approach the language generation quality of the full model.
- The third point concerns expert dynamic selection. Our experiments indicate that the number of experts selected during inference can be flexible. We aim to explore under what circumstances we can reduce the number of experts and when full expert selection is necessary for optimal inference performance.

