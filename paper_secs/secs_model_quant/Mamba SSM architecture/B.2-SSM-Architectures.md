# B.2 SSM Architectures

We use SSM architectures or state space neural networks (SSNN) to refer to deep neural network architectures incorporating one of the previous SSMs as a black box layer.

- GSS (Mehta et al. [2023\)](#page-20-10) was the first gated neural network architecture incorporating SSMs. It is motivated by the gated attention unit (GAU) of Hua et al. [\(2022\)](#page-19-8) and looks quite similar to our block, except with additional projections. Most importantly, its projection contracts the model dimension to reduce the state size of the SSM, while ours expands the model dimension in order to increase the state size, based on the motivation in Section [3.1.](#page-4-0)
- Mega (Ma et al. [2023\)](#page-20-2) combined the EMA simplification of S4 described above into a hybrid architecture using an efficient attention approximation.
- H3 (Dao, Fu, Saab, et al. [2023\)](#page-17-2) is motivated by combining S4 with linear attention (Katharopoulos et al. [2020\)](#page-19-4). It is the first to generalize this formulation of linear attention to more general recurrences, which is also the basis of later architectures.
- Selective S4 (J. Wang et al. [2023\)](#page-21-19) incorporates S4 as a black box to generate a binary mask which is multiplied on the input. While sharing the "selection" name, we consider this an architectural modification that is closer to architectural gating than a selection mechanism (Appendix [A\)](#page-23-1). For example, we hypothesize that it would not solve the Selective

Copying task because simply masking out the irrelevant inputs does not affect the spacing between the relevant ones (indeed, the Selective Copying task can even be viewed as coming pre-masked if the noise tokens are embedded to 0).

• RetNet (Y. Sun et al. [2023\)](#page-21-9) is also based on Linear Attention and very similar to H3, but reduces the inner S4 layer to a special case where the state dimension is = 1. Although not framed as such, its recurrence can be viewed as a special case of a linear SSM.

Its primary source of improvement is using a linear attention with large head dimension, which can be viewed as another method to perform input-dependent state expansion. Using a larger head dimension in the context of linear attention variants was first done by H3, but not extensively used since this requires a proportional amount of extra computation. RetNet avoids this with an alternate way to parallelize the computation with a variant of standard multi-head attention instead of convolutions, made feasible by their particular special case of SSMs which acts as a simple EMA.

• RWKV (B. Peng et al. [2023\)](#page-20-5) is another recent RNN designed for language modeling. It is based on AFT (attention-free Transformer (S. Zhai et al. [2021\)](#page-22-1)), another variant of linear attention. Its main "WKV" mechanism involves LTI recurrences and can be seen as the ratio of two SSMs.

We also highlight the gated attention unit (GAU) from Hua et al. [\(2022\)](#page-19-8), which was motivated by combining the Transformer's MHA and MLP blocks together and was an inspiration for our architecture (Section [3.4\)](#page-6-1) combining the H3 and MLP blocks.

