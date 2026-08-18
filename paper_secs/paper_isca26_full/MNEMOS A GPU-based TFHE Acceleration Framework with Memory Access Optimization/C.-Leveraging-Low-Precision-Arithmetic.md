# C. Leveraging Low-Precision Arithmetic

Despite the current FP64 requirement, exploring reducedprecision approaches remains a worthwhile direction for potentially accelerating TFHE on GPUs. One possible strategy is to decompose FP64 values into multiple lower-precision floating-point or fixed-point representations, which could then be processed using the higher throughput of GPU Tensor Cores for low-precision data types (e.g., FP16, INT8).

From a parameter-selection perspective, increasing the decomposition level  $\ell$  may offer a complementary path toward relaxing precision requirements. Although a larger  $\ell$  increases the number of FFT operations, it allows the decomposition

<sup>&</sup>lt;sup>3</sup>The original formula in [1] uses  $\beta$  to denote the decomposition base, which corresponds to  $2^{\beta}$  under the notation of our paper. For consistency, we adapt the formula to match our notation.

base parameter β to be reduced. Since the influence of ℓ on the noise variance is linear, whereas the impact of β is exponential (as 2 2β in the noise formula), appropriately increasing ℓ while decreasing β could, in certain configurations, relax the overall noise budget and potentially reduce the number of splits required in a precision-decomposition scheme.

