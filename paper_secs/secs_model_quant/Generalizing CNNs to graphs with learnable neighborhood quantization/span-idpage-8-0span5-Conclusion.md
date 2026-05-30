# <span id="page-8-0"></span>5 Conclusion

In this paper, we exploit the statistical properties of sparse CNNs and propose focused quantization to efficiently and effectively quantize model weights. The quantization strategy uses Gaussian mixture models to locate high-probability regions in the weight distributions and quantize them in fine levels. Coupled with pruning and encoding, we build a complete compression pipeline and demonstrate high compression ratios on a range of CNNs. In ResNet-18, we achieve 18*.*08× CR with minimal loss in accuracies. We additionally show FQ allows a design that is more efficient in hardware resources. Furthermore, the proposed quantization uses only powers-of-2 values and thus provides an efficient compute pattern. The significant reductions in model sizes and compute complexities can translate to direct savings in power efficiencies for future CNN accelerators on loT devices. Finally, FQ and the optimized models are fully open-source and released to the public[3](#page-8-3) .

<span id="page-8-3"></span><sup>3</sup>Available at: <https://github.com/deep-fry/mayo>.

