# 8 Conclusion

This paper introduces SeerAttention-R, a lightweight and flexible sparse attention framework that accelerates long decoding in reasoning models. Functioning as a plug-in gating, SeerAttention-R integrates seamlessly into existing pretrained models without modifying the original parameters, requiring only a lightweight training phase for its new gating parameters. SeerAttention-R maintains near-lossless reasoning accuracy in a post-training setting, even with coarse-grained attention block sizes. The highly optimized sparse decoding kernel using TileLang achieves a near-theoretical speedup at high sparsity ratios.

