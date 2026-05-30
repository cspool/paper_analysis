# 5 Conclusion

In this paper, we present IFMoE, an inference framework designed for fine-grained Mixture of Experts (MoE) models. By redesigning the parallelism mechanism and employing an MoE model with fewer experts as a draft model, IFMoE overcomes the limitations typically seen in achieving both high throughput and low latency. While IFMoE is not a completely lossless method, it effectively maintains downstream performance while significantly improving benchmark results and delivering substantial speedups in system inference.

