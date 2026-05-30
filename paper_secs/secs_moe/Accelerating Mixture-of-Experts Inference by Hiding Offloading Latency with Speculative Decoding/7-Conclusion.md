# 7 Conclusion

In this paper, we present SpecMoEOff, the first MoE offloading system that leverages speculative decoding to increase hardware utilization and improve throughput. SpecMoEOff employs speculative decoding to enhance both CPU and GPU utilization, thereby improving throughput. To fully exploit the potential of speculative decoding in offloading scenarios, SpecMoEOff carefully orchestrates the execution of the target and draft models, designing dedicated CPU-based chunked attention operators, memory-conscious draft generation, and hyperparameter optimization. The experimental results demonstrate that SpecMoEOff achieves a decode throughput of 2.5× superior to the state-of-the-art MoE offloading serving system, MoE-Lightning.

